#include "cupti_activity.h"

#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "event_buckets.h"
#include "debug_print.h"

#ifndef CUPTI_CALL
#define CUPTI_CALL(call)                                                        \
  do {                                                                          \
    CUptiResult _status = (call);                                               \
    if (_status != CUPTI_SUCCESS) {                                             \
      const char *errstr = nullptr;                                             \
      cuptiGetResultString(_status, &errstr);                                   \
      error_print("CUPTI error %d (%s) at %s:%d\n",                             \
              _status, errstr ? errstr : "unknown", __FILE__, __LINE__);        \
    }                                                                           \
  } while (0)
#endif

namespace {

// Keep buffers reasonably sized so CUPTI delivers records frequently.
constexpr size_t kBufferSize = 128 * 1024;
constexpr size_t kBufferAlign = 8;

std::atomic<bool> g_running{false};
std::atomic<uint64_t> g_droppedCupti{0};
BucketStore g_store;

// CUPTI activity timestamps are not wall-clock (epoch) timestamps.
// We keep a simple timebase mapping so we can bucket/drain by wall-clock.
std::atomic<uint64_t> g_timebase_cupti_ts{0};
std::atomic<uint64_t> g_timebase_wall_ts{0};

// Configurable kernel patterns: event_id -> list of pattern strings
std::unordered_map<uint64_t, std::vector<std::string>> g_kernelPatterns;
std::mutex g_kernelPatternsMutex;

static inline const char* stristr(const char* haystack, const char* needle) {
  if (!haystack || !needle || !*needle) return nullptr;
  for (const char* h = haystack; *h; ++h) {
    const char* n = needle;
    const char* p = h;
    while (*p && *n &&
           (std::tolower((unsigned char)*p) == std::tolower((unsigned char)*n))) {
      ++p;
      ++n;
    }
    if (!*n) return h;
  }
  return nullptr;
}

static uint64_t eventIdFromKernelNameUncached(const char* name) {
  if (!name) return 0;
  std::lock_guard<std::mutex> lock(g_kernelPatternsMutex);
  for (const auto& kv : g_kernelPatterns) {
    const uint64_t event_id = kv.first;
    const auto& patterns = kv.second;
    for (const auto& pattern : patterns) {
      if (stristr(name, pattern.c_str())) {
        return event_id;
      }
    }
  }
  return 0;
}

static inline uint64_t eventIdFromKernelName(const char* name) {
  if (!name) return 0;
  struct Entry { size_t h = 0; uint64_t event_id = 0; bool used = false; };
  // Thread-local direct-mapped cache of kernel-name hash -> event_id.
  // Power-of-two size so we can use `h & (size - 1)` indexing.
  constexpr size_t kCacheSize = 4096;
  thread_local std::array<Entry, kCacheSize> cache{};

  const size_t h = std::hash<std::string_view>{}(std::string_view{name});
  Entry& e = cache[h & (cache.size() - 1)];
  if (e.used && e.h == h) return e.event_id;

  const uint64_t event_id = eventIdFromKernelNameUncached(name);
  e.h = h;
  e.event_id = event_id;
  e.used = true;
  return event_id;
}

static const char* memcpyKindToStr(CUpti_ActivityMemcpyKind k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "memcpy_host_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "memcpy_device_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "memcpy_device_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "memcpy_host_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "memcpy_peer_to_peer";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "memcpy_host_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "memcpy_array_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "memcpy_array_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "memcpy_array_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "memcpy_device_to_array";
    default: return "memcpy_other";
  }
}

static std::mutex g_memcpyKindMu;
static std::unordered_map<std::string, uint64_t> g_memcpyKindToEventId;

static uint64_t eventIdFromMemcpyKind(const char* kind_str) {
  if (!kind_str) return 0;
  std::lock_guard<std::mutex> lock(g_memcpyKindMu);
  auto it = g_memcpyKindToEventId.find(kind_str);
  if (it == g_memcpyKindToEventId.end()) return 0;
  return it->second;
}

static inline uint64_t cuptiToWallTs(uint64_t cupti_ts) {
  const uint64_t base_cupti = g_timebase_cupti_ts.load(std::memory_order_relaxed);
  const uint64_t base_wall = g_timebase_wall_ts.load(std::memory_order_relaxed);
  // Use signed delta to avoid weirdness if clocks wrap or reorder.
  const int64_t delta = static_cast<int64_t>(cupti_ts) - static_cast<int64_t>(base_cupti);
  return static_cast<uint64_t>(static_cast<int64_t>(base_wall) + delta);
}

static void doFlushAll() {
  g_store.flushCurrentThread(BucketStore::nowNs(), true);
}

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *size = kBufferSize;
  *maxNumRecords = 0;

  debug_print("cupti bufferRequested: size=%zu", *size);

  void* p = nullptr;
  const int rc = posix_memalign(&p, kBufferAlign, kBufferSize);
  if (rc != 0 || !p) {
    *buffer = nullptr;
    *size = 0;
    error_print("cupti bufferRequested: alloc FAILED rc=%d\n", rc);
    return;
  }

  *buffer = static_cast<uint8_t*>(p);
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
  debug_print("cupti bufferCompleted: size=%zu validSize=%zu streamId=%u", size, validSize, streamId);

  if (validSize > 0) {
    CUpti_Activity* record = nullptr;
    CUptiResult status;
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS && record) {
        switch (record->kind) {
          case CUPTI_ACTIVITY_KIND_KERNEL:
          case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel4*>(record);
            const uint64_t event_id = eventIdFromKernelName(k->name);

            g_store.addKernelInterval(event_id, cuptiToWallTs(k->start), cuptiToWallTs(k->end));
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY: {
            const auto* m = reinterpret_cast<const CUpti_ActivityMemcpy*>(record);
            // Newer CUPTI headers may define copyKind as a byte type, so cast to enum.
            const char* kind_str =
                memcpyKindToStr(static_cast<CUpti_ActivityMemcpyKind>(m->copyKind));
            const uint64_t event_id = eventIdFromMemcpyKind(kind_str);

            g_store.addMemcpyInterval(
                event_id, cuptiToWallTs(m->start), cuptiToWallTs(m->end), m->bytes);
            break;
          }
          default:
            break;
        }
      } else if (status != CUPTI_SUCCESS && status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
        const char* errstr = nullptr;
        cuptiGetResultString(status, &errstr);
        error_print("cuptiActivityGetNextRecord error: %s\n", errstr ? errstr : "unknown");
      }
    } while (status == CUPTI_SUCCESS);
  }

  // Count any dropped records since the previous call. Use the callback-provided
  // (context, streamId) to be compatible across CUDA 12/13.
  size_t dropped = 0;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped > 0) {
    g_droppedCupti.fetch_add(static_cast<uint64_t>(dropped));
  }

  g_store.flushCurrentThread(BucketStore::nowNs(), false);

  std::free(buffer);

  debug_print("cupti bufferCompleted: size=%zu validSize=%zu dropped=%zu", size, validSize, dropped);
}

} // namespace

void cupti_activity_start(uint64_t resolution_ns, uint64_t activity_window_ns,
                          uint32_t debug_mode) {
  g_debug_enabled.store(debug_mode != 0, std::memory_order_relaxed);

  bool expected = false;
  if (!g_running.compare_exchange_strong(expected, true)) return;

  // Establish a timebase to convert CUPTI timestamps to wall-clock timestamps.
  uint64_t cupti_now = 0;
  CUPTI_CALL(cuptiGetTimestamp(&cupti_now));
  g_timebase_cupti_ts.store(cupti_now, std::memory_order_relaxed);
  g_timebase_wall_ts.store(BucketStore::nowNs(), std::memory_order_relaxed);

  g_store.init(resolution_ns, activity_window_ns);
  g_droppedCupti.store(0);

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

  debug_print("cupti_activity_start: resolution_ns=%llu activity_window_ns=%llu debug_mode=%d\n", resolution_ns, activity_window_ns, debug_mode);
}

void cupti_activity_stop() {
  bool expected = true;
  if (!g_running.compare_exchange_strong(expected, false)) return;

  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));

  // Merge any already-received thread-local buckets.
  doFlushAll();

  debug_print("cupti_activity_stop");
}

CuptiDrainResult cupti_activity_drain(uint64_t start_ts, uint64_t end_ts) {
  const CUptiResult flush_st = cuptiActivityFlushAll(0);
  if (flush_st != CUPTI_SUCCESS && flush_st != CUPTI_ERROR_NOT_INITIALIZED) {
    const char* errstr = nullptr;
    cuptiGetResultString(flush_st, &errstr);
    error_print("cuptiActivityFlushAll error: %s\n", errstr ? errstr : "unknown");
  }

  CuptiDrainResult out;
  out.buckets_json = g_store.drain(start_ts, end_ts);
  out.dropped = g_droppedCupti.load();
  return out;
}

void cupti_activity_add_kernel_pattern(uint64_t event_id, const char* pattern) {
  if (event_id == 0 || !pattern) return;
  std::lock_guard<std::mutex> lock(g_kernelPatternsMutex);
  g_kernelPatterns[event_id].push_back(std::string(pattern));
  debug_print("cupti_activity_add_kernel_pattern: event_id=%llu pattern=%s\n", event_id, pattern);
}

void cupti_activity_add_memcpy_kind(uint64_t event_id, const char* kind_str) {
  if (event_id == 0 || !kind_str) return;
  std::lock_guard<std::mutex> lock(g_memcpyKindMu);
  g_memcpyKindToEventId[std::string(kind_str)] = event_id;
  debug_print("cupti_activity_add_memcpy_kind: event_id=%llu kind_str=%s\n", event_id, kind_str);
}

void cupti_activity_set_debug_mode(uint32_t debug_mode) {
  g_debug_enabled.store(debug_mode != 0, std::memory_order_relaxed);
}

uint32_t cupti_activity_get_debug_mode() {
  return g_debug_enabled.load(std::memory_order_relaxed) ? 1u : 0u;
}
