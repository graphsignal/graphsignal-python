// Always-on low-overhead CUPTI Activity collector (wall-clock time bucketing)
//
// Goals:
//  - Always-on, low-overhead: use CUPTI Activity (CONCURRENT_KERNEL + MEMCPY)
//  - Configurable resolution buckets aligned to wall-clock time (like function_profiler)
//  - Python drains buckets for a time range and they are cleaned up
//  - No per-record locking: bufferCompleted accumulates locally, then merges once
//
// C ABI for Python (ctypes):
//  - prof_start(uint64_t resolution_ns, uint64_t activity_window_ns, uint32_t debug_mode)
//  - prof_stop()
//  - prof_drain_json(uint64_t start_ts, uint64_t end_ts) // returns buckets in range
//  - prof_free(const char*)
//  - prof_add_kernel_pattern(uint64_t event_id, const char* pattern)
//  - prof_add_memcpy_kind(uint64_t event_id, const char* kind_str)
//  - prof_set_debug_mode(uint32_t debug_mode)
//  - prof_get_debug_mode() -> uint32_t
//
// JSON shape:
// {
//   "dropped_cupti": 0,
//   "buckets": [
//      {
//        "bucket_ts": 100,
//        "events": {
//          "event_id": {
//            "num_running": 0,
//            "num_exited": 1,
//            "num_errors": 0,
//            "enter_offset_ns": 123,
//            "exit_offset_ns": 456,
//            "bytes": 1024
//          }
//        }
//      },
//      ...
//   ]
// }

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cupti_activity.h"
#include "debug_print.h"

static std::string buildDrainJson(const std::string& buckets_json_items,
                                  uint64_t dropped_cupti) {
  std::string j;
  j.reserve(1024 + buckets_json_items.size());
  j += "{";
  j += "\"dropped_cupti\":" + std::to_string(dropped_cupti) + ",";
  j += "\"buckets\":[";
  j += buckets_json_items;
  j += "]";
  j += "}";
  return j;
}

extern "C" {

void prof_start(uint64_t resolution_ns, uint64_t activity_window_ns,
                uint32_t debug_mode) {
  cupti_activity_start(resolution_ns, activity_window_ns, debug_mode);
}

void prof_stop() {
  cupti_activity_stop();
}

const char* prof_drain_json(uint64_t start_ts, uint64_t end_ts) {
  const CuptiDrainResult result = cupti_activity_drain(start_ts, end_ts);
  const std::string json =
      buildDrainJson(result.buckets_json, result.dropped);

  char* out = static_cast<char*>(std::malloc(json.size() + 1));
  if (!out) return nullptr;
  std::memcpy(out, json.c_str(), json.size() + 1);
  return out;
}

const char* prof_drain_debug(uint32_t max_messages) {
  const std::string s = debug_drain_captured(static_cast<std::size_t>(max_messages));
  if (s.empty()) return nullptr;
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  if (!out) return nullptr;
  std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

void prof_free(const char* p) {
  std::free((void*)p);
}

void prof_add_kernel_pattern(uint64_t event_id, const char* pattern) {
  cupti_activity_add_kernel_pattern(event_id, pattern);
}

void prof_add_memcpy_kind(uint64_t event_id, const char* kind_str) {
  cupti_activity_add_memcpy_kind(event_id, kind_str);
}

void prof_set_debug_mode(uint32_t debug_mode) {
  cupti_activity_set_debug_mode(debug_mode);
}

uint32_t prof_get_debug_mode() {
  return cupti_activity_get_debug_mode();
}

} // extern "C"
