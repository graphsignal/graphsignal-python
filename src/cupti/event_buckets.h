#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct EventBucket {
  int64_t num_running = 0;
  uint64_t num_exited = 0;
  uint64_t num_errors = 0;
  uint64_t enter_offset_ns = 0;
  uint64_t exit_offset_ns = 0;
  uint64_t bytes = 0; // used for memcpy (0 for kernels)
};

struct Bucket {
  uint64_t bucket_ts = 0; // wall-clock bucket timestamp (aligned to resolution_ns)
  std::unordered_map<uint64_t, EventBucket> events;
};

class BucketStore {
public:
  void init(uint64_t resolution_ns, uint64_t activity_window_ns);
  uint64_t resolutionNs() const;

  static uint64_t nowNs();
  uint64_t alignDown(uint64_t ts_ns) const;

  void addKernelInterval(uint64_t event_id, uint64_t start_ts, uint64_t end_ts);
  void addMemcpyInterval(uint64_t event_id, uint64_t start_ts, uint64_t end_ts, uint64_t bytes);

  void flushCurrentThread(uint64_t now_ns, bool force = false);

  std::string drain(uint64_t start_ts, uint64_t end_ts);

private:
  std::string bucketToJson(const Bucket& bucket);
  void mergeBucket(Bucket& dst, const Bucket& src);
  void flushThreadLocalBuckets(struct ThreadLocalBuckets* store, uint64_t now_ns);

  mutable std::mutex mu_;
  uint64_t resolution_ns_ = 10'000'000ULL;
  std::atomic<uint64_t> activity_window_ns_{0};
  std::unordered_map<uint64_t, Bucket> buckets_;
};
