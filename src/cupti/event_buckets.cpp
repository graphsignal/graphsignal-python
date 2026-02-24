#include "event_buckets.h"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>

#include "debug_print.h"

struct ThreadLocalBuckets {
  std::unordered_map<uint64_t, Bucket> buckets;
  uint64_t last_flush_ts = 0;
};

thread_local ThreadLocalBuckets g_thread_buckets;

static ThreadLocalBuckets* getThreadLocalBuckets() {
  return &g_thread_buckets;
}

void BucketStore::init(uint64_t resolution_ns, uint64_t activity_window_ns) {
  std::lock_guard<std::mutex> g(mu_);
  resolution_ns_ = resolution_ns;
  activity_window_ns_.store(activity_window_ns, std::memory_order_relaxed);
}

uint64_t BucketStore::resolutionNs() const {
  std::lock_guard<std::mutex> g(mu_);
  return resolution_ns_;
}

uint64_t BucketStore::nowNs() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

uint64_t BucketStore::alignDown(uint64_t ts_ns) const {
  return (ts_ns / resolution_ns_) * resolution_ns_;
}

void BucketStore::addKernelInterval(uint64_t event_id, uint64_t start_ts, uint64_t end_ts) {
  if (end_ts <= start_ts || resolution_ns_ == 0) return;

  const uint64_t start_bucket = alignDown(start_ts);
  const uint64_t end_bucket = alignDown(end_ts - 1);
  const uint64_t activity_window = activity_window_ns_.load(std::memory_order_relaxed);
  const uint64_t cutoff_ts = end_ts - activity_window;
  const uint64_t cutoff_bucket = alignDown(cutoff_ts);

  auto* store = getThreadLocalBuckets();

  uint64_t bucket_ts = start_bucket;
  if (cutoff_bucket > start_bucket) {
    bucket_ts = cutoff_bucket;
  }

  while (bucket_ts <= end_bucket) {
    const uint64_t bucket_end = bucket_ts + resolution_ns_;
    auto& bucket = store->buckets[bucket_ts];
    if (bucket.bucket_ts == 0) bucket.bucket_ts = bucket_ts;
    auto& eb = bucket.events[event_id];

    if (bucket_ts == start_bucket) {
      eb.enter_offset_ns += (start_ts - bucket_ts);
    }

    if (end_ts <= bucket_end) {
      eb.exit_offset_ns += (end_ts - bucket_ts);
      eb.num_exited += 1;
      break;
    } else {
      eb.num_running += 1;
    }

    bucket_ts = bucket_end;
  }
}

void BucketStore::addMemcpyInterval(uint64_t event_id, uint64_t start_ts, uint64_t end_ts, uint64_t bytes) {
  if (end_ts <= start_ts || resolution_ns_ == 0) return;

  const uint64_t total = end_ts - start_ts;
  uint64_t remaining = bytes;

  const uint64_t start_bucket = alignDown(start_ts);
  const uint64_t end_bucket = alignDown(end_ts - 1);
  const uint64_t activity_window = activity_window_ns_.load(std::memory_order_relaxed);
  const uint64_t cutoff_ts = end_ts - activity_window;
  const uint64_t cutoff_bucket = alignDown(cutoff_ts);

  auto* store = getThreadLocalBuckets();

  uint64_t bucket_ts = start_bucket;
  if (cutoff_bucket > start_bucket) {
    bucket_ts = cutoff_bucket;
  }

  while (bucket_ts <= end_bucket) {
    const uint64_t bucket_end = bucket_ts + resolution_ns_;
    const uint64_t o_start = (start_ts > bucket_ts) ? start_ts : bucket_ts;
    const uint64_t o_end = (end_ts < bucket_end) ? end_ts : bucket_end;
    if (o_end <= o_start) {
      bucket_ts = bucket_end;
      continue;
    }

    const uint64_t overlap = (o_end - o_start);
    uint64_t add_bytes = 0;
    if (bucket_end >= end_ts) {
      add_bytes = remaining;
    } else {
      add_bytes = (bytes * overlap) / total;
      if (add_bytes > remaining) add_bytes = remaining;
    }
    remaining -= add_bytes;

    auto& bucket = store->buckets[bucket_ts];
    if (bucket.bucket_ts == 0) bucket.bucket_ts = bucket_ts;
    auto& eb = bucket.events[event_id];
    eb.bytes += add_bytes;

    if (bucket_ts == start_bucket) {
      eb.enter_offset_ns += (start_ts - bucket_ts);
    }

    if (end_ts <= bucket_end) {
      eb.exit_offset_ns += (end_ts - bucket_ts);
      eb.num_exited += 1;
      break;
    } else {
      eb.num_running += 1;
    }

    bucket_ts = bucket_end;
  }
}

void BucketStore::flushCurrentThread(uint64_t now_ns, bool force) {
  auto* store = getThreadLocalBuckets();
  if (force) {
    store->last_flush_ts = 0;
  }
  flushThreadLocalBuckets(store, now_ns);
}

std::string BucketStore::drain(uint64_t start_ts, uint64_t end_ts) {
  const uint64_t start_bucket = alignDown(start_ts);
  const uint64_t end_bucket = alignDown(end_ts - 1);

  std::string out;
  std::vector<uint64_t> to_emit;
  std::lock_guard<std::mutex> lock(mu_);
  to_emit.reserve(buckets_.size());
  for (const auto& kv : buckets_) {
    const uint64_t bucket_ts = kv.first;
    if (bucket_ts >= start_bucket && bucket_ts <= end_bucket) {
      to_emit.push_back(bucket_ts);
    }
  }
  std::sort(to_emit.begin(), to_emit.end());

  for (uint64_t bucket_ts : to_emit) {
    auto it = buckets_.find(bucket_ts);
    if (it == buckets_.end()) continue;
    if (!out.empty()) out += ",";
    out += bucketToJson(it->second);
    buckets_.erase(it);
  }
  
  if (!out.empty()) {
    debug_print("event_buckets drain: emitted=%zu json_bytes=%zu remaining_global_buckets=%zu",
                to_emit.size(),
                out.size(),
                buckets_.size());
  }

  return out;
}

std::string BucketStore::bucketToJson(const Bucket& bucket) {
  std::string j;
  j.reserve(512);
  j += "{";
  j += "\"bucket_ts\":" + std::to_string(bucket.bucket_ts) + ",";

  j += "\"events\":{";
  bool first = true;
  for (const auto& kv : bucket.events) {
    if (!first) j += ",";
    first = false;
    const auto& eb = kv.second;
    j += "\"" + std::to_string(kv.first) + "\":{";
    j += "\"num_running\":" + std::to_string(eb.num_running) + ",";
    j += "\"num_exited\":" + std::to_string(eb.num_exited) + ",";
    j += "\"num_errors\":" + std::to_string(eb.num_errors) + ",";
    j += "\"enter_offset_ns\":" + std::to_string(eb.enter_offset_ns) + ",";
    j += "\"exit_offset_ns\":" + std::to_string(eb.exit_offset_ns) + ",";
    j += "\"bytes\":" + std::to_string(eb.bytes);
    j += "}";
  }
  j += "}";

  j += "}";
  return j;
}

void BucketStore::mergeBucket(Bucket& dst, const Bucket& src) {
  for (const auto& kv : src.events) {
    auto& eb = dst.events[kv.first];
    const auto& add = kv.second;
    eb.num_running += add.num_running;
    eb.num_exited += add.num_exited;
    eb.num_errors += add.num_errors;
    eb.enter_offset_ns += add.enter_offset_ns;
    eb.exit_offset_ns += add.exit_offset_ns;
    eb.bytes += add.bytes;
  }
}

void BucketStore::flushThreadLocalBuckets(ThreadLocalBuckets* store, uint64_t now_ns) {
  const uint64_t activity_window =
      activity_window_ns_.load(std::memory_order_relaxed);
  const uint64_t flush_interval = activity_window;
  if (store->last_flush_ts != 0 &&
      (now_ns - store->last_flush_ts) < flush_interval) {
    return;
  }

  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& kv : store->buckets) {
    auto& bucket = buckets_[kv.first];
    if (bucket.bucket_ts == 0) bucket.bucket_ts = kv.first;
    mergeBucket(bucket, kv.second);
  }

  debug_print("event_buckets flushThreadLocalBuckets: merged tls_buckets=%zu global_buckets=%zu",
              store->buckets.size(),
              buckets_.size());

  store->buckets.clear();
  store->last_flush_ts = now_ns;
}
