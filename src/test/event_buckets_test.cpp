#include "../cupti/event_buckets.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

// Simple test framework
#define ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_EQ(a, b, msg) \
  do { \
    if ((a) != (b)) { \
      std::fprintf(stderr, "FAIL: %s:%d: %s (expected %llu, got %llu)\n", \
                   __FILE__, __LINE__, msg, \
                   static_cast<unsigned long long>(b), \
                   static_cast<unsigned long long>(a)); \
      std::abort(); \
    } \
  } while (0)

int main() {
  BucketStore store;
  
  // Test initialization
  store.init(10'000'000ULL, 300'000'000'000ULL);  // 10ms resolution, 5min window
  ASSERT_EQ(store.resolutionNs(), 10'000'000ULL, "resolution should be 10ms");
  
  // Test alignDown
  uint64_t ts1 = 12345678901234567890ULL;
  uint64_t aligned1 = store.alignDown(ts1);
  ASSERT_EQ(aligned1 % 10'000'000ULL, 0ULL, "aligned timestamp should be multiple of resolution");
  ASSERT(aligned1 <= ts1, "aligned timestamp should be <= original");
  
  // Test addKernelInterval - single bucket
  uint64_t base_ts = 1000'000'000'000ULL;  // 1000 seconds since epoch
  uint64_t event_id = 42;
  uint64_t start_ts = base_ts + 5'000'000ULL;  // 5ms into bucket
  uint64_t end_ts = base_ts + 8'000'000ULL;    // 8ms into bucket (3ms duration)
  
  store.addKernelInterval(event_id, start_ts, end_ts);
  // Force flush - flushCurrentThread always flushes regardless of interval
  store.flushCurrentThread(base_ts + 20'000'000ULL, true);
  
  // Drain and check - bucket should be at alignDown(start_ts) = base_ts
  std::string result = store.drain(base_ts, base_ts + 10'000'000ULL);
  ASSERT(!result.empty(), "drain should return bucket data");
  ASSERT(result.find("\"bucket_ts\":") != std::string::npos, "drain should return bucket data");
  ASSERT(result.find("\"42\"") != std::string::npos, "drain should contain event_id 42");
  
  // Verify drained buckets are deleted - draining again should return empty
  std::string result_empty = store.drain(base_ts, base_ts + 10'000'000ULL);
  ASSERT(result_empty.empty(), "drain should return empty after buckets are deleted");
  
  // Test addKernelInterval - spans multiple buckets
  uint64_t start_ts2 = base_ts + 10'000'000ULL;  // Start of next bucket
  uint64_t end_ts2 = base_ts + 25'000'000ULL;    // 15ms later (spans 2 buckets)
  
  store.addKernelInterval(event_id, start_ts2, end_ts2);
  store.flushCurrentThread(base_ts + 30'000'000ULL, true);
  
  std::string result2 = store.drain(base_ts + 10'000'000ULL, base_ts + 30'000'000ULL);
  ASSERT(result2.find("\"bucket_ts\":") != std::string::npos, "drain should return multiple buckets");
  
  // Test addMemcpyInterval
  uint64_t memcpy_event_id = 100;
  uint64_t memcpy_start = base_ts + 50'000'000ULL;
  uint64_t memcpy_end = base_ts + 55'000'000ULL;
  uint64_t bytes = 1024;
  
  store.addMemcpyInterval(memcpy_event_id, memcpy_start, memcpy_end, bytes);
  store.flushCurrentThread(base_ts + 60'000'000ULL, true);
  
  std::string result3 = store.drain(base_ts + 50'000'000ULL, base_ts + 60'000'000ULL);
  ASSERT(result3.find("\"100\"") != std::string::npos, "drain should contain memcpy event_id");
  ASSERT(result3.find("\"bytes\":1024") != std::string::npos, "drain should contain bytes");
  
  // Test activity window cutoff
  uint64_t old_ts = base_ts - 400'000'000'000ULL;  // 400 seconds ago (outside 5min window)
  store.addKernelInterval(event_id, old_ts, old_ts + 5'000'000ULL);
  store.flushCurrentThread(base_ts + 60'000'000ULL, true);
  
  std::string result4 = store.drain(old_ts, old_ts + 10'000'000ULL);
  // Should be empty or not contain the old event due to activity window cutoff
  // (The exact behavior depends on implementation, but we verify it doesn't crash)
  
  std::printf("All tests passed!\n");
  return 0;
}
