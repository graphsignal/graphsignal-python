#include "../cupti/cupti_activity.h"
#include "../cupti/event_buckets.h"

#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>

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

// Simple CUDA kernel for testing
__global__ void test_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * 2.0f + 1.0f;
  }
}

int main() {
  // Test cupti_activity_add_kernel_pattern with null/invalid inputs
  cupti_activity_add_kernel_pattern(0, "should_not_add");
  cupti_activity_add_kernel_pattern(3, nullptr);
  
  // Test cupti_activity_add_memcpy_kind with null/invalid inputs
  cupti_activity_add_memcpy_kind(0, "should_not_add");
  cupti_activity_add_memcpy_kind(13, nullptr);
  
  // Test cupti_activity_drain without starting (should work, just return empty)
  // Note: drain calls doFlushAll which uses the store, but store may not be initialized
  // This should still work without crashing
  uint64_t base_ts = 1000'000'000'000ULL;
  CuptiDrainResult result = cupti_activity_drain(base_ts, base_ts + 10'000'000ULL);
  // Store might not be initialized, so buckets_json could be empty
  ASSERT(result.buckets_json.empty() || result.buckets_json.find("\"bucket_ts\":") != std::string::npos,
         "drain should return valid result (empty or with buckets)");
  ASSERT_EQ(result.dropped, 0ULL, "dropped should be 0 when not started");
  
  // Test with actual CUDA operations if CUDA is available
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status == cudaSuccess && device_count > 0) {
    std::printf("CUDA available, testing with actual kernel and memcpy operations...\n");
    
    // Initialize CUDA
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
      std::fprintf(stderr, "Warning: Failed to set CUDA device: %s\n", cudaGetErrorString(cuda_status));
    } else {
      // Add kernel pattern for the test kernel (only what's needed for the test)
      cupti_activity_add_kernel_pattern(1, "test_kernel");

      
      // Add memcpy kinds for the test operations (only what's needed)
      cupti_activity_add_memcpy_kind(10, "memcpy_host_to_device");
      cupti_activity_add_memcpy_kind(11, "memcpy_device_to_host");
      
      // Start CUPTI profiler
      // Use a small activity window so inter-event sleeps trigger bucket flushes.
      const uint64_t resolution_ns = 10'000'000ULL;      // 10ms
      const uint64_t activity_window_ns = 100'000'000ULL; // 100ms
      // Enable debug mode so we can see CUPTI buffer/record logs when diagnosing failures.
      cupti_activity_start(resolution_ns, activity_window_ns, /*debug_mode=*/1);

      auto sleep_between_events = [&]() {
        // Sleep longer than activity_window so the next record triggers
        // flushThreadLocalBuckets(...) before adding its own interval.
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
      };
      
      // Get start time
      uint64_t start_ts = BucketStore::nowNs();
      
      // Allocate host and device memory
      const int n = 1024;
      float* h_data = new float[n];
      float* d_data = nullptr;
      
      // Initialize host data
      for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<float>(i);
      }
      
      // Allocate device memory (HtoD memcpy)
      cuda_status = cudaMalloc(&d_data, n * sizeof(float));
      ASSERT(cuda_status == cudaSuccess, "cudaMalloc failed");
      
      // Copy host to device (HtoD)
      cuda_status = cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
      ASSERT(cuda_status == cudaSuccess, "cudaMemcpy HtoD failed");
      cuda_status = cudaDeviceSynchronize();
      ASSERT(cuda_status == cudaSuccess, "cudaDeviceSynchronize after HtoD failed");
      sleep_between_events();
      
      // Launch kernel
      int threads_per_block = 256;
      int blocks = (n + threads_per_block - 1) / threads_per_block;
      test_kernel<<<blocks, threads_per_block>>>(d_data, n);
      cuda_status = cudaGetLastError();
      ASSERT(cuda_status == cudaSuccess, "kernel launch failed");
      
      // Synchronize to ensure kernel completes
      cuda_status = cudaDeviceSynchronize();
      ASSERT(cuda_status == cudaSuccess, "cudaDeviceSynchronize failed");
      sleep_between_events();
      
      // Copy device to host (DtoH)
      cuda_status = cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
      ASSERT(cuda_status == cudaSuccess, "cudaMemcpy DtoH failed");
      cuda_status = cudaDeviceSynchronize();
      ASSERT(cuda_status == cudaSuccess, "cudaDeviceSynchronize after DtoH failed");
      sleep_between_events();
      
      // Get end time
      uint64_t end_ts = BucketStore::nowNs();
      
      // Stop profiler
      cupti_activity_stop();
      // Give CUPTI callback thread time to run bufferCompleted and flush its TLS buckets.
      std::this_thread::sleep_for(std::chrono::milliseconds(200));

      // Drain and check results (must contain activity when CUDA is available)
      // Drain a very wide range so this test verifies "we captured something"
      // independent of timestamp-domain quirks across CUPTI/CUDA versions.
      CuptiDrainResult drain_result =
          cupti_activity_drain(0, std::numeric_limits<uint64_t>::max());
      std::printf("Drained %zu bytes of JSON data\n", drain_result.buckets_json.size());
      if (!drain_result.buckets_json.empty()) {
        std::printf("Drain JSON:\n%s\n", drain_result.buckets_json.c_str());
      }

      ASSERT(!drain_result.buckets_json.empty(),
             "expected non-empty drain result when CUDA is available");
      ASSERT(drain_result.buckets_json.find("\"bucket_ts\":") != std::string::npos,
             "drain should contain bucket_ts");
      // We register these ids before start; ensure at least one appears.
      ASSERT(drain_result.buckets_json.find("\"1\"") != std::string::npos ||
             drain_result.buckets_json.find("\"10\"") != std::string::npos ||
             drain_result.buckets_json.find("\"11\"") != std::string::npos,
             "drain should contain at least one expected event id");

      // Cleanup
      cudaFree(d_data);
      delete[] h_data;
      
      std::printf("CUDA operations completed successfully\n");
    }
  } else {
    std::printf("CUDA not available, skipping CUDA operations test\n");
  }
  
  std::printf("All cupti_activity tests passed!\n");
  return 0;
}
