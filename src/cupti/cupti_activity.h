#pragma once

#include <cstdint>
#include <string>

struct CuptiDrainResult {
  std::string buckets_json;
  uint64_t dropped = 0;
};

void cupti_activity_start(uint64_t resolution_ns, uint64_t activity_window_ns,
                          uint32_t debug_mode);
void cupti_activity_stop();
CuptiDrainResult cupti_activity_drain(uint64_t start_ts, uint64_t end_ts);
void cupti_activity_add_kernel_pattern(uint64_t event_id, const char* pattern);
void cupti_activity_add_memcpy_kind(uint64_t event_id, const char* kind_str);
void cupti_activity_set_debug_mode(uint32_t debug_mode);
uint32_t cupti_activity_get_debug_mode();
