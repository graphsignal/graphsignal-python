#pragma once

#include <atomic>
#include <cstddef>
#include <string>

// Shared debug flag across the CUPTI components.
// Note: defined in debug_print.cpp (single instance across the library).
extern std::atomic<bool> g_debug_enabled;

// Debug print helper. Prints only when g_debug_enabled is true.
// Always appends a trailing newline.
void debug_print(const char* fmt, ...);

// Error print helper. Always prints and always enqueues to the drain buffer.
// Always appends a trailing newline.
void error_print(const char* fmt, ...);

// Drain captured debug_print lines (bounded in-memory buffer).
// max_messages==0 drains all currently buffered messages.
// Returns JSON array: [{"ts":<int>,"msg":"<escaped>"}, ...].
std::string debug_drain_captured(std::size_t max_messages);

