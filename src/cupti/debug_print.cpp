#include "debug_print.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

std::atomic<bool> g_debug_enabled{false};

namespace {
static inline uint64_t now_ns() {
  auto t = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t).count());
}

std::mutex g_capture_mu;
std::deque<std::pair<uint64_t, std::string>> g_capture_q;

constexpr std::size_t kCaptureMaxMessages = 1000;
constexpr std::size_t kCaptureMaxLineBytes = 2048;

static inline void capture_line(uint64_t ts, const char* s) {
  if (!s || !*s) return;
  std::lock_guard<std::mutex> lock(g_capture_mu);
  g_capture_q.emplace_back(ts, s);
  while (g_capture_q.size() > kCaptureMaxMessages) {
    g_capture_q.pop_front();
  }
}

// Escape string for JSON: \ " and control chars.
static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    if (c == '\\') out += "\\\\";
    else if (c == '"') out += "\\\"";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 32) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}
} // namespace

static void print_and_capture_v(const char* fmt, va_list ap) {
  const uint64_t ts = now_ns();

  // Capture a formatted version (best-effort, bounded). Do not strip newlines.
  char buf[kCaptureMaxLineBytes];
  buf[0] = '\0';
  {
    va_list ap2;
    va_copy(ap2, ap);
    std::vsnprintf(buf, sizeof(buf), fmt, ap2);
    va_end(ap2);
  }
  std::string line;
  line.reserve(12 + std::strlen(buf));
  line += "graphsignal: ";
  line += buf;
  line.push_back('\n'); // match stderr output behavior
  capture_line(ts, line.c_str());

  std::fputs("graphsignal: ", stderr);
  std::vfprintf(stderr, fmt, ap);
  std::fputc('\n', stderr);
}

std::string debug_drain_captured(std::size_t max_messages) {
  std::lock_guard<std::mutex> lock(g_capture_mu);
  if (g_capture_q.empty()) return "[]";

  const std::size_t n =
      (max_messages == 0)
          ? g_capture_q.size()
          : std::min<std::size_t>(max_messages, g_capture_q.size());

  std::string out;
  out.reserve(n * 120);
  out += "[";
  for (std::size_t i = 0; i < n; ++i) {
    const auto& p = g_capture_q.front();
    if (i != 0) out += ",";
    out += "{\"ts\":";
    out += std::to_string(p.first);
    out += ",\"msg\":\"";
    out += json_escape(p.second);
    out += "\"}";
    g_capture_q.pop_front();
  }
  out += "]";
  return out;
}

void debug_print(const char* fmt, ...) {
  if (!g_debug_enabled.load(std::memory_order_relaxed)) return;
  va_list ap;
  va_start(ap, fmt);
  print_and_capture_v(fmt, ap);
  va_end(ap);
}

void error_print(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  print_and_capture_v(fmt, ap);
  va_end(ap);
}

