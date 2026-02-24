SHELL := /usr/bin/env bash

# Build the CUPTI profiler shared library used by `graphsignal.profilers.cupti_profiler`
#
# Requirements:
# - CUDA Toolkit installed (for libcupti.so; usually under CUDA_HOME)
# - CUPTI headers installed (Ubuntu: `apt-get install libcupti-dev`, usually /usr/include/cupti.h)
# - Linux is the intended target (CUPTI is not supported on macOS)
#
# Typical usage:
#   make cupti
#   make cupti-install
#
# Override CUDA_HOME if needed:
#   make cupti CUDA_HOME=/usr/local/cuda-12.3

# ---------------- Platform ----------------
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# ---------------- CUDA / CUPTI ----------------
CUDA_HOME ?= /usr/local/cuda
# CUDA major version used for the build artifacts (e.g. 12, 13).
# You can override explicitly: `make cupti CUDA_MAJOR=13 CUDA_HOME=/usr/local/cuda-13.1`
CUDA_MAJOR ?=
# On Ubuntu, `cupti.h` typically comes from `libcupti-dev` (often installed to /usr/include).
# libcupti.so typically comes from the CUDA Toolkit under CUDA_HOME.
CUDA_TARGET ?= $(if $(filter aarch64 arm64,$(UNAME_M)),sbsa-linux,x86_64-linux)

# Prefer system header from libcupti-dev; fall back to CUDA_HOME.
CUPTI_INCLUDE_DIR ?= $(if $(wildcard /usr/include/cupti.h),/usr/include,$(CUDA_HOME)/targets/$(CUDA_TARGET)/include)

# Default to CUDA_HOME targets layout; pick lib64 if present, otherwise lib.
CUPTI_LIB_DIR ?= $(shell \
	if [ -f "$(CUDA_HOME)/targets/$(CUDA_TARGET)/lib64/libcupti.so" ] || [ -f "$(CUDA_HOME)/targets/$(CUDA_TARGET)/lib64/libcupti.so.1" ]; then \
		echo "$(CUDA_HOME)/targets/$(CUDA_TARGET)/lib64"; \
	else \
		echo "$(CUDA_HOME)/targets/$(CUDA_TARGET)/lib"; \
	fi \
)

# ---------------- Toolchain ----------------
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O2 -fPIC -std=c++17
CPPFLAGS ?= -I$(CUPTI_INCLUDE_DIR)
NVCCFLAGS ?= -O2 -std=c++17 -I$(CUPTI_INCLUDE_DIR) -I$(CUDA_HOME)/include --compiler-options -fPIC

# ---------------- Build outputs ----------------
SRC := src/cupti/cupti_profiler.cpp src/cupti/cupti_activity.cpp src/cupti/event_buckets.cpp src/cupti/debug_print.cpp
BUILD_DIR := build
OUT := $(BUILD_DIR)/libgscuptiprof.so$(if $(CUDA_MAJOR),.$(CUDA_MAJOR),)

# ---------------- Tests ----------------
TEST_SRC := src/test/event_buckets_test.cpp src/cupti/event_buckets.cpp src/cupti/debug_print.cpp
TEST_OUT := $(BUILD_DIR)/event_buckets_test

TEST_CUPTI_TEST_SRC := src/test/cupti_activity_test.cpp
TEST_CUPTI_OUT := $(BUILD_DIR)/cupti_activity_test
TEST_CUPTI_TEST_OBJ := $(BUILD_DIR)/cupti_activity_test.o
TEST_CUPTI_OTHER_OBJ := $(BUILD_DIR)/cupti_activity.o $(BUILD_DIR)/event_buckets.o $(BUILD_DIR)/debug_print.o
TEST_CUPTI_LDFLAGS := --compiler-options -pthread
TEST_CUPTI_LDLIBS := -L$(CUPTI_LIB_DIR) -lcupti -ldl -lcudart

LDFLAGS ?= -shared
LDLIBS ?= -L$(CUPTI_LIB_DIR) -lcupti -ldl -pthread

TEST_LDFLAGS ?=
TEST_LDLIBS ?= -pthread

# ---------------- Packaging ----------------
# Where the .so should live inside the installed Python package
PKG_NATIVE_DIR := graphsignal/_native
PKG_ARCH := $(if $(filter aarch64 arm64,$(UNAME_M)),arm64,amd64)
PKG_OUT_DIR := $(PKG_NATIVE_DIR)/$(PKG_ARCH)-cu$(CUDA_MAJOR)
PKG_OUT := $(PKG_OUT_DIR)/libgscuptiprof.so

# ---------------- Docker buildx (prebuilt artifacts) ----------------
#
# These targets build precompiled Linux libraries for common arch/CUDA majors
# using Docker buildx. No GPU is required to *build*; only to run CUPTI tests.
#
# Output layout:
#   dist/cupti/<arch>-cu<major>/libgscuptiprof.so
#
BUILDX_PLATFORMS ?= linux/amd64,linux/arm64
BUILDX_BUILDER ?= graphsignal-builder
BUILDX_OUT_DIR ?= dist/cupti
CUDA12_VERSION ?= 12.4.1
CUDA13_VERSION ?= 13.0.0

.PHONY: all cupti cupti-install test test-event-buckets test-cupti-activity clean distclean info \
	buildx-setup cupti-buildx cupti-buildx-cu12 cupti-buildx-cu13 cupti-buildx-install

all: cupti

test: test-event-buckets test-cupti-activity

info:
	@echo "UNAME_S=$(UNAME_S)"
	@echo "UNAME_M=$(UNAME_M)"
	@echo "CUDA_TARGET=$(CUDA_TARGET)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "CUDA_MAJOR=$(CUDA_MAJOR)"
	@echo "CUPTI_INCLUDE_DIR=$(CUPTI_INCLUDE_DIR)"
	@echo "CUPTI_LIB_DIR=$(CUPTI_LIB_DIR)"
	@echo "PKG_ARCH=$(PKG_ARCH)"
	@echo "PKG_OUT=$(PKG_OUT)"
	@echo "BUILDX_PLATFORMS=$(BUILDX_PLATFORMS)"
	@echo "BUILDX_BUILDER=$(BUILDX_BUILDER)"
	@echo "BUILDX_OUT_DIR=$(BUILDX_OUT_DIR)"
	@echo "CUDA12_VERSION=$(CUDA12_VERSION)"
	@echo "CUDA13_VERSION=$(CUDA13_VERSION)"
	@echo "CXX=$(CXX)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"

$(BUILD_DIR):
	@mkdir -p "$(BUILD_DIR)"

# ---------------- Build / install ----------------
cupti: $(OUT)

$(OUT): $(SRC) | $(BUILD_DIR)
	@if [[ "$(UNAME_S)" == "Darwin" ]]; then \
		echo "CUPTI build is not supported on macOS (CUPTI is Linux/Windows only)."; \
		exit 1; \
	fi
	@if [[ -z "$(CUDA_MAJOR)" ]]; then \
		echo "Error: CUDA_MAJOR is required (e.g. make cupti CUDA_MAJOR=13 CUDA_HOME=/usr/local/cuda-13.1)"; \
		exit 1; \
	fi
	@echo "Building $@"
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -o "$@" $(SRC) $(LDLIBS)

cupti-install: $(OUT)
	@if [[ -z "$(CUDA_MAJOR)" ]]; then \
		echo "Error: CUDA_MAJOR is required (e.g. make cupti-install CUDA_MAJOR=13 CUDA_HOME=/usr/local/cuda-13.1)"; \
		exit 1; \
	fi
	@mkdir -p "$(PKG_OUT_DIR)"
	@cp -f "$(OUT)" "$(PKG_OUT)"
	@echo "Installed: $(PKG_OUT)"
	@echo "Note: runtime still requires libcupti.so to be discoverable (e.g. set LD_LIBRARY_PATH to $(CUPTI_LIB_DIR))."

# ---------------- Docker buildx ----------------
buildx-setup:
	@docker buildx inspect "$(BUILDX_BUILDER)" >/dev/null 2>&1 && docker buildx use "$(BUILDX_BUILDER)" >/dev/null 2>&1 || \
		docker buildx create --name "$(BUILDX_BUILDER)" --driver docker-container --use >/dev/null
	@docker run --privileged --rm tonistiigi/binfmt --install all
	@docker buildx inspect --bootstrap >/dev/null
	@echo "buildx is ready"

cupti-buildx: cupti-buildx-cu12 cupti-buildx-cu13

cupti-buildx-cu12:
	@mkdir -p "$(BUILDX_OUT_DIR)"
	docker buildx build \
		--builder "$(BUILDX_BUILDER)" \
		--platform "$(BUILDX_PLATFORMS)" \
		--build-arg CUDA_VERSION="$(CUDA12_VERSION)" \
		--build-arg CUDA_MAJOR=12 \
		-f Dockerfile.cupti \
		--output type=local,dest="$(BUILDX_OUT_DIR)" \
		.

cupti-buildx-cu13:
	@mkdir -p "$(BUILDX_OUT_DIR)"
	docker buildx build \
		--builder "$(BUILDX_BUILDER)" \
		--platform "$(BUILDX_PLATFORMS)" \
		--build-arg CUDA_VERSION="$(CUDA13_VERSION)" \
		--build-arg CUDA_MAJOR=13 \
		-f Dockerfile.cupti \
		--output type=local,dest="$(BUILDX_OUT_DIR)" \
		.

cupti-buildx-install:
	@set -euo pipefail; \
	shopt -s nullglob; \
	# buildx local multi-platform output may be nested under linux_<arch>/...
	paths=( \
		"$(BUILDX_OUT_DIR)"/*-cu*/libgscuptiprof.so \
		"$(BUILDX_OUT_DIR)"/linux_*/*-cu*/libgscuptiprof.so \
	); \
	real_paths=(); \
	for p in "$${paths[@]}"; do \
		[[ -f "$$p" ]] && real_paths+=( "$$p" ); \
	done; \
	paths=( "$${real_paths[@]}" ); \
	if (( $${#paths[@]} == 0 )); then \
		echo "No buildx artifacts found under $(BUILDX_OUT_DIR)/<arch>-cu<major>/libgscuptiprof.so"; \
		echo "Also checked: $(BUILDX_OUT_DIR)/linux_<arch>/<arch>-cu<major>/libgscuptiprof.so"; \
		echo "Run: make cupti-buildx (or cupti-buildx-cu12/cu13)"; \
		exit 1; \
	fi; \
	for p in "$${paths[@]}"; do \
		d="$$(basename "$$(dirname "$$p")")"; \
		mkdir -p "$(PKG_NATIVE_DIR)/$$d"; \
		cp -f "$$p" "$(PKG_NATIVE_DIR)/$$d/libgscuptiprof.so"; \
		echo "Installed: $(PKG_NATIVE_DIR)/$$d/libgscuptiprof.so"; \
	done

# ---------------- Tests ----------------
test-event-buckets: $(TEST_OUT)
	@echo "Running test: $(TEST_OUT)"
	@./$(TEST_OUT)

$(TEST_OUT): $(TEST_SRC) | $(BUILD_DIR)
	@echo "Building test: $@"
	$(CXX) $(CXXFLAGS) $(TEST_LDFLAGS) -o "$@" $(TEST_SRC) $(TEST_LDLIBS)

test-cupti-activity:
	@bash -e -c '\
		if [[ "$(UNAME_S)" == "Darwin" ]]; then \
			echo "Skipping cupti_activity test (CUPTI is Linux/Windows only)."; \
			exit 0; \
		fi; \
		if [[ ! -f "$(CUPTI_INCLUDE_DIR)/cupti.h" ]]; then \
			echo "Skipping cupti_activity test: cupti.h not found under $(CUPTI_INCLUDE_DIR)"; \
			echo "Install headers (Ubuntu: apt-get install libcupti-dev) or set CUPTI_INCLUDE_DIR."; \
			exit 0; \
		fi; \
		if [[ ! -f "$(CUPTI_LIB_DIR)/libcupti.so" && ! -f "$(CUPTI_LIB_DIR)/libcupti.so.1" ]]; then \
			echo "Skipping cupti_activity test: libcupti.so not found under $(CUPTI_LIB_DIR)"; \
			echo "Install CUDA Toolkit / CUPTI runtime or set CUDA_HOME / CUPTI_LIB_DIR."; \
			exit 0; \
		fi; \
		$(MAKE) $(TEST_CUPTI_OUT); \
		echo "Running test: $(TEST_CUPTI_OUT)"; \
		./$(TEST_CUPTI_OUT); \
	'

$(TEST_CUPTI_OUT): $(TEST_CUPTI_TEST_OBJ) $(TEST_CUPTI_OTHER_OBJ) | $(BUILD_DIR)
	@echo "Linking test: $@"
	$(NVCC) $(TEST_CUPTI_LDFLAGS) -o "$@" $(TEST_CUPTI_TEST_OBJ) $(TEST_CUPTI_OTHER_OBJ) $(TEST_CUPTI_LDLIBS)

$(TEST_CUPTI_TEST_OBJ): $(TEST_CUPTI_TEST_SRC) | $(BUILD_DIR)
	@echo "Compiling test file with nvcc: $@"
	$(NVCC) $(NVCCFLAGS) -x cu -c -o "$@" $(TEST_CUPTI_TEST_SRC)

$(BUILD_DIR)/cupti_activity.o: src/cupti/cupti_activity.cpp | $(BUILD_DIR)
	@echo "Compiling cupti_activity.cpp with g++: $@"
	@if [[ ! -f "$(CUPTI_INCLUDE_DIR)/cupti.h" ]]; then \
		echo "Error: cupti.h not found under $(CUPTI_INCLUDE_DIR)"; \
		echo "Set CUDA_HOME or CUPTI_INCLUDE_DIR to the correct path."; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o "$@" src/cupti/cupti_activity.cpp

$(BUILD_DIR)/event_buckets.o: src/cupti/event_buckets.cpp | $(BUILD_DIR)
	@echo "Compiling event_buckets.cpp with g++: $@"
	$(CXX) $(CXXFLAGS) -c -o "$@" src/cupti/event_buckets.cpp

$(BUILD_DIR)/debug_print.o: src/cupti/debug_print.cpp | $(BUILD_DIR)
	@echo "Compiling debug_print.cpp with g++: $@"
	$(CXX) $(CXXFLAGS) -c -o "$@" src/cupti/debug_print.cpp

# ---------------- Clean ----------------
clean:
	@rm -rf "$(BUILD_DIR)"

distclean: clean
	@rm -f "$(PKG_OUT)"

