ROOT := $(abspath ../..)
HERE := $(abspath .)

ifeq ($(shell uname -s),Darwin)
BACKEND ?= METAL
else
BACKEND ?= CUDA
endif
BUILD_TYPE ?= Release
IMAGE_DECODER ?= RUST
ALLOCATOR ?= SYSTEM
TIMING ?= OFF
SDK_BUILD_DIR ?= $(HERE)/build/sdk
TARGET ?= $(HERE)/build/laya
CC ?= cc
CFLAGS ?= -O3
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=
SDK_CFLAGS ?=
SDK_CXXFLAGS ?=
SDK_CMAKE_ARGS ?=
APP_CPPFLAGS := -I$(ROOT)/include -I$(HERE)/src
SDK_LIBRARY := $(SDK_BUILD_DIR)/libinference_sdk.a
RUST_LIBRARY := $(SDK_BUILD_DIR)/cargo/release/libimage_decoder_ffi.a
REPORT_TOOL := $(HERE)/build/benchmark_report

APP_SOURCES := $(HERE)/src/main.c $(HERE)/src/laya_model.c
APP_LDLIBS += -lm -pthread
ifeq ($(shell uname -s),Darwin)
APP_LDLIBS += -framework Accelerate -framework Foundation -framework Metal -framework MetalPerformanceShaders -lc++
else
APP_LDLIBS += -lcudart -lcublas -lstdc++
endif

.PHONY: all configure sdk run clean help

all: $(REPORT_TOOL) $(TARGET)

configure:
	cmake -S "$(ROOT)" -B "$(SDK_BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DCMAKE_C_COMPILER="$(CC)" \
		-DCMAKE_C_FLAGS="$(SDK_CFLAGS)" \
		-DCMAKE_CXX_FLAGS="$(SDK_CXXFLAGS)" \
		-DINFERENCE_SDK_BACKEND="$(BACKEND)" \
		-DINFERENCE_SDK_METALLIB_DIR="$(SDK_BUILD_DIR)/metal" \
		-DINFERENCE_SDK_BUILD_TESTS=OFF \
		-DINFERENCE_SDK_IMAGE_DECODER="$(IMAGE_DECODER)" \
		-DINFERENCE_SDK_ALLOCATOR="$(ALLOCATOR)" \
		-DINFERENCE_SDK_ENABLE_TIMING="$(TIMING)" \
		$(SDK_CMAKE_ARGS)

sdk: configure
	cmake --build "$(SDK_BUILD_DIR)" --target inference_sdk

$(TARGET): sdk $(APP_SOURCES)
	@mkdir -p "$(dir $@)"
	$(CC) $(CPPFLAGS) $(APP_CPPFLAGS) $(CFLAGS) $(APP_SOURCES) "$(SDK_LIBRARY)" "$(RUST_LIBRARY)" $(LDFLAGS) $(LDLIBS) $(APP_LDLIBS) -o "$@"

$(REPORT_TOOL): $(HERE)/benchmark_report.c $(ROOT)/src/common/benchmark.c $(ROOT)/include/benchmark.h
	@mkdir -p "$(dir $@)"
	$(CC) -O2 -I"$(ROOT)/include" "$(HERE)/benchmark_report.c" "$(ROOT)/src/common/benchmark.c" -lm -o "$@"

run: $(TARGET)
	"$(TARGET)" $(ARGS)

clean:
	cmake -E rm -rf "$(HERE)/build"

help:
	@printf '%s\n' \
	  'make [BACKEND=METAL|CUDA] [BUILD_TYPE=Release|Debug] [TIMING=ON|OFF]' \
	  '     [SDK_CFLAGS="..."] [SDK_CXXFLAGS="..."] [SDK_CMAKE_ARGS="..."]' \
	  'make run ARGS="<model-dir> [runs] [warmup-runs] [text|json]"' \
	  'The native model path is GPU-only; CPU fallback is disabled.'
