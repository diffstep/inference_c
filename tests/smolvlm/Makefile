ROOT := $(abspath ../..)
HERE := $(abspath .)

ifeq ($(shell uname -s),Darwin)
BACKEND ?= METAL
else
BACKEND ?= CUDA
endif
IMAGE_DECODER ?= STB
ALLOCATOR ?= SYSTEM
TIMING ?= OFF
ENABLE_GGML ?= OFF
BUILD_TYPE ?= Release
ARGS ?= --version
SDK_BUILD_DIR ?= $(HERE)/build
TARGET ?= $(HERE)/build/smolvlm
CC ?= cc
CXX ?= c++
CFLAGS ?= -O2
CPPFLAGS ?=
APP_CPPFLAGS := -I$(ROOT)/include -I$(HERE)/include
ifeq ($(TIMING),ON)
APP_CPPFLAGS += -DTENSOR_TIMING_ENABLE
endif
SDK_CFLAGS ?=
SDK_CXXFLAGS ?=
SDK_CMAKE_ARGS ?=

APP_SOURCES := \
	$(HERE)/src/app/main.c \
	$(HERE)/src/app/smolvlm_config.c \
	$(HERE)/src/app/text_model.c \
	$(HERE)/src/app/vision_model.c
SDK_LIBRARY := $(SDK_BUILD_DIR)/libinference_sdk.a
ifeq ($(IMAGE_DECODER),RUST)
RUST_LIBRARY := $(SDK_BUILD_DIR)/cargo/release/libimage_decoder_ffi.a
endif

APP_LDLIBS += -lm -pthread
ifeq ($(shell uname -s),Darwin)
APP_LDLIBS += -framework Accelerate
ifeq ($(BACKEND),METAL)
APP_LDLIBS += -framework Foundation -framework Metal -framework MetalPerformanceShaders
endif
ifeq ($(IMAGE_DECODER),RUST)
APP_LDLIBS += -lc++
endif
else ifeq ($(BACKEND),CUDA)
APP_LDLIBS += -lcudart -lcublas -lstdc++
endif

.PHONY: all configure sdk run clean help

all: $(TARGET)

configure:
	cmake -S "$(ROOT)" -B "$(SDK_BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DCMAKE_C_COMPILER="$(CC)" \
		-DCMAKE_CXX_COMPILER="$(CXX)" \
		-DCMAKE_C_FLAGS="$(SDK_CFLAGS)" \
		-DCMAKE_CXX_FLAGS="$(SDK_CXXFLAGS)" \
		-DINFERENCE_SDK_BACKEND="$(BACKEND)" \
		-DINFERENCE_SDK_METALLIB_DIR="$(SDK_BUILD_DIR)/metal" \
		-DINFERENCE_SDK_BUILD_TESTS=OFF \
		-DINFERENCE_SDK_IMAGE_DECODER="$(IMAGE_DECODER)" \
		-DINFERENCE_SDK_ALLOCATOR="$(ALLOCATOR)" \
		-DINFERENCE_SDK_ENABLE_TIMING="$(TIMING)" \
		-DINFERENCE_SDK_ENABLE_GGML="$(ENABLE_GGML)" \
		$(SDK_CMAKE_ARGS)

sdk: configure
	cmake --build "$(SDK_BUILD_DIR)" --target inference_sdk

$(TARGET): sdk $(APP_SOURCES) $(RUST_LIBRARY)
	@mkdir -p "$(dir $@)"
	$(CC) $(CPPFLAGS) $(APP_CPPFLAGS) $(CFLAGS) $(APP_SOURCES) "$(SDK_LIBRARY)" $(RUST_LIBRARY) $(LDFLAGS) $(LDLIBS) $(APP_LDLIBS) -o "$@"

run: $(TARGET)
ifeq ($(BACKEND),METAL)
ifeq ($(IMAGE_DECODER),RUST)
	"$(TARGET)" $(ARGS)

else
	TENSOR_METALLIB_PATH="$(SDK_BUILD_DIR)/metal/attention.metallib" \
	TENSOR_KERNELS_METALLIB_PATH="$(SDK_BUILD_DIR)/metal/tensor_kernels.metallib" \
	"$(TARGET)" $(ARGS)

endif
else
	"$(TARGET)" $(ARGS)
endif

clean:
	cmake -E rm -rf "$(SDK_BUILD_DIR)"

help:
	@printf '%s\n' \
	'make [BACKEND=METAL|CUDA|CPU] [IMAGE_DECODER=STB|RUST] [ALLOCATOR=SYSTEM|MIMALLOC] [TIMING=ON|OFF] [ENABLE_GGML=ON|OFF]' \
	'make SDK_CFLAGS="-O3 -march=native" SDK_CXXFLAGS="..." SDK_CMAKE_ARGS="..."' \
		'make run ARGS="--version"' \
		'make run BACKEND=METAL ARGS="config sample-model"' \
	'make clean'
