CC ?= cc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude -Iinference_sdk/include -Isrc -Iinference_sdk/src -Iinference_sdk/src/platform/metal
UNAME_S := $(shell uname -s)
ATTENTION_BACKEND ?= auto
IMAGE_DECODER ?= stb
ALLOCATOR ?= system
TIMING ?= 0

ifeq ($(ALLOCATOR),mimalloc)
CPPFLAGS += -DUSE_MIMALLOC -Iinference_sdk/third_party/mimalloc/include
MIMALLOC_OBJECT := build/mimalloc.o
else ifneq ($(ALLOCATOR),system)
$(error ALLOCATOR must be system or mimalloc)
endif

ifeq ($(TIMING),1)
CPPFLAGS += -DTENSOR_TIMING_ENABLE
else ifneq ($(TIMING),0)
$(error TIMING must be 0 or 1)
endif

ifeq ($(IMAGE_DECODER),rust)
CPPFLAGS += -DIMAGE_DECODER_RUST
else ifneq ($(IMAGE_DECODER),stb)
$(error IMAGE_DECODER must be stb or rust)
endif
ifneq ($(IMAGE_DECODER),stb)
RUST_IMAGE_LIBRARY := inference_sdk/rust_decoder/target/release/libimage_decoder_ffi.a
RUST_IMAGE_LIBS := $(RUST_IMAGE_LIBRARY) -ldl -lpthread
ifeq ($(UNAME_S),Darwin)
RUST_IMAGE_LIBS += -lc++
endif
endif

SOURCES := $(shell find src -type f -name '*.c' -print) $(shell find inference_sdk/src/common -type f -name '*.c' -print) inference_sdk/src/platform/cuda/attention_cuda_stub.c inference_sdk/src/platform/cuda/tensor_compute_cuda_stub.c inference_sdk/src/platform/metal/attention_metal_stub.c inference_sdk/src/platform/metal/tensor_compute_metal_stub.c
ATTENTION_METAL_BACKEND := inference_sdk/src/platform/metal/attention_metal_stub.c
ATTENTION_CUDA_BACKEND := inference_sdk/src/platform/cuda/attention_cuda_stub.c
TENSOR_METAL_BACKEND := inference_sdk/src/platform/metal/tensor_compute_metal_stub.c
TENSOR_CUDA_BACKEND := inference_sdk/src/platform/cuda/tensor_compute_cuda_stub.c
METAL_HOST_OBJECT :=
METAL_SHADER_DEPENDENCY :=
TENSOR_METAL_HOST_OBJECT :=
TENSOR_METAL_SHADER_DEPENDENCY :=
CUDA_BACKEND_OBJECTS :=
CUDA_ATTENTION_OBJECT :=
CUDA_TENSOR_OBJECT :=
CUDA_LIBS :=
METAL_FRAMEWORKS :=
PLATFORM_FRAMEWORKS :=
METAL_TEST_ENV :=
RUST_IMAGE_CARGO_ENV :=
RUST_IMAGE_CARGO_FEATURES :=
RUST_IMAGE_EXTRA_DEPS :=

ifneq ($(filter $(ATTENTION_BACKEND),auto cpu metal cuda),$(ATTENTION_BACKEND))
$(error ATTENTION_BACKEND must be auto, cpu, metal, or cuda)
endif
ifeq ($(ATTENTION_BACKEND),cuda)
ifneq ($(UNAME_S),Linux)
$(error CUDA backend currently supports Linux only)
endif
CUDA_TOOLCHAIN_AVAILABLE := $(shell test -x "$(NVCC)" && echo yes)
ifneq ($(CUDA_TOOLCHAIN_AVAILABLE),yes)
$(error CUDA backend requested but nvcc was not found at $(NVCC))
endif
SOURCES := $(filter-out inference_sdk/src/platform/cuda/attention_cuda_stub.c inference_sdk/src/platform/cuda/tensor_compute_cuda_stub.c,$(SOURCES))
CUDA_ATTENTION_OBJECT := build/attention_cuda.o
CUDA_TENSOR_OBJECT := build/tensor_compute_cuda.o
CUDA_BACKEND_OBJECTS := $(CUDA_ATTENTION_OBJECT) $(CUDA_TENSOR_OBJECT)
ATTENTION_CUDA_BACKEND := $(CUDA_ATTENTION_OBJECT)
TENSOR_CUDA_BACKEND := $(CUDA_TENSOR_OBJECT)
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++
endif

ifeq ($(UNAME_S),Darwin)
METAL_TOOLCHAIN_AVAILABLE := $(shell xcrun -sdk macosx metal --version >/dev/null 2>&1 && echo yes)
PLATFORM_FRAMEWORKS := -framework Accelerate
endif
ifeq ($(ATTENTION_BACKEND),metal)
ifneq ($(UNAME_S),Darwin)
$(error Metal backend is only available on macOS)
endif
ifneq ($(METAL_TOOLCHAIN_AVAILABLE),yes)
$(error Metal toolchain missing; install it with xcodebuild -downloadComponent MetalToolchain)
endif
ATTENTION_METAL_BACKEND := build/attention_metal.o
TENSOR_METAL_BACKEND := build/tensor_compute_metal.o
TENSOR_METAL_HOST_OBJECT := build/tensor_compute_metal.o
TENSOR_METAL_SHADER_DEPENDENCY := build/tensor_kernels.metallib
else ifeq ($(ATTENTION_BACKEND),auto)
ifeq ($(METAL_TOOLCHAIN_AVAILABLE),yes)
ATTENTION_METAL_BACKEND := build/attention_metal.o
TENSOR_METAL_BACKEND := build/tensor_compute_metal.o
TENSOR_METAL_HOST_OBJECT := build/tensor_compute_metal.o
TENSOR_METAL_SHADER_DEPENDENCY := build/tensor_kernels.metallib
else ifeq ($(UNAME_S),Darwin)
$(warning Metal toolchain unavailable; attention will use CPU fallback)
endif
endif

ifeq ($(ATTENTION_METAL_BACKEND),build/attention_metal.o)
SOURCES := $(filter-out inference_sdk/src/platform/metal/attention_metal_stub.c,$(SOURCES))
SOURCES := $(filter-out inference_sdk/src/platform/metal/tensor_compute_metal_stub.c,$(SOURCES))
METAL_HOST_OBJECT := build/attention_metal.o
METAL_SHADER_DEPENDENCY := build/attention.metallib $(TENSOR_METAL_SHADER_DEPENDENCY)
METAL_FRAMEWORKS := -framework Foundation -framework Metal -framework MetalPerformanceShaders
METAL_TEST_ENV := TENSOR_METALLIB_PATH=build/attention.metallib
endif
ifeq ($(ATTENTION_METAL_BACKEND),build/attention_metal.o)
ifeq ($(IMAGE_DECODER),rust)
CPPFLAGS += -DEMBED_METALLIBS
RUST_IMAGE_LIBRARY := inference_sdk/rust_decoder/target/metal/release/libimage_decoder_ffi.a
RUST_IMAGE_LIBS := $(RUST_IMAGE_LIBRARY) -ldl -lpthread
RUST_IMAGE_LIBS += -lc++
RUST_IMAGE_CARGO_ENV := CARGO_TARGET_DIR=inference_sdk/rust_decoder/target/metal INFERENCE_SDK_METALLIB_DIR=$(CURDIR)/build
RUST_IMAGE_CARGO_FEATURES := --features embedded-metallibs
RUST_IMAGE_EXTRA_DEPS := $(METAL_SHADER_DEPENDENCY)
endif
endif
TARGET := build/smolvlm-c
TEST_TARGET := build/test-safetensors
JSON_TEST_TARGET := build/test-json
CONFIG_TEST_TARGET := build/test-smolvlm-config
STORE_TEST_TARGET := build/test-tensor-store
BUFFER_TEST_TARGET := build/test-tensor-buffer
OPS_TEST_TARGET := build/test-tensor-ops
WORKSPACE_TEST_TARGET := build/test-tensor-workspace
PREFETCH_TEST_TARGET := build/test-tensor-prefetch
ATTENTION_TEST_TARGET := build/test-attention
TOKENIZER_TEST_TARGET := build/test-tokenizer
COMPUTE_TEST_TARGET := build/test-tensor-compute
IMAGE_TEST_TARGET := build/test-image
IMAGE_DECODE_TEST_TARGET := build/test-image-decode
BENCHMARK_TEST_TARGET := build/test-benchmark
THREAD_FLAGS ?= -pthread

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES) $(METAL_HOST_OBJECT) $(TENSOR_METAL_HOST_OBJECT) $(CUDA_BACKEND_OBJECTS) $(METAL_SHADER_DEPENDENCY) $(MIMALLOC_OBJECT) $(wildcard include/*.h) $(wildcard inference_sdk/include/*.h) $(wildcard inference_sdk/src/platform/metal/*.h) $(RUST_IMAGE_LIBRARY)
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(THREAD_FLAGS) $(SOURCES) $(METAL_HOST_OBJECT) $(TENSOR_METAL_HOST_OBJECT) $(CUDA_BACKEND_OBJECTS) $(MIMALLOC_OBJECT) $(RUST_IMAGE_LIBS) -lm $(PLATFORM_FRAMEWORKS) $(METAL_FRAMEWORKS) $(CUDA_LIBS) -o $@

build/attention_cuda.o: inference_sdk/src/platform/cuda/attention_cuda.cu inference_sdk/src/platform/cuda/cuda_common.cuh inference_sdk/src/attention_backend.h inference_sdk/include/attention.h
	@mkdir -p build
	$(NVCC) -O2 -std=c++17 -I$(CUDA_HOME)/include -Iinclude -Iinference_sdk/include -Isrc -Iinference_sdk/src -Iinference_sdk/src/platform/metal -c $< -o $@

build/tensor_compute_cuda.o: inference_sdk/src/platform/cuda/tensor_compute_cuda.cu inference_sdk/src/platform/cuda/cuda_common.cuh inference_sdk/include/tensor_compute_backend.h
	@mkdir -p build
	$(NVCC) -O2 -std=c++17 -I$(CUDA_HOME)/include -Iinclude -Iinference_sdk/include -Isrc -Iinference_sdk/src -Iinference_sdk/src/platform/metal -c $< -o $@

build/mimalloc.o: inference_sdk/third_party/mimalloc/src/static.c $(shell find inference_sdk/third_party/mimalloc/src inference_sdk/third_party/mimalloc/include -type f -print)
	@mkdir -p build
	$(CC) -O2 -DNDEBUG -std=c11 -DMI_MALLOC_OVERRIDE -Iinference_sdk/third_party/mimalloc/include -c $< -o $@

$(RUST_IMAGE_LIBRARY): inference_sdk/rust_decoder/Cargo.toml inference_sdk/rust_decoder/Cargo.lock inference_sdk/rust_decoder/src/lib.rs $(RUST_IMAGE_EXTRA_DEPS)
	$(RUST_IMAGE_CARGO_ENV) cargo build --manifest-path inference_sdk/rust_decoder/Cargo.toml --release $(RUST_IMAGE_CARGO_FEATURES)

ifeq ($(ATTENTION_METAL_BACKEND),build/attention_metal.o)
build/attention_kernels.air: inference_sdk/src/platform/metal/attention_kernels.metal ./inference_sdk/vendor/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal
	@mkdir -p build
	xcrun -sdk macosx metal -fmodules-cache-path=build/metal_module_cache -c $< -o $@

build/attention.metallib: build/attention_kernels.air
	xcrun -sdk macosx metallib $< -o $@

build/tensor_kernels.air: inference_sdk/src/platform/metal/tensor_kernels.metal
	@mkdir -p build
	xcrun -sdk macosx metal -fmodules-cache-path=build/metal_module_cache -c $< -o $@

build/tensor_kernels.metallib: build/tensor_kernels.air
	xcrun -sdk macosx metallib $< -o $@

build/attention_metal.o: inference_sdk/src/platform/metal/attention_metal.m inference_sdk/src/platform/metal/attention_metal.h inference_sdk/src/platform/metal/metal_library.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fobjc-arc -c $< -o $@

build/tensor_compute_metal.o: inference_sdk/src/platform/metal/tensor_compute_metal.m inference_sdk/include/tensor_compute_backend.h inference_sdk/src/platform/metal/metal_library.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fobjc-arc -c $< -o $@
endif

$(TEST_TARGET): inference_sdk/tests/test_safetensors.c inference_sdk/src/common/safetensors.c inference_sdk/include/safetensors.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_safetensors.c inference_sdk/src/common/safetensors.c -o $@

$(JSON_TEST_TARGET): inference_sdk/tests/test_json.c inference_sdk/src/common/json.c inference_sdk/include/json.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_json.c inference_sdk/src/common/json.c -o $@

$(CONFIG_TEST_TARGET): tests/test_smolvlm_config.c src/app/smolvlm_config.c inference_sdk/src/common/json.c include/smolvlm_config.h inference_sdk/include/json.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_smolvlm_config.c src/app/smolvlm_config.c inference_sdk/src/common/json.c -o $@


$(STORE_TEST_TARGET): inference_sdk/tests/test_tensor_store.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c inference_sdk/include/tensor_buffer.h inference_sdk/include/tensor_workspace.h inference_sdk/include/tensor_store.h inference_sdk/include/tensor_f16.h inference_sdk/include/safetensors.h inference_sdk/include/json.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_tensor_store.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c -o $@

$(BUFFER_TEST_TARGET): inference_sdk/tests/test_tensor_buffer.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/include/tensor_buffer.h inference_sdk/include/tensor_workspace.h inference_sdk/include/tensor_store.h inference_sdk/include/tensor_f16.h inference_sdk/include/safetensors.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_tensor_buffer.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c -o $@

$(OPS_TEST_TARGET): inference_sdk/tests/test_tensor_ops.c inference_sdk/src/common/tensor_ops.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/include/tensor_ops.h inference_sdk/include/tensor_buffer.h inference_sdk/include/tensor_workspace.h inference_sdk/include/tensor_store.h inference_sdk/include/tensor_f16.h inference_sdk/include/safetensors.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_tensor_ops.c inference_sdk/src/common/tensor_ops.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c -lm -o $@

$(WORKSPACE_TEST_TARGET): inference_sdk/tests/test_tensor_workspace.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c inference_sdk/include/tensor_workspace.h inference_sdk/include/tensor_buffer.h inference_sdk/include/tensor_store.h inference_sdk/include/tensor_f16.h inference_sdk/include/safetensors.h inference_sdk/include/json.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_tensor_workspace.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c -o $@

$(PREFETCH_TEST_TARGET): inference_sdk/tests/test_tensor_prefetch.c inference_sdk/src/common/tensor_prefetch.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c inference_sdk/include/tensor_prefetch.h inference_sdk/include/tensor_buffer.h inference_sdk/include/tensor_workspace.h inference_sdk/include/tensor_store.h inference_sdk/include/tensor_f16.h inference_sdk/include/safetensors.h inference_sdk/include/json.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(THREAD_FLAGS) inference_sdk/tests/test_tensor_prefetch.c inference_sdk/src/common/tensor_prefetch.c inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c -o $@

$(ATTENTION_TEST_TARGET): inference_sdk/tests/test_attention.c inference_sdk/src/common/attention.c $(ATTENTION_METAL_BACKEND) $(ATTENTION_CUDA_BACKEND) $(METAL_SHADER_DEPENDENCY) $(RUST_IMAGE_LIBRARY) inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c inference_sdk/include/attention.h inference_sdk/include/tensor_buffer.h inference_sdk/include/tensor_workspace.h inference_sdk/include/tensor_store.h inference_sdk/include/tensor_f16.h inference_sdk/include/safetensors.h inference_sdk/include/json.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(THREAD_FLAGS) inference_sdk/tests/test_attention.c inference_sdk/src/common/attention.c $(ATTENTION_METAL_BACKEND) $(ATTENTION_CUDA_BACKEND) inference_sdk/src/common/tensor_buffer.c inference_sdk/src/common/tensor_workspace.c inference_sdk/src/common/tensor_store.c inference_sdk/src/common/tensor_f16.c inference_sdk/src/common/safetensors.c inference_sdk/src/common/json.c $(RUST_IMAGE_LIBS) -lm $(PLATFORM_FRAMEWORKS) $(METAL_FRAMEWORKS) $(CUDA_LIBS) -o $@

$(TOKENIZER_TEST_TARGET): inference_sdk/tests/test_tokenizer.c inference_sdk/src/common/tokenizer.c inference_sdk/src/common/json.c inference_sdk/include/tokenizer.h inference_sdk/include/json.h $(RUST_IMAGE_LIBRARY)
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_tokenizer.c inference_sdk/src/common/tokenizer.c inference_sdk/src/common/json.c $(RUST_IMAGE_LIBS) -o $@

$(COMPUTE_TEST_TARGET): inference_sdk/tests/test_tensor_compute.c $(TENSOR_METAL_BACKEND) $(TENSOR_CUDA_BACKEND) $(METAL_HOST_OBJECT) $(METAL_SHADER_DEPENDENCY) $(RUST_IMAGE_LIBRARY) inference_sdk/include/tensor_compute_backend.h inference_sdk/include/tensor_f16.h inference_sdk/src/common/tensor_f16.c
	@mkdir -p build
	$(CC) $(CPPFLAGS) -Isrc $(CFLAGS) inference_sdk/tests/test_tensor_compute.c inference_sdk/src/common/tensor_f16.c $(TENSOR_METAL_BACKEND) $(TENSOR_CUDA_BACKEND) $(METAL_HOST_OBJECT) $(RUST_IMAGE_LIBS) $(METAL_FRAMEWORKS) $(CUDA_LIBS) -o $@

$(IMAGE_TEST_TARGET): inference_sdk/tests/test_image.c inference_sdk/src/common/image.c inference_sdk/include/image.h $(RUST_IMAGE_LIBRARY)
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_image.c inference_sdk/src/common/image.c $(RUST_IMAGE_LIBS) -lm -o $@

$(IMAGE_DECODE_TEST_TARGET): inference_sdk/tests/test_image_decode.c inference_sdk/src/common/image.c inference_sdk/include/image.h $(RUST_IMAGE_LIBRARY)
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_image_decode.c inference_sdk/src/common/image.c $(RUST_IMAGE_LIBS) -lm -o $@

$(BENCHMARK_TEST_TARGET): inference_sdk/tests/test_benchmark.c inference_sdk/src/common/benchmark.c inference_sdk/include/benchmark.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) inference_sdk/tests/test_benchmark.c inference_sdk/src/common/benchmark.c -lm -o $@

test: $(TEST_TARGET) $(JSON_TEST_TARGET) $(CONFIG_TEST_TARGET) $(STORE_TEST_TARGET) $(BUFFER_TEST_TARGET) $(OPS_TEST_TARGET) $(WORKSPACE_TEST_TARGET) $(PREFETCH_TEST_TARGET) $(ATTENTION_TEST_TARGET) $(TOKENIZER_TEST_TARGET) $(COMPUTE_TEST_TARGET) $(IMAGE_TEST_TARGET) $(BENCHMARK_TEST_TARGET)
	./$(TEST_TARGET)
	./$(JSON_TEST_TARGET)
	./$(CONFIG_TEST_TARGET)
	./$(STORE_TEST_TARGET)
	./$(BUFFER_TEST_TARGET)
	./$(OPS_TEST_TARGET)
	./$(WORKSPACE_TEST_TARGET)
	./$(PREFETCH_TEST_TARGET)
	$(METAL_TEST_ENV) ./$(ATTENTION_TEST_TARGET)
	./$(TOKENIZER_TEST_TARGET)
	TENSOR_KERNELS_METALLIB_PATH=build/tensor_kernels.metallib TENSOR_BACKEND=metal ./$(COMPUTE_TEST_TARGET)
	./$(IMAGE_TEST_TARGET)
	./$(BENCHMARK_TEST_TARGET)

clean:
	rm -rf build
