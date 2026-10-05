BUILD_DIR ?= build
BACKEND ?= CPU
IMAGE_DECODER ?= STB
ALLOCATOR ?= SYSTEM

.PHONY: all configure test clean

all: configure
	cmake --build "$(BUILD_DIR)"

configure:
	cmake -S . -B "$(BUILD_DIR)" -DINFERENCE_SDK_BACKEND="$(BACKEND)" \
		-DINFERENCE_SDK_IMAGE_DECODER="$(IMAGE_DECODER)" \
		-DINFERENCE_SDK_ALLOCATOR="$(ALLOCATOR)"

test: all
	ctest --test-dir "$(BUILD_DIR)" --output-on-failure

clean:
	cmake -E rm -rf "$(BUILD_DIR)"
