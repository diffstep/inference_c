cmake_minimum_required(VERSION 3.18)
project(InferenceSDK VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(INFERENCE_SDK_BACKEND "CPU" CACHE STRING "Compute backend: CPU, METAL, or CUDA")
set_property(CACHE INFERENCE_SDK_BACKEND PROPERTY STRINGS CPU METAL CUDA)
option(INFERENCE_SDK_BUILD_TESTS "Build standalone SDK tests" ON)
option(INFERENCE_SDK_ENABLE_TIMING "Enable detailed engine timing logs" OFF)
set(INFERENCE_SDK_IMAGE_DECODER "STB" CACHE STRING "Image/tokenizer implementation: STB or RUST")
set_property(CACHE INFERENCE_SDK_IMAGE_DECODER PROPERTY STRINGS STB RUST)
set(INFERENCE_SDK_ALLOCATOR "SYSTEM" CACHE STRING "Memory allocator: SYSTEM or MIMALLOC")
set_property(CACHE INFERENCE_SDK_ALLOCATOR PROPERTY STRINGS SYSTEM MIMALLOC)
string(TOUPPER "${INFERENCE_SDK_ALLOCATOR}" INFERENCE_SDK_ALLOCATOR)
if(NOT INFERENCE_SDK_ALLOCATOR STREQUAL "SYSTEM" AND
   NOT INFERENCE_SDK_ALLOCATOR STREQUAL "MIMALLOC")
    message(FATAL_ERROR "INFERENCE_SDK_ALLOCATOR must be SYSTEM or MIMALLOC")
endif()
string(TOUPPER "${INFERENCE_SDK_IMAGE_DECODER}" INFERENCE_SDK_IMAGE_DECODER)
if(NOT INFERENCE_SDK_IMAGE_DECODER STREQUAL "STB" AND
   NOT INFERENCE_SDK_IMAGE_DECODER STREQUAL "RUST")
    message(FATAL_ERROR "INFERENCE_SDK_IMAGE_DECODER must be STB or RUST")
endif()
string(TOUPPER "${INFERENCE_SDK_BACKEND}" INFERENCE_SDK_BACKEND)
if(NOT INFERENCE_SDK_BACKEND STREQUAL "CPU" AND
   NOT INFERENCE_SDK_BACKEND STREQUAL "METAL" AND
   NOT INFERENCE_SDK_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "INFERENCE_SDK_BACKEND must be CPU, METAL, or CUDA")
endif()

file(GLOB INFERENCE_SDK_COMMON CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.c")
set(METAL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal")
set(INFERENCE_SDK_SOURCES
    ${INFERENCE_SDK_COMMON}
    src/platform/cuda/attention_cuda_stub.c
    src/platform/cuda/tensor_compute_cuda_stub.c
    src/platform/metal/attention_metal_stub.c
    src/platform/metal/tensor_compute_metal_stub.c)

if(INFERENCE_SDK_BACKEND STREQUAL "CUDA")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "The CUDA backend is currently supported on Linux only")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(REMOVE_ITEM INFERENCE_SDK_SOURCES
        src/platform/cuda/attention_cuda_stub.c
        src/platform/cuda/tensor_compute_cuda_stub.c)
    list(APPEND INFERENCE_SDK_SOURCES
        src/platform/cuda/attention_cuda.cu
        src/platform/cuda/tensor_compute_cuda.cu)
elseif(INFERENCE_SDK_BACKEND STREQUAL "METAL")
    if(NOT APPLE)
        message(FATAL_ERROR "The Metal backend requires macOS")
    endif()
    enable_language(OBJC)
    list(REMOVE_ITEM INFERENCE_SDK_SOURCES
        src/platform/metal/attention_metal_stub.c
        src/platform/metal/tensor_compute_metal_stub.c)
    list(APPEND INFERENCE_SDK_SOURCES
        src/platform/metal/attention_metal.m
        src/platform/metal/tensor_compute_metal.m)
endif()

add_library(inference_sdk STATIC ${INFERENCE_SDK_SOURCES})
add_library(InferenceSDK::InferenceSDK ALIAS inference_sdk)
target_include_directories(inference_sdk
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/metal)
target_compile_definitions(inference_sdk PUBLIC INFERENCE_SDK_BACKEND_${INFERENCE_SDK_BACKEND}=1)

find_package(Threads REQUIRED)
target_link_libraries(inference_sdk PUBLIC Threads::Threads)
if(UNIX)
    target_link_libraries(inference_sdk PUBLIC m)
endif()
if(APPLE)
    target_link_libraries(inference_sdk PUBLIC "-framework Accelerate")
endif()
if(INFERENCE_SDK_ENABLE_TIMING)
    target_compile_definitions(inference_sdk PUBLIC TENSOR_TIMING_ENABLE)
endif()

if(INFERENCE_SDK_ALLOCATOR STREQUAL "MIMALLOC")
    target_sources(inference_sdk PRIVATE third_party/mimalloc/src/static.c)
    target_include_directories(inference_sdk PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mimalloc/include")
    target_compile_definitions(inference_sdk PRIVATE
        USE_MIMALLOC MI_STATIC_LIB MI_MALLOC_OVERRIDE)
endif()

if(INFERENCE_SDK_BACKEND STREQUAL "CUDA")
    set_target_properties(inference_sdk PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(inference_sdk PUBLIC CUDA::cudart CUDA::cublas)
elseif(INFERENCE_SDK_BACKEND STREQUAL "METAL")
    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
    file(MAKE_DIRECTORY "${METAL_OUTPUT_DIR}")
    add_custom_command(OUTPUT "${METAL_OUTPUT_DIR}/attention.air"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c
                -fmodules-cache-path="${METAL_OUTPUT_DIR}/module-cache"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/metal/attention_kernels.metal"
                -o "${METAL_OUTPUT_DIR}/attention.air"
        DEPENDS src/platform/metal/attention_kernels.metal
                vendor/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal
        VERBATIM)
    add_custom_command(OUTPUT "${METAL_OUTPUT_DIR}/attention.metallib"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
                "${METAL_OUTPUT_DIR}/attention.air" -o "${METAL_OUTPUT_DIR}/attention.metallib"
        DEPENDS "${METAL_OUTPUT_DIR}/attention.air" VERBATIM)
    add_custom_command(OUTPUT "${METAL_OUTPUT_DIR}/tensor.air"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c
                -fmodules-cache-path="${METAL_OUTPUT_DIR}/module-cache"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/metal/tensor_kernels.metal"
                -o "${METAL_OUTPUT_DIR}/tensor.air"
        DEPENDS src/platform/metal/tensor_kernels.metal VERBATIM)
    add_custom_command(OUTPUT "${METAL_OUTPUT_DIR}/tensor_kernels.metallib"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
                "${METAL_OUTPUT_DIR}/tensor.air" -o "${METAL_OUTPUT_DIR}/tensor_kernels.metallib"
        DEPENDS "${METAL_OUTPUT_DIR}/tensor.air" VERBATIM)
    add_custom_target(inference_sdk_metal_shaders ALL
        DEPENDS "${METAL_OUTPUT_DIR}/attention.metallib"
                "${METAL_OUTPUT_DIR}/tensor_kernels.metallib")
    add_dependencies(inference_sdk inference_sdk_metal_shaders)
    set_source_files_properties(src/platform/metal/attention_metal.m
        src/platform/metal/tensor_compute_metal.m PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(inference_sdk PUBLIC
        "-framework Accelerate" "-framework Foundation" "-framework Metal"
        "-framework MetalPerformanceShaders")
endif()

if(INFERENCE_SDK_IMAGE_DECODER STREQUAL "RUST")
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
    set(RUST_STATIC_LIBRARY "${RUST_TARGET_DIR}/release/libimage_decoder_ffi.a")
    set(RUST_CARGO_FEATURES)
    if(INFERENCE_SDK_BACKEND STREQUAL "METAL")
        list(APPEND RUST_CARGO_FEATURES --features embedded-metallibs)
        set(RUST_METALLIB_ENV "INFERENCE_SDK_METALLIB_DIR=${METAL_OUTPUT_DIR}")
        target_compile_definitions(inference_sdk PRIVATE EMBED_METALLIBS)
    endif()
    add_custom_command(OUTPUT "${RUST_STATIC_LIBRARY}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "CARGO_TARGET_DIR=${RUST_TARGET_DIR}"
                ${RUST_METALLIB_ENV}
                "${CARGO_EXECUTABLE}" build --manifest-path
                "${CMAKE_CURRENT_SOURCE_DIR}/rust_decoder/Cargo.toml"
                --release ${RUST_CARGO_FEATURES}
        DEPENDS rust_decoder/Cargo.toml rust_decoder/Cargo.lock rust_decoder/src/lib.rs
        VERBATIM)
    add_custom_target(inference_sdk_rust_decoder DEPENDS "${RUST_STATIC_LIBRARY}")
    if(INFERENCE_SDK_BACKEND STREQUAL "METAL")
        add_dependencies(inference_sdk_rust_decoder inference_sdk_metal_shaders)
    endif()
    add_dependencies(inference_sdk inference_sdk_rust_decoder)
    target_compile_definitions(inference_sdk PRIVATE IMAGE_DECODER_RUST)
    target_link_libraries(inference_sdk PUBLIC "${RUST_STATIC_LIBRARY}" ${CMAKE_DL_LIBS})
    if(APPLE)
        target_link_libraries(inference_sdk PUBLIC pthread c++)
    else()
        target_link_libraries(inference_sdk PUBLIC pthread m rt util gcc_s stdc++)
    endif()
endif()

if(INFERENCE_SDK_BUILD_TESTS)
    include(CTest)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/build"
                        "${CMAKE_CURRENT_BINARY_DIR}/tests/fixtures")
    configure_file(tests/fixtures/tokenizer.json
                   "${CMAKE_CURRENT_BINARY_DIR}/tests/fixtures/tokenizer.json"
                   COPYONLY)
    function(add_inference_sdk_test test_name test_source)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE InferenceSDK::InferenceSDK)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endfunction()
    add_inference_sdk_test(test_benchmark tests/test_benchmark.c)
    add_inference_sdk_test(test_json tests/test_json.c)
    add_inference_sdk_test(test_safetensors tests/test_safetensors.c)
    add_inference_sdk_test(test_tensor_buffer tests/test_tensor_buffer.c)
    add_inference_sdk_test(test_tensor_store tests/test_tensor_store.c)
    add_inference_sdk_test(test_tensor_ops tests/test_tensor_ops.c)
    add_inference_sdk_test(test_tensor_workspace tests/test_tensor_workspace.c)
    add_inference_sdk_test(test_tensor_prefetch tests/test_tensor_prefetch.c)
    add_inference_sdk_test(test_attention tests/test_attention.c)
    add_inference_sdk_test(test_tokenizer tests/test_tokenizer.c)
    if(INFERENCE_SDK_IMAGE_DECODER STREQUAL "RUST")
        target_compile_definitions(test_tokenizer PRIVATE IMAGE_DECODER_RUST)
    endif()
    add_inference_sdk_test(test_tensor_compute tests/test_tensor_compute.c)
    target_include_directories(test_tensor_compute PRIVATE include src)
    add_inference_sdk_test(test_image tests/test_image.c)
endif()

install(TARGETS inference_sdk EXPORT InferenceSDKTargets ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT InferenceSDKTargets NAMESPACE InferenceSDK:: DESTINATION lib/cmake/InferenceSDK)
