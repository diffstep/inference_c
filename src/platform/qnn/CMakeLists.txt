if(NOT ANDROID)
    message(FATAL_ERROR "QNN HTP runtime is currently supported on Android only")
endif()
if(NOT DEFINED QAIRT_SDK_ROOT OR NOT EXISTS "${QAIRT_SDK_ROOT}/include/QNN/QnnInterface.h")
    message(FATAL_ERROR "Set QAIRT_SDK_ROOT to an installed QAIRT SDK")
endif()
get_filename_component(INFERENCE_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_library(inference_sdk_qnn_htp STATIC
    qnn_runtime.cpp
    qnn_graph.cpp
    ../../common/safetensors.c
    ../../common/tensor_f16.c)
add_library(InferenceSDK::QNNHTP ALIAS inference_sdk_qnn_htp)
set_target_properties(inference_sdk_qnn_htp PROPERTIES EXPORT_NAME QNNHTP)
target_include_directories(inference_sdk_qnn_htp
    PUBLIC $<BUILD_INTERFACE:${INFERENCE_SDK_ROOT}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE "${QAIRT_SDK_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(inference_sdk_qnn_htp PUBLIC cxx_std_17)
target_compile_options(inference_sdk_qnn_htp PRIVATE -Wall -Wextra -Werror)
target_link_libraries(inference_sdk_qnn_htp PUBLIC android log dl)
