#include "qnn_runtime.h"
#include "qnn_htp.h"
extern "C" {
#include "safetensors.h"
#include "tensor_f16.h"
}
#include <android/log.h>

#include <QNN/QnnBackend.h>
#include <QNN/QnnDevice.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnOpDef.h>

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

namespace {
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

void qnnLogCallback(const char* format, QnnLog_Level_t level, uint64_t, va_list args) {
    int priority = ANDROID_LOG_INFO;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR: priority = ANDROID_LOG_ERROR; break;
        case QNN_LOG_LEVEL_WARN: priority = ANDROID_LOG_WARN; break;
        case QNN_LOG_LEVEL_DEBUG: priority = ANDROID_LOG_DEBUG; break;
        case QNN_LOG_LEVEL_VERBOSE: priority = ANDROID_LOG_VERBOSE; break;
        default: break;
    }
    __android_log_vprint(priority, "SmolVLM-QNN", format, args);
}

struct Runtime {
    void* library = nullptr;
    const QnnInterface_t* provider = nullptr;
    Qnn_LogHandle_t log = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    const QnnDevice_PlatformInfo_t* platformInfo = nullptr;

    ~Runtime() {
        if (provider != nullptr) {
            const auto& api = provider->QNN_INTERFACE_VER_NAME;
            if (context != nullptr && api.contextFree != nullptr) api.contextFree(context, nullptr);
            if (device != nullptr && api.deviceFree != nullptr) api.deviceFree(device);
            if (platformInfo != nullptr && api.deviceFreePlatformInfo != nullptr)
                api.deviceFreePlatformInfo(log, platformInfo);
            if (backend != nullptr && api.backendFree != nullptr) api.backendFree(backend);
            if (log != nullptr && api.logFree != nullptr) api.logFree(log);
        }
        if (library != nullptr) dlclose(library);
    }
};

bool fail(char* message, size_t capacity, const char* text, int status = 0) {
    if (capacity != 0) {
        if (status == 0) std::snprintf(message, capacity, "%s", text);
        else std::snprintf(message, capacity, "%s (QNN status %d)", text, status);
    }
    return false;
}

bool probe(Runtime& runtime, const char* htpDirectory, unsigned smokeRuns,
           const char* modelPath,
           char* message, size_t capacity) {
    std::string adspPath = htpDirectory;
    adspPath += ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp;";
    setenv("ADSP_LIBRARY_PATH", adspPath.c_str(), 1);
    runtime.library = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (runtime.library == nullptr) {
        const char* error = dlerror();
        return fail(message, capacity, error == nullptr ? "cannot load libQnnHtp.so" : error);
    }
    auto getProviders = reinterpret_cast<GetProviders>(dlsym(runtime.library, "QnnInterface_getProviders"));
    if (getProviders == nullptr) return fail(message, capacity, "QNN provider entrypoint is missing");

    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    Qnn_ErrorHandle_t status = getProviders(&providers, &count);
    if (status != QNN_SUCCESS || providers == nullptr || count == 0)
        return fail(message, capacity, "cannot enumerate QNN HTP providers", status);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& v = providers[i]->apiVersion.coreApiVersion;
        if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
            runtime.provider = providers[i];
            break;
        }
    }
    if (runtime.provider == nullptr) return fail(message, capacity, "no compatible QNN interface version");

    const auto& api = runtime.provider->QNN_INTERFACE_VER_NAME;
    if (api.backendCreate == nullptr || api.backendFree == nullptr ||
        api.deviceCreate == nullptr || api.deviceGetPlatformInfo == nullptr ||
        api.deviceFreePlatformInfo == nullptr || api.deviceFree == nullptr ||
        api.contextCreate == nullptr || api.contextFree == nullptr ||
        api.graphCreate == nullptr || api.graphAddNode == nullptr ||
        api.graphFinalize == nullptr || api.graphExecute == nullptr ||
        api.tensorCreateGraphTensor == nullptr) {
        return fail(message, capacity, "QNN HTP provider is missing required graph APIs");
    }
    // Errors are forwarded to logcat without adding per-op debug overhead.
    if (api.logCreate != nullptr)
        api.logCreate(qnnLogCallback, QNN_LOG_LEVEL_ERROR, &runtime.log);
    status = api.backendCreate(runtime.log, nullptr, &runtime.backend);
    if (status != QNN_SUCCESS) return fail(message, capacity, "QNN HTP backend initialization failed", status);
    if (api.deviceCreate == nullptr || api.deviceGetPlatformInfo == nullptr)
        return fail(message, capacity, "QNN HTP device API is unavailable");
    status = api.deviceGetPlatformInfo(runtime.log, &runtime.platformInfo);
    if (status != QNN_SUCCESS || runtime.platformInfo == nullptr)
        return fail(message, capacity, "QNN HTP platform discovery failed", status);
    if (runtime.platformInfo->version != QNN_DEVICE_PLATFORM_INFO_VERSION_1 ||
        runtime.platformInfo->v1.numHwDevices == 0 || runtime.platformInfo->v1.hwDevices == nullptr) {
        std::snprintf(message, capacity, "QNN HTP reported no supported hardware devices (platform version %u)",
                      static_cast<unsigned>(runtime.platformInfo->version));
        return false;
    }
    const auto& hw = runtime.platformInfo->v1.hwDevices[0];
    __android_log_print(ANDROID_LOG_INFO, "SmolVLM-QNN",
                        "HTP platform devices=%u first_device_id=%u type=%u cores=%u",
                        runtime.platformInfo->v1.numHwDevices, hw.v1.deviceId,
                        hw.v1.deviceType, hw.v1.numCores);
    // Match QAIRT SampleApp's default device setup. Supplying partial platform
    // metadata here can be rejected by HTP as QNN_DEVICE_ERROR_INVALID_CONFIG.
    status = api.deviceCreate(runtime.log, nullptr, &runtime.device);
    if (status != QNN_SUCCESS) {
        const char* detail = nullptr;
        if (api.errorGetVerboseMessage != nullptr &&
            api.errorGetVerboseMessage(status, &detail) == QNN_SUCCESS && detail != nullptr) {
            std::snprintf(message, capacity, "QNN HTP device initialization failed: %s", detail);
            if (api.errorFreeVerboseMessage != nullptr) api.errorFreeVerboseMessage(detail);
        } else {
            std::snprintf(message, capacity,
                          "QNN HTP device initialization failed (QNN status %lu, devices %u, id %u, type %u, cores %u)",
                          static_cast<unsigned long>(status), runtime.platformInfo->v1.numHwDevices,
                          hw.v1.deviceId, hw.v1.deviceType, hw.v1.numCores);
        }
        return false;
    }

    // Execute a tiny known-answer ReLU graph. This proves tensors are compiled,
    // dispatched to HTP and copied back; provider/device initialization alone
    // is not enough to claim that accelerator execution works.
    status = api.contextCreate(runtime.backend, runtime.device, nullptr, &runtime.context);
    if (status != QNN_SUCCESS)
        return fail(message, capacity, "QNN HTP context creation failed", status);
    Qnn_GraphHandle_t graph = nullptr;
    status = api.graphCreate(runtime.context, "qnn_htp_smoke", nullptr, &graph);
    if (status != QNN_SUCCESS) return fail(message, capacity, "QNN HTP graph creation failed", status);

    uint32_t dimensions[] = {1, 4};
    Qnn_Tensor_t input{};
    input.version = QNN_TENSOR_VERSION_1;
    input.v1.id = 0;
    input.v1.name = "input";
    input.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    input.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    input.v1.dataType = QNN_DATATYPE_FLOAT_32;
    input.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    input.v1.rank = 2;
    input.v1.dimensions = dimensions;
    input.v1.memType = QNN_TENSORMEMTYPE_RAW;
    input.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    Qnn_Tensor_t output{};
    output.version = QNN_TENSOR_VERSION_1;
    output.v1.id = 1;
    output.v1.name = "output";
    output.v1.type = QNN_TENSOR_TYPE_APP_READ;
    output.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    output.v1.dataType = QNN_DATATYPE_FLOAT_32;
    output.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    output.v1.rank = 2;
    output.v1.dimensions = dimensions;
    output.v1.memType = QNN_TENSORMEMTYPE_RAW;
    output.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    status = api.tensorCreateGraphTensor(graph, &input);
    if (status != QNN_SUCCESS) return fail(message, capacity, "QNN HTP input tensor creation failed", status);
    status = api.tensorCreateGraphTensor(graph, &output);
    if (status != QNN_SUCCESS) return fail(message, capacity, "QNN HTP output tensor creation failed", status);

    Qnn_Tensor_t inputRef{};
    inputRef.version = QNN_TENSOR_VERSION_1;
    inputRef.v1.id = input.v1.id;
    Qnn_Tensor_t outputRef{};
    outputRef.version = QNN_TENSOR_VERSION_1;
    outputRef.v1.id = output.v1.id;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "smoke_relu";
    op.v1.packageName = "qti.aisw";
    op.v1.typeName = "Relu";
    op.v1.numOfParams = 0;
    op.v1.params = nullptr;
    op.v1.numOfInputs = 1;
    op.v1.inputTensors = &inputRef;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &outputRef;
    status = api.graphAddNode(graph, op);
    if (status != QNN_SUCCESS) return fail(message, capacity, "QNN HTP ReLU node creation failed", status);
    status = api.graphFinalize(graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return fail(message, capacity, "QNN HTP graph finalize failed", status);

    float inputValues[] = {-1.0f, 2.0f, -3.0f, 4.0f};
    float outputValues[] = {0.0f, 0.0f, 0.0f, 0.0f};
    input.v1.clientBuf.data = inputValues;
    input.v1.clientBuf.dataSize = sizeof(inputValues);
    output.v1.clientBuf.data = outputValues;
    output.v1.clientBuf.dataSize = sizeof(outputValues);
    const auto start = std::chrono::steady_clock::now();
    for (unsigned run = 0; run < smokeRuns; ++run) {
        status = api.graphExecute(graph, &input, 1, &output, 1, nullptr, nullptr);
        if (status != QNN_SUCCESS)
            return fail(message, capacity, "QNN HTP graph execution failed", status);
        if (outputValues[0] != 0.0f || outputValues[1] != 2.0f ||
            outputValues[2] != 0.0f || outputValues[3] != 4.0f) {
            std::snprintf(message, capacity,
                          "QNN HTP graph output mismatch on run %u: %.1f %.1f %.1f %.1f",
                          run + 1, outputValues[0], outputValues[1], outputValues[2], outputValues[3]);
            return false;
        }
    }
    const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();

    const auto& version = runtime.provider->apiVersion.coreApiVersion;
    std::snprintf(message, capacity,
                  "QNN HTP ReLU graph passed (%u/%u runs; %.1f us/run; output 0,2,0,4; API %u.%u).",
                  smokeRuns, smokeRuns, static_cast<double>(elapsedUs) / smokeRuns,
                  version.major, version.minor);

    if (modelPath != nullptr) {
        SafeTensors* weightsFile = nullptr;
        char error[256] = {};
        if (!safetensors_open(modelPath, &weightsFile, error, sizeof(error))) {
            std::snprintf(message, capacity, "cannot open model Safetensors: %s", error);
            return false;
        }
        const SafeTensorInfo* weightInfo = safetensors_find(
            weightsFile, "model.text_model.layers.0.self_attn.q_proj.weight");
        constexpr uint32_t width = 576;
        constexpr size_t weightElements = static_cast<size_t>(width) * width;
        if (weightInfo == nullptr || weightInfo->rank != 2 ||
            weightInfo->shape[0] != width || weightInfo->shape[1] != width ||
            (weightInfo->dtype != TENSOR_DTYPE_F32 && weightInfo->dtype != TENSOR_DTYPE_F16)) {
            safetensors_close(weightsFile);
            std::snprintf(message, capacity,
                          "q_proj.weight must be F32/F16 [576,576] in the selected model");
            return false;
        }
        std::vector<uint16_t> halfWeights(weightElements);
        if (weightInfo->dtype == TENSOR_DTYPE_F16) {
            if (!safetensors_read_raw(weightsFile, weightInfo, halfWeights.data(),
                                      halfWeights.size() * sizeof(uint16_t), error, sizeof(error))) {
                safetensors_close(weightsFile);
                std::snprintf(message, capacity, "cannot read q_proj.weight: %s", error);
                return false;
            }
        } else {
            std::vector<float> sourceWeights(weightElements);
            if (!safetensors_read_raw(weightsFile, weightInfo, sourceWeights.data(),
                                      sourceWeights.size() * sizeof(float), error, sizeof(error))) {
                safetensors_close(weightsFile);
                std::snprintf(message, capacity, "cannot read q_proj.weight: %s", error);
                return false;
            }
            for (size_t i = 0; i < weightElements; ++i)
                halfWeights[i] = tensor_f32_to_f16(sourceWeights[i]);
        }
        safetensors_close(weightsFile);

        InferenceSdkQnnHtpSession *session = nullptr;
        InferenceSdkQnnHtpLinear *linear = nullptr;
        char detail[256] = {};
        if (!inference_sdk_qnn_htp_session_create(htpDirectory, &session, detail, sizeof(detail))) {
            std::snprintf(message, capacity, "QNN session failed: %s", detail);
            return false;
        }
        SafeTensors *sdkWeightsFile = nullptr;
        if (!safetensors_open(modelPath, &sdkWeightsFile, error, sizeof(error))) {
            inference_sdk_qnn_htp_session_destroy(session);
            std::snprintf(message, capacity, "cannot reopen model for SDK Safetensors loader: %s", error);
            return false;
        }
        const int linearLoaded = inference_sdk_qnn_htp_linear_create_from_safetensors(
            session, "smolvlm_q_proj_layer0_sdk", sdkWeightsFile,
            "model.text_model.layers.0.self_attn.q_proj.weight", 1, width, width,
            &linear, detail, sizeof(detail));
        safetensors_close(sdkWeightsFile);
        if (!linearLoaded) {
            inference_sdk_qnn_htp_session_destroy(session);
            std::snprintf(message, capacity, "QNN q_proj graph failed: %s", detail);
            return false;
        }
        std::vector<uint16_t> inputValues(width), outputValues(width);
        std::vector<float> reference(width, 0.0f);
        for (uint32_t i = 0; i < width; ++i)
            inputValues[i] = tensor_f32_to_f16((static_cast<int>(i % 17) - 8) / 16.0f);
        for (uint32_t row = 0; row < width; ++row) {
            float sum = 0.0f;
            for (uint32_t col = 0; col < width; ++col)
                sum += tensor_f16_to_f32(inputValues[col]) *
                       tensor_f16_to_f32(halfWeights[static_cast<size_t>(row) * width + col]);
            reference[row] = sum;
        }
        const auto linearStart = std::chrono::steady_clock::now();
        for (unsigned run = 0; run < smokeRuns; ++run) {
            if (!inference_sdk_qnn_htp_linear_execute(linear, inputValues.data(), 1,
                    outputValues.data(), detail, sizeof(detail))) {
                inference_sdk_qnn_htp_linear_destroy(linear);
                inference_sdk_qnn_htp_session_destroy(session);
                std::snprintf(message, capacity, "QNN q_proj execution failed: %s", detail);
                return false;
            }
        }
        const auto linearElapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - linearStart).count();
        float maxAbsError = 0.0f, maxRelativeError = 0.0f;
        for (uint32_t i = 0; i < width; ++i) {
            const float actual = tensor_f16_to_f32(outputValues[i]);
            const float absError = std::abs(actual - reference[i]);
            maxAbsError = std::max(maxAbsError, absError);
            maxRelativeError = std::max(maxRelativeError,
                absError / std::max(0.1f, std::abs(reference[i])));
        }
        inference_sdk_qnn_htp_linear_destroy(linear);
        inference_sdk_qnn_htp_session_destroy(session);
        if (maxAbsError > 0.08f || maxRelativeError > 0.08f) {
            std::snprintf(message, capacity,
                          "HTP q_proj consistency failed: max_abs=%.5f max_rel=%.5f",
                          maxAbsError, maxRelativeError);
            return false;
        }
        std::snprintf(message, capacity,
                      "HTP q_proj through reusable SDK graph passed (%u/%u; max_abs=%.5f max_rel=%.5f; %.1f us/run).",
                      smokeRuns, smokeRuns, maxAbsError, maxRelativeError,
                      static_cast<double>(linearElapsedUs) / smokeRuns);
    }
    return true;
}
}

extern "C" int inference_sdk_qnn_htp_smoke_test(const char* htpDirectory,
                                                unsigned iterations,
                                                char* message,
                                                size_t messageCapacity) {
    if (htpDirectory == nullptr || message == nullptr || messageCapacity == 0 || iterations == 0) {
        return 0;
    }
    Runtime runtime;
    message[0] = '\0';
    try {
        return probe(runtime, htpDirectory, iterations, nullptr, message, messageCapacity) ? 1 : 0;
    } catch (const std::exception& error) {
        std::snprintf(message, messageCapacity, "QNN HTP smoke test allocation/runtime error: %s", error.what());
        return 0;
    } catch (...) {
        std::snprintf(message, messageCapacity, "QNN HTP smoke test failed with an unknown exception");
        return 0;
    }
}

extern "C" int inference_sdk_qnn_htp_safetensors_linear_test(const char* htpDirectory,
                                                              const char* modelPath,
                                                              unsigned iterations,
                                                              char* message,
                                                              size_t messageCapacity) {
    if (htpDirectory == nullptr || modelPath == nullptr || message == nullptr ||
        messageCapacity == 0 || iterations == 0) {
        return 0;
    }
    Runtime runtime;
    message[0] = '\0';
    try {
        return probe(runtime, htpDirectory, iterations, modelPath, message, messageCapacity) ? 1 : 0;
    } catch (const std::exception& error) {
        std::snprintf(message, messageCapacity, "HTP Safetensors test allocation/runtime error: %s", error.what());
        return 0;
    } catch (...) {
        std::snprintf(message, messageCapacity, "HTP Safetensors test failed with an unknown exception");
        return 0;
    }
}
