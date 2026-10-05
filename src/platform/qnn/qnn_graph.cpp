#include "qnn_htp.h"

#include <QNN/QnnBackend.h>
#include <QNN/QnnDevice.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnOpDef.h>

#include <android/log.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include "tensor_f16.h"
}

namespace {
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
std::atomic<uint32_t> g_nextTensorId{0};

uint32_t nextTensorId() {
    return g_nextTensorId.fetch_add(1, std::memory_order_relaxed);
}

void logCallback(const char *format, QnnLog_Level_t level, uint64_t, va_list args) {
    int priority = ANDROID_LOG_ERROR;
    if (level == QNN_LOG_LEVEL_WARN) priority = ANDROID_LOG_WARN;
    else if (level == QNN_LOG_LEVEL_INFO) priority = ANDROID_LOG_INFO;
    __android_log_vprint(priority, "DiffStep-QNN", format, args);
}
bool setError(char *out, size_t cap, const char *text, Qnn_ErrorHandle_t status = QNN_SUCCESS) {
    if (out && cap) {
        if (status == QNN_SUCCESS) std::snprintf(out, cap, "%s", text);
        else std::snprintf(out, cap, "%s (QNN status %lu)", text,
                           static_cast<unsigned long>(status));
    }
    return false;
}
}

struct InferenceSdkQnnHtpSession {
    void *library = nullptr;
    const QnnInterface_t *provider = nullptr;
    bool useGpu = false;
    Qnn_LogHandle_t log = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    const QnnDevice_PlatformInfo_t *platform = nullptr;
    ~InferenceSdkQnnHtpSession() {
        if (provider) {
            const auto &api = provider->QNN_INTERFACE_VER_NAME;
            if (context && api.contextFree) api.contextFree(context, nullptr);
            if (device && api.deviceFree) api.deviceFree(device);
            if (platform && api.deviceFreePlatformInfo) api.deviceFreePlatformInfo(log, platform);
            if (backend && api.backendFree) api.backendFree(backend);
            if (log && api.logFree) api.logFree(log);
        }
        if (library) dlclose(library);
    }
};

struct InferenceSdkQnnHtpLinear {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 1;
    Qnn_Tensor_t input{}, output{};
    std::vector<uint32_t> inputDims, outputDims, weightDims;
    std::vector<uint16_t> weights;
};

struct InferenceSdkQnnHtpRmsNorm {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, width = 0;
    Qnn_Tensor_t input{}, output{};
    std::vector<uint32_t> inputDims, outputDims, scaleDims;
    std::vector<uint16_t> scale;
};

struct InferenceSdkQnnHtpEmbedding {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, vocabulary = 0, width = 0;
    Qnn_Tensor_t ids{}, output{};
    std::vector<uint32_t> idDims, tableDims, outputDims;
    std::vector<uint16_t> table;
};

struct InferenceSdkQnnHtpRope {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t heads = 0, rows = 0, width = 0, maxPositions = 0;
    Qnn_Tensor_t input{}, positions{}, output{};
    std::vector<uint32_t> inputDims, positionDims, outputDims, cacheDims;
    std::vector<uint16_t> cosine, sine;
};

struct InferenceSdkQnnHtpMlp {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, hidden = 0, intermediate = 0;
    Qnn_Tensor_t input{}, residual{}, output{};
    std::vector<uint32_t> hiddenDims, intermediateDims;
    std::vector<uint16_t> gate, up, down;
};

struct InferenceSdkQnnHtpAttention {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t qHeads = 0, kvHeads = 0, qRows = 0, kRows = 0, width = 0;
    Qnn_Tensor_t query{}, key{}, value{}, maskTensor{}, output{};
    std::vector<uint32_t> qDims, kvDims, scoreDims, maskDims, indexDims, scaleDims;
    std::vector<int32_t> headMap;
    std::vector<uint16_t> mask, scale;
};

struct InferenceSdkQnnHtpLayerNorm {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, width = 0;
    Qnn_Tensor_t input{}, output{};
    std::vector<uint32_t> inputDims, outputDims, channelDims;
    std::vector<uint16_t> scale, bias;
};

struct InferenceSdkQnnHtpLinearBias {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, inputWidth = 0, outputWidth = 0;
    Qnn_Tensor_t input{}, output{};
    std::vector<uint32_t> inputDims, weightDims, outputDims, biasDims;
    std::vector<uint16_t> weights, bias;
};

struct InferenceSdkQnnHtpElementwise {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, rhsRows = 0, width = 0;
    InferenceSdkQnnHtpElementwiseOp operation = INFERENCE_SDK_QNN_HTP_ADD;
    Qnn_Tensor_t lhs{}, rhs{}, output{};
    std::vector<uint32_t> lhsDims, rhsDims, outputDims;
};

struct InferenceSdkQnnHtpGelu {
    InferenceSdkQnnHtpSession *session = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t rows = 0, width = 0;
    Qnn_Tensor_t input{}, output{};
    std::vector<uint32_t> dims;
};

static void initTensor(Qnn_Tensor_t &t, const char *tensorName,
        Qnn_TensorType_t type, Qnn_DataType_t dtype, uint32_t *dims,
        uint32_t rank, void *data, uint32_t bytes) {
    t = Qnn_Tensor_t{};
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.id = nextTensorId();
    t.v1.name = tensorName;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dtype;
    t.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    t.v1.clientBuf.data = data;
    t.v1.clientBuf.dataSize = bytes;
}

extern "C" int inference_sdk_qnn_htp_session_create(const char *directory,
        InferenceSdkQnnHtpSession **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!directory || !result || !message || !capacity)
        return setError(message, capacity, "invalid QNN session arguments");
    *result = nullptr;
    message[0] = '\0';
    std::unique_ptr<InferenceSdkQnnHtpSession> owned(new (std::nothrow) InferenceSdkQnnHtpSession());
    auto *s = owned.get();
    if (!s) return setError(message, capacity, "QNN session allocation failed");
    const char *backendSetting = std::getenv("INFERENCE_SDK_QNN_BACKEND");
    const bool useGpu = backendSetting && std::strcmp(backendSetting, "gpu") == 0;
    s->useGpu = useGpu;
    const char *libraryName = useGpu ? "libQnnGpu.so" : "libQnnHtp.so";
    if (!useGpu) {
        std::string paths(directory);
        paths += ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp;";
        setenv("ADSP_LIBRARY_PATH", paths.c_str(), 1);
    }
    s->library = dlopen(libraryName, RTLD_NOW | RTLD_LOCAL);
    if (!s->library) {
        const char *error = dlerror();
        setError(message, capacity, error ? error : "cannot load QNN backend library");
        return 0;
    }
    auto getProviders = reinterpret_cast<GetProviders>(dlsym(s->library, "QnnInterface_getProviders"));
    if (!getProviders) { setError(message, capacity, "QNN provider entrypoint is missing"); return 0; }
    const QnnInterface_t **providers = nullptr; uint32_t count = 0;
    auto status = getProviders(&providers, &count);
    if (status != QNN_SUCCESS || !providers || !count) { setError(message, capacity, "cannot enumerate QNN providers", status); return 0; }
    for (uint32_t i = 0; i < count; ++i) {
        const auto &v = providers[i]->apiVersion.coreApiVersion;
        if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) { s->provider = providers[i]; break; }
    }
    if (!s->provider) { setError(message, capacity, "no compatible QNN interface version"); return 0; }
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    if (!api.backendCreate || !api.backendFree ||
        (!useGpu && (!api.deviceGetPlatformInfo || !api.deviceFreePlatformInfo)) ||
        !api.deviceCreate || !api.deviceFree ||
        !api.contextCreate || !api.contextFree || !api.graphCreate ||
        !api.graphAddNode || !api.graphFinalize || !api.graphExecute ||
        !api.tensorCreateGraphTensor) {
        setError(message, capacity, "QNN provider is missing required graph APIs");
        return 0;
    }
    if (api.logCreate) api.logCreate(logCallback, QNN_LOG_LEVEL_ERROR, &s->log);
    status = api.backendCreate(s->log, nullptr, &s->backend);
    if (status != QNN_SUCCESS) { setError(message, capacity, "QNN backend initialization failed", status); return 0; }
    if (!useGpu) {
        status = api.deviceGetPlatformInfo(s->log, &s->platform);
        if (status != QNN_SUCCESS || !s->platform || s->platform->version != QNN_DEVICE_PLATFORM_INFO_VERSION_1 || s->platform->v1.numHwDevices == 0) {
            setError(message, capacity, "QNN HTP reported no supported hardware devices", status); return 0;
        }
    }
    status = api.deviceCreate(s->log, nullptr, &s->device);
    if (status != QNN_SUCCESS) { setError(message, capacity, "QNN device initialization failed", status); return 0; }
    status = api.contextCreate(s->backend, s->device, nullptr, &s->context);
    if (status != QNN_SUCCESS) { setError(message, capacity, "QNN HTP context creation failed", status); return 0; }
    *result = owned.release();
    std::snprintf(message, capacity, "QNN %s session ready (API %u.%u)", useGpu ? "GPU" : "HTP",
                  s->provider->apiVersion.coreApiVersion.major, s->provider->apiVersion.coreApiVersion.minor);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN session initialization failed with an allocation/runtime exception");
}

extern "C" void inference_sdk_qnn_htp_session_destroy(InferenceSdkQnnHtpSession *s) { delete s; }

extern "C" int inference_sdk_qnn_htp_linear_create(InferenceSdkQnnHtpSession *s,
        const char *name, const uint16_t *weights, uint32_t rows, uint32_t in, uint32_t out,
        InferenceSdkQnnHtpLinear **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !weights || !rows || !in || !out || !result || !message || !capacity)
        return setError(message, capacity, "invalid QNN linear graph arguments");
    if (static_cast<uint64_t>(in) * out > UINT32_MAX / sizeof(uint16_t))
        return setError(message, capacity, "QNN FP16 linear weights exceed the client buffer size limit");
    if (static_cast<uint64_t>(rows) * std::max(in, out) > UINT32_MAX / sizeof(uint16_t))
        return setError(message, capacity, "QNN FP16 linear activation exceeds the client buffer size limit");
    *result = nullptr; message[0] = '\0';
    std::unique_ptr<InferenceSdkQnnHtpLinear> owned(new (std::nothrow) InferenceSdkQnnHtpLinear());
    auto *l = owned.get();
    if (!l) return setError(message, capacity, "QNN linear allocation failed");
    l->session = s; l->rows = rows;
    l->weights.resize(static_cast<size_t>(in) * out);
    // QNN MatMul consumes [M,K] x [K,N]. SDK/SafeTensors linear weights are [N,K].
    for (uint32_t output = 0; output < out; ++output)
        for (uint32_t inputIndex = 0; inputIndex < in; ++inputIndex)
            l->weights[static_cast<size_t>(inputIndex) * out + output] =
                weights[static_cast<size_t>(output) * in + inputIndex];
    l->inputDims = {rows, in}; l->weightDims = {in, out}; l->outputDims = {rows, out};
    auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &l->graph);
    if (status != QNN_SUCCESS) { setError(message, capacity, "QNN linear graph creation failed", status); return 0; }
    auto initTensor = [](Qnn_Tensor_t &t, const char *tensorName, Qnn_TensorType_t type,
                         uint32_t *dims, uint32_t rank, void *data, uint32_t bytes) {
        t = Qnn_Tensor_t{}; t.version = QNN_TENSOR_VERSION_1; t.v1.id = nextTensorId();
        t.v1.name = tensorName; t.v1.type = type;
        t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER; t.v1.dataType = QNN_DATATYPE_FLOAT_16;
        t.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT; t.v1.rank = rank; t.v1.dimensions = dims;
        t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
        t.v1.clientBuf.data = data; t.v1.clientBuf.dataSize = bytes;
    };
    Qnn_Tensor_t w{};
    initTensor(l->input, "input", QNN_TENSOR_TYPE_APP_WRITE, l->inputDims.data(), 2, nullptr, 0);
    initTensor(w, "weight", QNN_TENSOR_TYPE_STATIC, l->weightDims.data(), 2, l->weights.data(), static_cast<uint32_t>(l->weights.size() * 2));
    initTensor(l->output, "output", QNN_TENSOR_TYPE_APP_READ, l->outputDims.data(), 2, nullptr, 0);
    status = api.tensorCreateGraphTensor(l->graph, &l->input);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(l->graph, &w);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(l->graph, &l->output);
    if (status != QNN_SUCCESS) { setError(message, capacity, "QNN linear tensor creation failed", status); return 0; }
    __android_log_print(ANDROID_LOG_INFO, "DiffStep-QNN",
        "Linear graph tensors: input=%u weight=%u output=%u",
        l->input.v1.id, w.v1.id, l->output.v1.id);
    Qnn_Tensor_t inRef{}, weightRef{}, outRef{};
    inRef.version = weightRef.version = outRef.version = QNN_TENSOR_VERSION_1;
    inRef.v1.id = l->input.v1.id; weightRef.v1.id = w.v1.id; outRef.v1.id = l->output.v1.id;
    Qnn_Tensor_t inputs[] = {inRef, weightRef}; Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    Qnn_Param_t matmulParams[] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    for (auto &param : matmulParams) {
        param.paramType = QNN_PARAMTYPE_SCALAR;
        param.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
        param.scalarParam.bool8Value = 0;
    }
    matmulParams[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    matmulParams[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    op.v1.name = "linear_matmul"; op.v1.packageName = "qti.aisw"; op.v1.typeName = QNN_OP_MAT_MUL;
    op.v1.numOfParams = 2; op.v1.params = matmulParams; op.v1.numOfInputs = 2; op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &outRef;
    if (api.backendValidateOpConfig) {
        Qnn_Tensor_t validationInputs[] = {l->input, w};
        Qnn_Tensor_t validationOutput = l->output;
        Qnn_OpConfig_t validationOp = op;
        validationOp.v1.inputTensors = validationInputs;
        validationOp.v1.outputTensors = &validationOutput;
        const auto validationStatus = api.backendValidateOpConfig(s->backend, validationOp);
        if (validationStatus != QNN_SUCCESS && validationStatus != QNN_BACKEND_ERROR_NOT_SUPPORTED) {
            __android_log_print(ANDROID_LOG_ERROR, "DiffStep-QNN",
                "QNN %s linear validation failed: status=%lu", s->useGpu ? "GPU" : "HTP",
                static_cast<unsigned long>(validationStatus));
            setError(message, capacity, "QNN linear operator validation failed", validationStatus);
            return 0;
        }
        if (validationStatus == QNN_SUCCESS) {
            __android_log_print(ANDROID_LOG_INFO, "DiffStep-QNN",
                "QNN %s linear op validation passed", s->useGpu ? "GPU" : "HTP");
        }
    }
    status = api.graphAddNode(l->graph, op);
    if (status == QNN_SUCCESS) status = api.graphFinalize(l->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) { setError(message, capacity, "QNN linear graph compile failed", status); return 0; }
    *result = owned.release(); std::snprintf(message, capacity,
        "QNN FP16 linear graph ready (%u rows, %u x %u)", rows, in, out); return 1;
} catch (...) {
    return setError(message, capacity, "QNN linear graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_linear_create_from_safetensors(
        InferenceSdkQnnHtpSession *s, const char *graphName,
        const SafeTensors *weightsFile, const char *tensorName,
        uint32_t rows, uint32_t inputWidth, uint32_t outputWidth,
        InferenceSdkQnnHtpLinear **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !graphName || !weightsFile || !tensorName || !rows || !inputWidth ||
        !outputWidth || !result || !message || !capacity)
        return setError(message, capacity, "invalid Safetensors QNN linear arguments");
    const SafeTensorInfo *info = safetensors_find(weightsFile, tensorName);
    const uint64_t count = static_cast<uint64_t>(inputWidth) * outputWidth;
    if (!info || info->rank != 2 || info->shape[0] != outputWidth ||
        info->shape[1] != inputWidth || count > UINT32_MAX / sizeof(uint16_t) ||
        (info->dtype != TENSOR_DTYPE_F16 && info->dtype != TENSOR_DTYPE_F32 &&
         info->dtype != TENSOR_DTYPE_BF16) ||
        count > UINT64_MAX / tensor_dtype_size(info->dtype) ||
        info->byte_length > SIZE_MAX ||
        info->byte_length != count * tensor_dtype_size(info->dtype))
        return setError(message, capacity, "Safetensors projection has incompatible name, shape, or dtype");
    std::vector<uint16_t> converted(static_cast<size_t>(count));
    std::vector<uint8_t> raw(static_cast<size_t>(info->byte_length));
    if (!safetensors_read_raw(weightsFile, info, raw.data(), raw.size(), message, capacity))
        return 0;
    if (info->dtype == TENSOR_DTYPE_F16) {
        std::memcpy(converted.data(), raw.data(), raw.size());
    } else if (info->dtype == TENSOR_DTYPE_F32) {
        for (size_t i = 0; i < converted.size(); ++i) {
            float value;
            std::memcpy(&value, raw.data() + i * sizeof(float), sizeof(float));
            converted[i] = tensor_f32_to_f16(value);
        }
    } else {
        for (size_t i = 0; i < converted.size(); ++i) {
            uint16_t value;
            std::memcpy(&value, raw.data() + i * sizeof(uint16_t), sizeof(uint16_t));
            converted[i] = tensor_f32_to_f16(tensor_bf16_to_f32(value));
        }
    }
    if (!inference_sdk_qnn_htp_linear_create(s, graphName, converted.data(), rows,
            inputWidth, outputWidth, result, message, capacity))
        return 0;
    std::snprintf(message, capacity, "Safetensors projection '%s' loaded and compiled (%u x %u)",
                  tensorName, outputWidth, inputWidth);
    return 1;
} catch (...) {
    return setError(message, capacity, "Safetensors QNN linear load failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_linear_execute(InferenceSdkQnnHtpLinear *l,
        const uint16_t *input, uint32_t rows, uint16_t *output, char *message, size_t capacity) {
    if (!l || !input || !output || !message || !capacity) return 0;
    if (rows != l->rows) return setError(message, capacity,
        "QNN linear execution row count must match its compiled graph");
    l->input.v1.clientBuf.data = const_cast<uint16_t *>(input);
    l->input.v1.clientBuf.dataSize = rows * l->inputDims[1] * 2;
    l->output.v1.clientBuf.data = output;
    l->output.v1.clientBuf.dataSize = rows * l->outputDims[1] * 2;
    auto status = l->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        l->graph, &l->input, 1, &l->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN linear execution failed", status);
    if (l->output.v1.clientBuf.dataSize >= sizeof(uint16_t) &&
        *static_cast<const uint16_t *>(l->output.v1.clientBuf.data) == 0) {
        __android_log_print(ANDROID_LOG_ERROR, "DiffStep-QNN",
            "Linear output[0]=0 input_id=%u output_id=%u input_ptr=%p output_ptr=%p input0=0x%04x weight0=0x%04x",
            l->input.v1.id, l->output.v1.id, l->input.v1.clientBuf.data,
            l->output.v1.clientBuf.data, *static_cast<const uint16_t *>(l->input.v1.clientBuf.data),
            l->weights.empty() ? 0 : l->weights[0]);
    }
    std::snprintf(message, capacity, "QNN FP16 linear executed (%u rows)", rows); return 1;
}

extern "C" void inference_sdk_qnn_htp_linear_destroy(InferenceSdkQnnHtpLinear *l) { delete l; }

extern "C" int inference_sdk_qnn_htp_rms_norm_create(InferenceSdkQnnHtpSession *s,
        const char *name, const uint16_t *scale, uint32_t rows, uint32_t width,
        float epsilon, InferenceSdkQnnHtpRmsNorm **result,
        char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !scale || !rows || !width || !(epsilon > 0.0f) ||
        !result || !message || !capacity)
        return setError(message, capacity, "invalid QNN RMSNorm arguments");
    if (static_cast<uint64_t>(rows) * width > UINT32_MAX / sizeof(uint16_t))
        return setError(message, capacity, "QNN RMSNorm activation exceeds client buffer size limit");
    std::unique_ptr<InferenceSdkQnnHtpRmsNorm> owned(new (std::nothrow) InferenceSdkQnnHtpRmsNorm());
    auto *n = owned.get();
    if (!n) return setError(message, capacity, "QNN RMSNorm allocation failed");
    n->session = s; n->rows = rows; n->width = width;
    n->scale.assign(scale, scale + width);
    n->inputDims = {rows, width}; n->outputDims = {rows, width}; n->scaleDims = {width};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &n->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RMSNorm graph creation failed", status);
    Qnn_Tensor_t scaleTensor{};
    initTensor(n->input, "input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               n->inputDims.data(), 2, nullptr, 0);
    initTensor(scaleTensor, "scale", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               n->scaleDims.data(), 1, n->scale.data(), static_cast<uint32_t>(width * sizeof(uint16_t)));
    initTensor(n->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
               n->outputDims.data(), 2, nullptr, 0);
    status = api.tensorCreateGraphTensor(n->graph, &n->input);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(n->graph, &scaleTensor);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(n->graph, &n->output);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RMSNorm tensor creation failed", status);

    Qnn_Tensor_t inputRef{}, scaleRef{}, outputRef{};
    inputRef.version = scaleRef.version = outputRef.version = QNN_TENSOR_VERSION_1;
    inputRef.v1.id = n->input.v1.id; scaleRef.v1.id = scaleTensor.v1.id;
    outputRef.v1.id = n->output.v1.id;
    uint32_t axisDims[] = {1}; uint32_t axis = 1;
    Qnn_Tensor_t axes{};
    initTensor(axes, "axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32,
               axisDims, 1, &axis, sizeof(axis));
    status = api.tensorCreateGraphTensor(n->graph, &axes);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RMSNorm axes tensor creation failed", status);
    Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    params[0].paramType = QNN_PARAMTYPE_TENSOR;
    params[0].name = QNN_OP_RMS_NORM_PARAM_AXES;
    params[0].tensorParam = axes;
    params[1].paramType = QNN_PARAMTYPE_SCALAR;
    params[1].name = QNN_OP_RMS_NORM_PARAM_EPSILON;
    params[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    params[1].scalarParam.floatValue = epsilon;
    Qnn_Tensor_t inputs[] = {inputRef, scaleRef};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "rms_norm"; op.v1.packageName = "qti.aisw"; op.v1.typeName = QNN_OP_RMS_NORM;
    op.v1.numOfParams = 2; op.v1.params = params;
    op.v1.numOfInputs = 2; op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
    status = api.graphAddNode(n->graph, op);
    if (status == QNN_SUCCESS) status = api.graphFinalize(n->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RMSNorm graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 RMSNorm graph ready (%u rows x %u, epsilon=%g)", rows, width, epsilon);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN RMSNorm graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_rms_norm_execute(InferenceSdkQnnHtpRmsNorm *n,
        const uint16_t *input, uint32_t rows, uint16_t *output,
        char *message, size_t capacity) {
    if (!n || !input || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN RMSNorm execution arguments");
    if (rows != n->rows)
        return setError(message, capacity, "QNN RMSNorm row count must match its compiled graph");
    n->input.v1.clientBuf.data = const_cast<uint16_t *>(input);
    n->input.v1.clientBuf.dataSize = rows * n->width * sizeof(uint16_t);
    n->output.v1.clientBuf.data = output;
    n->output.v1.clientBuf.dataSize = rows * n->width * sizeof(uint16_t);
    auto status = n->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        n->graph, &n->input, 1, &n->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RMSNorm execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 RMSNorm executed (%u rows)", rows);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_rms_norm_destroy(InferenceSdkQnnHtpRmsNorm *n) { delete n; }

extern "C" int inference_sdk_qnn_htp_embedding_create(InferenceSdkQnnHtpSession *s,
        const char *name, const uint16_t *table, uint32_t vocabulary, uint32_t width,
        uint32_t rows, InferenceSdkQnnHtpEmbedding **result,
        char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !table || !vocabulary || !width || !rows || !result || !message || !capacity)
        return setError(message, capacity, "invalid QNN embedding arguments");
    if (static_cast<uint64_t>(vocabulary) * width > UINT32_MAX / sizeof(uint16_t) ||
        static_cast<uint64_t>(rows) * width > UINT32_MAX / sizeof(uint16_t) ||
        static_cast<uint64_t>(rows) > UINT32_MAX / sizeof(int32_t))
        return setError(message, capacity, "QNN embedding tensor exceeds client buffer size limit");
    std::unique_ptr<InferenceSdkQnnHtpEmbedding> owned(new (std::nothrow) InferenceSdkQnnHtpEmbedding());
    auto *e = owned.get();
    if (!e) return setError(message, capacity, "QNN embedding allocation failed");
    e->session = s; e->rows = rows; e->vocabulary = vocabulary; e->width = width;
    e->table.assign(table, table + static_cast<size_t>(vocabulary) * width);
    e->idDims = {rows}; e->tableDims = {vocabulary, width}; e->outputDims = {rows, width};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &e->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN embedding graph creation failed", status);
    Qnn_Tensor_t weights{};
    initTensor(e->ids, "token_ids", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_INT_32,
               e->idDims.data(), 1, nullptr, 0);
    initTensor(weights, "embedding_table", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               e->tableDims.data(), 2, e->table.data(), static_cast<uint32_t>(e->table.size() * sizeof(uint16_t)));
    initTensor(e->output, "embeddings", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
               e->outputDims.data(), 2, nullptr, 0);
    status = api.tensorCreateGraphTensor(e->graph, &e->ids);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(e->graph, &weights);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(e->graph, &e->output);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN embedding tensor creation failed", status);
    Qnn_Tensor_t idRef{}, weightRef{}, outputRef{};
    idRef.version = weightRef.version = outputRef.version = QNN_TENSOR_VERSION_1;
    idRef.v1.id = e->ids.v1.id; weightRef.v1.id = weights.v1.id; outputRef.v1.id = e->output.v1.id;
    Qnn_Param_t axis = QNN_PARAM_INIT;
    axis.paramType = QNN_PARAMTYPE_SCALAR; axis.name = QNN_OP_GATHER_PARAM_AXIS;
    axis.scalarParam.dataType = QNN_DATATYPE_INT_32; axis.scalarParam.int32Value = 0;
    Qnn_Tensor_t inputs[] = {weightRef, idRef};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "token_embedding_gather"; op.v1.packageName = "qti.aisw"; op.v1.typeName = QNN_OP_GATHER;
    op.v1.numOfParams = 1; op.v1.params = &axis; op.v1.numOfInputs = 2; op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
    status = api.graphAddNode(e->graph, op);
    if (status == QNN_SUCCESS) status = api.graphFinalize(e->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN embedding graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 embedding graph ready (%u rows, vocab=%u, width=%u)", rows, vocabulary, width);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN embedding graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_embedding_execute(InferenceSdkQnnHtpEmbedding *e,
        const int32_t *ids, uint32_t rows, uint16_t *output, char *message, size_t capacity) {
    if (!e || !ids || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN embedding execution arguments");
    if (rows != e->rows)
        return setError(message, capacity, "QNN embedding row count must match its compiled graph");
    for (uint32_t i = 0; i < rows; ++i)
        if (ids[i] < 0 || static_cast<uint32_t>(ids[i]) >= e->vocabulary)
            return setError(message, capacity, "QNN embedding token ID is outside the vocabulary");
    e->ids.v1.clientBuf.data = const_cast<int32_t *>(ids);
    e->ids.v1.clientBuf.dataSize = rows * sizeof(int32_t);
    e->output.v1.clientBuf.data = output;
    e->output.v1.clientBuf.dataSize = rows * e->width * sizeof(uint16_t);
    auto status = e->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        e->graph, &e->ids, 1, &e->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN embedding execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 embedding executed (%u rows)", rows);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_embedding_destroy(InferenceSdkQnnHtpEmbedding *e) { delete e; }

extern "C" int inference_sdk_qnn_htp_rope_create(InferenceSdkQnnHtpSession *s,
        const char *name, uint32_t heads, uint32_t rows, uint32_t width,
        uint32_t maxPositions, float theta, InferenceSdkQnnHtpRope **result,
        char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !heads || !rows || width < 2 || width % 2 || !maxPositions ||
        !(theta > 0.0f) || !result || !message || !capacity)
        return setError(message, capacity, "invalid QNN RoPE arguments");
    const uint64_t cacheCount = static_cast<uint64_t>(maxPositions) * (width / 2);
    const uint64_t activationCount = static_cast<uint64_t>(heads) * rows * width;
    if (cacheCount > UINT32_MAX / sizeof(uint16_t) || activationCount > UINT32_MAX / sizeof(uint16_t) ||
        rows > UINT32_MAX / sizeof(int32_t))
        return setError(message, capacity, "QNN RoPE tensor exceeds client buffer size limit");
    std::unique_ptr<InferenceSdkQnnHtpRope> owned(new (std::nothrow) InferenceSdkQnnHtpRope());
    auto *r = owned.get();
    if (!r) return setError(message, capacity, "QNN RoPE allocation failed");
    r->session = s; r->heads = heads; r->rows = rows; r->width = width; r->maxPositions = maxPositions;
    r->inputDims = {1, heads, rows, width}; r->positionDims = {1, rows};
    r->outputDims = r->inputDims; r->cacheDims = {maxPositions, width / 2};
    r->cosine.resize(static_cast<size_t>(cacheCount)); r->sine.resize(static_cast<size_t>(cacheCount));
    for (uint32_t pos = 0; pos < maxPositions; ++pos) {
        for (uint32_t d = 0; d < width / 2; ++d) {
            const float angle = static_cast<float>(pos) /
                std::pow(theta, 2.0f * static_cast<float>(d) / static_cast<float>(width));
            const size_t index = static_cast<size_t>(pos) * (width / 2) + d;
            r->cosine[index] = tensor_f32_to_f16(std::cos(angle));
            r->sine[index] = tensor_f32_to_f16(std::sin(angle));
        }
    }
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &r->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RoPE graph creation failed", status);
    Qnn_Tensor_t cosTensor{}, sinTensor{};
    initTensor(r->input, "input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               r->inputDims.data(), 4, nullptr, 0);
    initTensor(r->positions, "position_ids", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_INT_32,
               r->positionDims.data(), 2, nullptr, 0);
    initTensor(cosTensor, "cos_cache", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               r->cacheDims.data(), 2, r->cosine.data(), static_cast<uint32_t>(cacheCount * sizeof(uint16_t)));
    initTensor(sinTensor, "sin_cache", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               r->cacheDims.data(), 2, r->sine.data(), static_cast<uint32_t>(cacheCount * sizeof(uint16_t)));
    initTensor(r->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
               r->outputDims.data(), 4, nullptr, 0);
    status = api.tensorCreateGraphTensor(r->graph, &r->input);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(r->graph, &r->positions);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(r->graph, &cosTensor);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(r->graph, &sinTensor);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(r->graph, &r->output);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RoPE tensor creation failed", status);
    Qnn_Tensor_t inputRef{}, cosRef{}, sinRef{}, positionRef{}, outputRef{};
    inputRef.version = cosRef.version = sinRef.version = positionRef.version = outputRef.version = QNN_TENSOR_VERSION_1;
    inputRef.v1.id = r->input.v1.id; cosRef.v1.id = cosTensor.v1.id;
    sinRef.v1.id = sinTensor.v1.id; positionRef.v1.id = r->positions.v1.id;
    outputRef.v1.id = r->output.v1.id;
    Qnn_Param_t interleaved = QNN_PARAM_INIT;
    interleaved.paramType = QNN_PARAMTYPE_SCALAR;
    interleaved.name = QNN_OP_ROTARY_EMBEDDING_PARAM_INTERLEAVED;
    interleaved.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    interleaved.scalarParam.bool8Value = 0;
    Qnn_Tensor_t inputs[] = {inputRef, cosRef, sinRef, positionRef};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "rotary_embedding"; op.v1.packageName = "qti.aisw"; op.v1.typeName = QNN_OP_ROTARY_EMBEDDING;
    op.v1.numOfParams = 1; op.v1.params = &interleaved;
    op.v1.numOfInputs = 4; op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
    status = api.graphAddNode(r->graph, op);
    if (status == QNN_SUCCESS) status = api.graphFinalize(r->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RoPE graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 split-half RoPE graph ready (heads=%u, rows=%u, width=%u, max_positions=%u)",
                  heads, rows, width, maxPositions);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN RoPE graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_rope_execute(InferenceSdkQnnHtpRope *r,
        const uint16_t *input, const int32_t *positionIds, uint16_t *output,
        char *message, size_t capacity) {
    if (!r || !input || !positionIds || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN RoPE execution arguments");
    for (uint32_t row = 0; row < r->rows; ++row)
        if (positionIds[row] < 0 || static_cast<uint32_t>(positionIds[row]) >= r->maxPositions)
            return setError(message, capacity, "QNN RoPE position ID is outside the cached context");
    const uint32_t inputBytes = r->heads * r->rows * r->width * sizeof(uint16_t);
    r->input.v1.clientBuf.data = const_cast<uint16_t *>(input); r->input.v1.clientBuf.dataSize = inputBytes;
    r->positions.v1.clientBuf.data = const_cast<int32_t *>(positionIds);
    r->positions.v1.clientBuf.dataSize = r->rows * sizeof(int32_t);
    r->output.v1.clientBuf.data = output; r->output.v1.clientBuf.dataSize = inputBytes;
    Qnn_Tensor_t inputs[] = {r->input, r->positions};
    auto status = r->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        r->graph, inputs, 2, &r->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN RoPE execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 RoPE executed (%u rows)", r->rows);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_rope_destroy(InferenceSdkQnnHtpRope *r) { delete r; }

extern "C" int inference_sdk_qnn_htp_mlp_create(InferenceSdkQnnHtpSession *s,
        const char *name, const uint16_t *gate, const uint16_t *up,
        const uint16_t *down, uint32_t rows, uint32_t hidden, uint32_t intermediate,
        InferenceSdkQnnHtpMlp **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !gate || !up || !down || !rows || !hidden || !intermediate ||
        !result || !message || !capacity)
        return setError(message, capacity, "invalid QNN gated MLP arguments");
    const uint64_t hi = static_cast<uint64_t>(hidden) * intermediate;
    const uint64_t hh = static_cast<uint64_t>(hidden) * hidden;
    if (hi > UINT32_MAX / sizeof(uint16_t) || hh > UINT32_MAX / sizeof(uint16_t) ||
        static_cast<uint64_t>(rows) * std::max(hidden, intermediate) > UINT32_MAX / sizeof(uint16_t))
        return setError(message, capacity, "QNN gated MLP tensor exceeds client buffer size limit");
    std::unique_ptr<InferenceSdkQnnHtpMlp> owned(new (std::nothrow) InferenceSdkQnnHtpMlp());
    auto *m = owned.get();
    if (!m) return setError(message, capacity, "QNN gated MLP allocation failed");
    m->session = s; m->rows = rows; m->hidden = hidden; m->intermediate = intermediate;
    m->gate.assign(gate, gate + static_cast<size_t>(hi));
    m->up.assign(up, up + static_cast<size_t>(hi));
    m->down.assign(down, down + static_cast<size_t>(hh));
    m->hiddenDims = {rows, hidden}; m->intermediateDims = {rows, intermediate};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &m->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN gated MLP graph creation failed", status);
    Qnn_Tensor_t gateW{}, upW{}, downW{};
    std::vector<uint32_t> gateDims = {intermediate, hidden};
    std::vector<uint32_t> downDims = {hidden, intermediate};
    initTensor(m->input, "input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               m->hiddenDims.data(), 2, nullptr, 0);
    initTensor(m->residual, "residual", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               m->hiddenDims.data(), 2, nullptr, 0);
    initTensor(gateW, "gate_weight", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               gateDims.data(), 2, m->gate.data(), static_cast<uint32_t>(hi * sizeof(uint16_t)));
    initTensor(upW, "up_weight", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               gateDims.data(), 2, m->up.data(), static_cast<uint32_t>(hi * sizeof(uint16_t)));
    initTensor(downW, "down_weight", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               downDims.data(), 2, m->down.data(), static_cast<uint32_t>(hh * sizeof(uint16_t)));
    initTensor(m->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
               m->hiddenDims.data(), 2, nullptr, 0);
    Qnn_Tensor_t gateOut{}, upOut{}, sigmoidOut{}, siluOut{}, gatedOut{}, downOut{};
    initTensor(gateOut, "gate_projection", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
               m->intermediateDims.data(), 2, nullptr, 0);
    initTensor(upOut, "up_projection", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
               m->intermediateDims.data(), 2, nullptr, 0);
    initTensor(sigmoidOut, "gate_sigmoid", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
               m->intermediateDims.data(), 2, nullptr, 0);
    initTensor(siluOut, "silu_gate", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
               m->intermediateDims.data(), 2, nullptr, 0);
    initTensor(gatedOut, "gated_activation", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
               m->intermediateDims.data(), 2, nullptr, 0);
    initTensor(downOut, "down_projection", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16,
               m->hiddenDims.data(), 2, nullptr, 0);
    Qnn_Tensor_t *tensors[] = {&m->input, &m->residual, &gateW, &upW, &downW,
                               &m->output, &gateOut, &upOut, &sigmoidOut, &siluOut, &gatedOut, &downOut};
    for (auto *t : tensors) {
        status = api.tensorCreateGraphTensor(m->graph, t);
        if (status != QNN_SUCCESS) return setError(message, capacity, "QNN gated MLP tensor creation failed", status);
    }
    auto ref = [](const Qnn_Tensor_t &tensor) {
        Qnn_Tensor_t value{}; value.version = QNN_TENSOR_VERSION_1; value.v1.id = tensor.v1.id; return value;
    };
    auto addMatMul = [&](const char *nodeName, const Qnn_Tensor_t &lhs,
                         const Qnn_Tensor_t &rhs, const Qnn_Tensor_t &out) {
        Qnn_Tensor_t inputs[] = {ref(lhs), ref(rhs)}, outputRef = ref(out);
        Qnn_Param_t transpose = QNN_PARAM_INIT; transpose.paramType = QNN_PARAMTYPE_SCALAR;
        transpose.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
        transpose.scalarParam.dataType = QNN_DATATYPE_BOOL_8; transpose.scalarParam.bool8Value = 1;
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = nodeName; op.v1.packageName = "qti.aisw"; op.v1.typeName = QNN_OP_MAT_MUL;
        op.v1.numOfParams = 1; op.v1.params = &transpose;
        op.v1.numOfInputs = 2; op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
        return api.graphAddNode(m->graph, op);
    };
    auto addUnary = [&](const char *nodeName, const char *opType,
                        const Qnn_Tensor_t &in, const Qnn_Tensor_t &out) {
        Qnn_Tensor_t inputRef = ref(in), outputRef = ref(out);
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = nodeName; op.v1.packageName = "qti.aisw"; op.v1.typeName = opType;
        op.v1.numOfInputs = 1; op.v1.inputTensors = &inputRef;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
        return api.graphAddNode(m->graph, op);
    };
    auto addBinary = [&](const char *nodeName, const char *opType,
                         const Qnn_Tensor_t &a, const Qnn_Tensor_t &b, const Qnn_Tensor_t &out) {
        Qnn_Tensor_t inputs[] = {ref(a), ref(b)}, outputRef = ref(out);
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = nodeName; op.v1.packageName = "qti.aisw"; op.v1.typeName = opType;
        op.v1.numOfInputs = 2; op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
        return api.graphAddNode(m->graph, op);
    };
    status = addMatMul("gate_matmul", m->input, gateW, gateOut);
    if (status == QNN_SUCCESS) status = addMatMul("up_matmul", m->input, upW, upOut);
    if (status == QNN_SUCCESS) status = addUnary("gate_sigmoid_op", QNN_OP_SIGMOID, gateOut, sigmoidOut);
    if (status == QNN_SUCCESS) status = addBinary("silu_gate", QNN_OP_ELEMENT_WISE_MULTIPLY, gateOut, sigmoidOut, siluOut);
    if (status == QNN_SUCCESS) status = addBinary("silu_times_up", QNN_OP_ELEMENT_WISE_MULTIPLY, siluOut, upOut, gatedOut);
    if (status == QNN_SUCCESS) status = addMatMul("down_matmul", gatedOut, downW, downOut);
    if (status == QNN_SUCCESS) status = addBinary("residual_add", QNN_OP_ELEMENT_WISE_ADD, downOut, m->residual, m->output);
    if (status == QNN_SUCCESS) status = api.graphFinalize(m->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN gated MLP graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 gated MLP graph ready (%u rows, %u -> %u -> %u)",
                  rows, hidden, intermediate, hidden);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN gated MLP graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_mlp_execute(InferenceSdkQnnHtpMlp *m,
        const uint16_t *input, const uint16_t *residual, uint16_t *output,
        char *message, size_t capacity) {
    if (!m || !input || !residual || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN gated MLP execution arguments");
    const uint32_t bytes = m->rows * m->hidden * sizeof(uint16_t);
    m->input.v1.clientBuf.data = const_cast<uint16_t *>(input); m->input.v1.clientBuf.dataSize = bytes;
    m->residual.v1.clientBuf.data = const_cast<uint16_t *>(residual); m->residual.v1.clientBuf.dataSize = bytes;
    m->output.v1.clientBuf.data = output; m->output.v1.clientBuf.dataSize = bytes;
    Qnn_Tensor_t inputs[] = {m->input, m->residual};
    auto status = m->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        m->graph, inputs, 2, &m->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN gated MLP execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 gated MLP executed (%u rows)", m->rows);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_mlp_destroy(InferenceSdkQnnHtpMlp *m) { delete m; }

extern "C" int inference_sdk_qnn_htp_attention_create(InferenceSdkQnnHtpSession *s,
        const char *name, uint32_t qHeads, uint32_t kvHeads, uint32_t qRows,
        uint32_t kRows, uint32_t width, uint32_t offset, int causal,
        InferenceSdkQnnHtpAttention **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !qHeads || !kvHeads || qHeads % kvHeads || !qRows || !kRows ||
        !width || !result || !message || !capacity || (causal != 0 && causal != 1) ||
        (causal && static_cast<uint64_t>(offset) + qRows > kRows))
        return setError(message, capacity, "invalid QNN attention arguments or causal lengths");
    const uint64_t qCount = static_cast<uint64_t>(qHeads) * qRows * width;
    const uint64_t kvCount = static_cast<uint64_t>(kvHeads) * kRows * width;
    const uint64_t scoreCount = static_cast<uint64_t>(qHeads) * qRows * kRows;
    if (qCount > UINT32_MAX / sizeof(uint16_t) || kvCount > UINT32_MAX / sizeof(uint16_t) ||
        scoreCount > UINT32_MAX / sizeof(uint16_t))
        return setError(message, capacity, "QNN attention tensor exceeds client buffer size limit");
    std::unique_ptr<InferenceSdkQnnHtpAttention> owned(new (std::nothrow) InferenceSdkQnnHtpAttention());
    auto *a = owned.get();
    if (!a) return setError(message, capacity, "QNN attention allocation failed");
    a->session = s; a->qHeads = qHeads; a->kvHeads = kvHeads;
    a->qRows = qRows; a->kRows = kRows; a->width = width;
    a->qDims = {1, qHeads, qRows, width}; a->kvDims = {1, kvHeads, kRows, width};
    a->scoreDims = {1, qHeads, qRows, kRows}; a->maskDims = {1, 1, qRows, kRows};
    a->indexDims = {qHeads}; a->scaleDims = {1};
    a->headMap.resize(qHeads);
    for (uint32_t h = 0; h < qHeads; ++h) a->headMap[h] = static_cast<int32_t>(h / (qHeads / kvHeads));
    a->mask.resize(static_cast<size_t>(qRows) * kRows);
    for (uint32_t q = 0; q < qRows; ++q)
        for (uint32_t k = 0; k < kRows; ++k) {
            const bool hidden = causal && k > offset + q;
            a->mask[static_cast<size_t>(q) * kRows + k] =
                tensor_f32_to_f16(hidden ? -10000.0f : 0.0f);
        }
    a->scale = {tensor_f32_to_f16(1.0f / std::sqrt(static_cast<float>(width)))};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &a->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN attention graph creation failed", status);
    Qnn_Tensor_t expandedKey{}, expandedValue{}, scores{}, scaled{}, masked{}, probabilities{}, headIndices{}, scaleTensor{};
    initTensor(a->query, "query", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               a->qDims.data(), 4, nullptr, 0);
    initTensor(a->key, "key", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               a->kvDims.data(), 4, nullptr, 0);
    initTensor(a->value, "value", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               a->kvDims.data(), 4, nullptr, 0);
    initTensor(a->output, "attention_output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
               a->qDims.data(), 4, nullptr, 0);
    initTensor(expandedKey, "expanded_key", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, a->qDims.data(), 4, nullptr, 0);
    initTensor(expandedValue, "expanded_value", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, a->qDims.data(), 4, nullptr, 0);
    initTensor(scores, "scores", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, a->scoreDims.data(), 4, nullptr, 0);
    initTensor(scaled, "scaled_scores", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, a->scoreDims.data(), 4, nullptr, 0);
    initTensor(masked, "masked_scores", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, a->scoreDims.data(), 4, nullptr, 0);
    initTensor(probabilities, "attention_probabilities", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, a->scoreDims.data(), 4, nullptr, 0);
    initTensor(headIndices, "kv_head_indices", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32,
               a->indexDims.data(), 1, a->headMap.data(), static_cast<uint32_t>(a->headMap.size() * sizeof(int32_t)));
    initTensor(scaleTensor, "attention_scale", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               a->scaleDims.data(), 1, a->scale.data(), sizeof(uint16_t));
    initTensor(a->maskTensor, "additive_mask", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               a->maskDims.data(), 4, nullptr, 0);
    Qnn_Tensor_t *tensors[] = {&a->query, &a->key, &a->value, &a->output, &expandedKey,
        &expandedValue, &scores, &scaled, &masked, &probabilities, &headIndices, &scaleTensor, &a->maskTensor};
    for (auto *t : tensors) {
        status = api.tensorCreateGraphTensor(a->graph, t);
        if (status != QNN_SUCCESS) return setError(message, capacity, "QNN attention tensor creation failed", status);
    }
    auto ref = [](const Qnn_Tensor_t &tensor) { Qnn_Tensor_t v{}; v.version = QNN_TENSOR_VERSION_1; v.v1.id = tensor.v1.id; return v; };
    auto addNode = [&](const char *nodeName, const char *opType, Qnn_Tensor_t *inputs,
                       uint32_t inputCount, const Qnn_Tensor_t &out,
                       Qnn_Param_t *params = nullptr, uint32_t paramCount = 0) {
        Qnn_Tensor_t outputRef = ref(out);
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = nodeName; op.v1.packageName = "qti.aisw"; op.v1.typeName = opType;
        op.v1.numOfParams = paramCount; op.v1.params = params;
        op.v1.numOfInputs = inputCount; op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
        return api.graphAddNode(a->graph, op);
    };
    Qnn_Tensor_t gatherKeyInputs[] = {ref(a->key), ref(headIndices)};
    status = addNode("gather_key_heads", QNN_OP_GATHER, gatherKeyInputs, 2, expandedKey,
        []() { static Qnn_Param_t p = QNN_PARAM_INIT; p.paramType = QNN_PARAMTYPE_SCALAR;
               p.name = QNN_OP_GATHER_PARAM_AXIS; p.scalarParam.dataType = QNN_DATATYPE_INT_32;
               p.scalarParam.int32Value = 1; return &p; }(), 1);
    Qnn_Tensor_t gatherValueInputs[] = {ref(a->value), ref(headIndices)};
    if (status == QNN_SUCCESS) status = addNode("gather_value_heads", QNN_OP_GATHER, gatherValueInputs, 2, expandedValue,
        []() { static Qnn_Param_t p = QNN_PARAM_INIT; p.paramType = QNN_PARAMTYPE_SCALAR;
               p.name = QNN_OP_GATHER_PARAM_AXIS; p.scalarParam.dataType = QNN_DATATYPE_INT_32;
               p.scalarParam.int32Value = 1; return &p; }(), 1);
    Qnn_Tensor_t matmulInputs[] = {ref(a->query), ref(expandedKey)};
    Qnn_Param_t transpose = QNN_PARAM_INIT; transpose.paramType = QNN_PARAMTYPE_SCALAR;
    transpose.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    transpose.scalarParam.dataType = QNN_DATATYPE_BOOL_8; transpose.scalarParam.bool8Value = 1;
    if (status == QNN_SUCCESS) status = addNode("attention_qk_matmul", QNN_OP_MAT_MUL, matmulInputs, 2, scores, &transpose, 1);
    Qnn_Tensor_t scaleInputs[] = {ref(scores), ref(scaleTensor)};
    if (status == QNN_SUCCESS) status = addNode("attention_scale_scores", QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                 scaleInputs, 2, scaled);
    Qnn_Tensor_t maskInputs[] = {ref(scaled), ref(a->maskTensor)};
    if (status == QNN_SUCCESS) status = addNode("attention_apply_mask", QNN_OP_ELEMENT_WISE_ADD, maskInputs, 2, masked);
    Qnn_Tensor_t softmaxInput = ref(masked);
    Qnn_Param_t axis = QNN_PARAM_INIT; axis.paramType = QNN_PARAMTYPE_SCALAR;
    axis.name = QNN_OP_SOFTMAX_PARAM_AXIS; axis.scalarParam.dataType = QNN_DATATYPE_INT_32; axis.scalarParam.int32Value = 3;
    if (status == QNN_SUCCESS) status = addNode("attention_softmax", QNN_OP_SOFTMAX, &softmaxInput, 1, probabilities, &axis, 1);
    Qnn_Tensor_t pvInputs[] = {ref(probabilities), ref(expandedValue)};
    if (status == QNN_SUCCESS) status = addNode("attention_pv_matmul", QNN_OP_MAT_MUL, pvInputs, 2, a->output);
    if (status == QNN_SUCCESS) status = api.graphFinalize(a->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN attention graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 %s attention graph ready (Q=%u heads, KV=%u heads, rows=%u/%u)",
                  causal ? "causal" : "bidirectional", qHeads, kvHeads, qRows, kRows);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN attention graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_attention_execute_masked(InferenceSdkQnnHtpAttention *a,
        const uint16_t *query, const uint16_t *key, const uint16_t *value,
        const uint16_t *additiveMask, uint16_t *output, char *message, size_t capacity) {
    if (!a || !query || !key || !value || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN attention execution arguments");
    const uint16_t *mask = additiveMask ? additiveMask : a->mask.data();
    a->query.v1.clientBuf.data = const_cast<uint16_t *>(query);
    a->query.v1.clientBuf.dataSize = a->qHeads * a->qRows * a->width * sizeof(uint16_t);
    a->key.v1.clientBuf.data = const_cast<uint16_t *>(key);
    a->key.v1.clientBuf.dataSize = a->kvHeads * a->kRows * a->width * sizeof(uint16_t);
    a->value.v1.clientBuf.data = const_cast<uint16_t *>(value);
    a->value.v1.clientBuf.dataSize = a->kvHeads * a->kRows * a->width * sizeof(uint16_t);
    a->maskTensor.v1.clientBuf.data = const_cast<uint16_t *>(mask);
    a->maskTensor.v1.clientBuf.dataSize = a->qRows * a->kRows * sizeof(uint16_t);
    a->output.v1.clientBuf.data = output;
    a->output.v1.clientBuf.dataSize = a->qHeads * a->qRows * a->width * sizeof(uint16_t);
    Qnn_Tensor_t inputs[] = {a->query, a->key, a->value, a->maskTensor};
    auto status = a->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        a->graph, inputs, 4, &a->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN attention execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 attention executed (%u query rows, %u key rows)", a->qRows, a->kRows);
    return 1;
}

extern "C" int inference_sdk_qnn_htp_attention_execute(InferenceSdkQnnHtpAttention *a,
        const uint16_t *query, const uint16_t *key, const uint16_t *value,
        uint16_t *output, char *message, size_t capacity) {
    return inference_sdk_qnn_htp_attention_execute_masked(a, query, key, value,
        nullptr, output, message, capacity);
}

extern "C" void inference_sdk_qnn_htp_attention_destroy(InferenceSdkQnnHtpAttention *a) { delete a; }

extern "C" int inference_sdk_qnn_htp_layer_norm_create(InferenceSdkQnnHtpSession *s,
        const char *name, const uint16_t *scale, const uint16_t *bias,
        uint32_t rows, uint32_t width, float epsilon,
        InferenceSdkQnnHtpLayerNorm **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !scale || !bias || !rows || !width || !(epsilon > 0.0f) ||
        !result || !message || !capacity || static_cast<uint64_t>(rows) * width > UINT32_MAX / 2)
        return setError(message, capacity, "invalid QNN LayerNorm arguments");
    std::unique_ptr<InferenceSdkQnnHtpLayerNorm> owned(new (std::nothrow) InferenceSdkQnnHtpLayerNorm());
    auto *n = owned.get();
    if (!n) return setError(message, capacity, "QNN LayerNorm allocation failed");
    n->session = s; n->rows = rows; n->width = width;
    n->scale.assign(scale, scale + width); n->bias.assign(bias, bias + width);
    n->inputDims = {rows, width}; n->outputDims = {rows, width}; n->channelDims = {width};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &n->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN LayerNorm graph creation failed", status);
    Qnn_Tensor_t scaleTensor{}, biasTensor{};
    initTensor(n->input, "input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
               n->inputDims.data(), 2, nullptr, 0);
    initTensor(scaleTensor, "scale", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               n->channelDims.data(), 1, n->scale.data(), width * sizeof(uint16_t));
    initTensor(biasTensor, "bias", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16,
               n->channelDims.data(), 1, n->bias.data(), width * sizeof(uint16_t));
    initTensor(n->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
               n->outputDims.data(), 2, nullptr, 0);
    status = api.tensorCreateGraphTensor(n->graph, &n->input);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(n->graph, &scaleTensor);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(n->graph, &biasTensor);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(n->graph, &n->output);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN LayerNorm tensor creation failed", status);
    Qnn_Tensor_t inputRef{}, scaleRef{}, biasRef{}, outputRef{};
    inputRef.version = scaleRef.version = biasRef.version = outputRef.version = QNN_TENSOR_VERSION_1;
    inputRef.v1.id = n->input.v1.id; scaleRef.v1.id = scaleTensor.v1.id;
    biasRef.v1.id = biasTensor.v1.id; outputRef.v1.id = n->output.v1.id;
    uint32_t axisDims[] = {1}; uint32_t axisValue = 1;
    Qnn_Tensor_t axes{};
    initTensor(axes, "axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, axisDims, 1, &axisValue, sizeof(axisValue));
    status = api.tensorCreateGraphTensor(n->graph, &axes);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN LayerNorm axes tensor creation failed", status);
    Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    params[0].paramType = QNN_PARAMTYPE_TENSOR; params[0].name = QNN_OP_LAYER_NORM_PARAM_AXES;
    params[0].tensorParam = axes;
    params[1].paramType = QNN_PARAMTYPE_SCALAR; params[1].name = QNN_OP_LAYER_NORM_PARAM_EPSILON;
    params[1].scalarParam.dataType = QNN_DATATYPE_FLOAT_32; params[1].scalarParam.floatValue = epsilon;
    Qnn_Tensor_t inputs[] = {inputRef, scaleRef, biasRef};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "layer_norm"; op.v1.packageName = "qti.aisw"; op.v1.typeName = QNN_OP_LAYER_NORM;
    op.v1.numOfParams = 2; op.v1.params = params; op.v1.numOfInputs = 3; op.v1.inputTensors = inputs;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &outputRef;
    status = api.graphAddNode(n->graph, op);
    if (status == QNN_SUCCESS) status = api.graphFinalize(n->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN LayerNorm graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 LayerNorm graph ready (%u rows x %u, epsilon=%g)", rows, width, epsilon);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN LayerNorm graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_layer_norm_execute(InferenceSdkQnnHtpLayerNorm *n,
        const uint16_t *input, uint16_t *output, char *message, size_t capacity) {
    if (!n || !input || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN LayerNorm execution arguments");
    const uint32_t bytes = n->rows * n->width * sizeof(uint16_t);
    n->input.v1.clientBuf.data = const_cast<uint16_t *>(input); n->input.v1.clientBuf.dataSize = bytes;
    n->output.v1.clientBuf.data = output; n->output.v1.clientBuf.dataSize = bytes;
    auto status = n->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        n->graph, &n->input, 1, &n->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN LayerNorm execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 LayerNorm executed (%u rows)", n->rows);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_layer_norm_destroy(InferenceSdkQnnHtpLayerNorm *n) { delete n; }

extern "C" int inference_sdk_qnn_htp_linear_bias_create(InferenceSdkQnnHtpSession *s,
        const char *name, const uint16_t *weights, const uint16_t *bias,
        uint32_t rows, uint32_t inputWidth, uint32_t outputWidth,
        InferenceSdkQnnHtpLinearBias **result, char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !weights || !bias || !rows || !inputWidth || !outputWidth ||
        !result || !message || !capacity || static_cast<uint64_t>(inputWidth) * outputWidth > UINT32_MAX / 2 ||
        static_cast<uint64_t>(rows) * std::max(inputWidth, outputWidth) > UINT32_MAX / 2)
        return setError(message, capacity, "invalid QNN biased linear arguments");
    std::unique_ptr<InferenceSdkQnnHtpLinearBias> owned(new (std::nothrow) InferenceSdkQnnHtpLinearBias());
    auto *l = owned.get();
    if (!l) return setError(message, capacity, "QNN biased linear allocation failed");
    l->session = s; l->rows = rows; l->inputWidth = inputWidth; l->outputWidth = outputWidth;
    l->weights.assign(weights, weights + static_cast<size_t>(inputWidth) * outputWidth);
    l->bias.assign(bias, bias + outputWidth);
    l->inputDims = {rows, inputWidth}; l->weightDims = {outputWidth, inputWidth};
    l->outputDims = {rows, outputWidth}; l->biasDims = {1, outputWidth};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &l->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN biased linear graph creation failed", status);
    Qnn_Tensor_t weightsTensor{}, biasTensor{}, matmulOut{};
    std::vector<uint32_t> outDims = l->outputDims;
    initTensor(l->input, "input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, l->inputDims.data(), 2, nullptr, 0);
    initTensor(weightsTensor, "weight", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, l->weightDims.data(), 2,
               l->weights.data(), static_cast<uint32_t>(l->weights.size() * sizeof(uint16_t)));
    initTensor(biasTensor, "bias", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, l->biasDims.data(), 2,
               l->bias.data(), static_cast<uint32_t>(l->bias.size() * sizeof(uint16_t)));
    initTensor(matmulOut, "matmul_output", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, outDims.data(), 2, nullptr, 0);
    initTensor(l->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, l->outputDims.data(), 2, nullptr, 0);
    Qnn_Tensor_t *tensors[] = {&l->input, &weightsTensor, &biasTensor, &matmulOut, &l->output};
    for (auto *t : tensors) {
        status = api.tensorCreateGraphTensor(l->graph, t);
        if (status != QNN_SUCCESS) return setError(message, capacity, "QNN biased linear tensor creation failed", status);
    }
    auto ref = [](const Qnn_Tensor_t &t) { Qnn_Tensor_t v{}; v.version = QNN_TENSOR_VERSION_1; v.v1.id = t.v1.id; return v; };
    Qnn_Tensor_t mmInputs[] = {ref(l->input), ref(weightsTensor)}, mmOut = ref(matmulOut);
    Qnn_Param_t transpose = QNN_PARAM_INIT; transpose.paramType = QNN_PARAMTYPE_SCALAR;
    transpose.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1; transpose.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    transpose.scalarParam.bool8Value = 1;
    Qnn_OpConfig_t mm = QNN_OPCONFIG_INIT;
    mm.v1.name = "linear_matmul"; mm.v1.packageName = "qti.aisw"; mm.v1.typeName = QNN_OP_MAT_MUL;
    mm.v1.numOfParams = 1; mm.v1.params = &transpose; mm.v1.numOfInputs = 2; mm.v1.inputTensors = mmInputs;
    mm.v1.numOfOutputs = 1; mm.v1.outputTensors = &mmOut;
    status = api.graphAddNode(l->graph, mm);
    Qnn_Tensor_t addInputs[] = {ref(matmulOut), ref(biasTensor)}, addOut = ref(l->output);
    Qnn_OpConfig_t add = QNN_OPCONFIG_INIT;
    add.v1.name = "linear_bias_add"; add.v1.packageName = "qti.aisw"; add.v1.typeName = QNN_OP_ELEMENT_WISE_ADD;
    add.v1.numOfInputs = 2; add.v1.inputTensors = addInputs; add.v1.numOfOutputs = 1; add.v1.outputTensors = &addOut;
    if (status == QNN_SUCCESS) status = api.graphAddNode(l->graph, add);
    if (status == QNN_SUCCESS) status = api.graphFinalize(l->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN biased linear graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 biased linear graph ready (%u rows, %u -> %u)", rows, inputWidth, outputWidth);
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN biased linear graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_linear_bias_execute(InferenceSdkQnnHtpLinearBias *l,
        const uint16_t *input, uint16_t *output, char *message, size_t capacity) {
    if (!l || !input || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN biased linear execution arguments");
    l->input.v1.clientBuf.data = const_cast<uint16_t *>(input);
    l->input.v1.clientBuf.dataSize = l->rows * l->inputWidth * sizeof(uint16_t);
    l->output.v1.clientBuf.data = output;
    l->output.v1.clientBuf.dataSize = l->rows * l->outputWidth * sizeof(uint16_t);
    auto status = l->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(
        l->graph, &l->input, 1, &l->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN biased linear execution failed", status);
    std::snprintf(message, capacity, "QNN FP16 biased linear executed (%u rows)", l->rows);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_linear_bias_destroy(InferenceSdkQnnHtpLinearBias *l) { delete l; }

extern "C" int inference_sdk_qnn_htp_elementwise_create(InferenceSdkQnnHtpSession *s,
        const char *name, uint32_t rows, uint32_t width, uint32_t rhsRows,
        InferenceSdkQnnHtpElementwiseOp operation, InferenceSdkQnnHtpElementwise **result,
        char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !rows || !width || (rhsRows != 1 && rhsRows != rows) ||
        operation < INFERENCE_SDK_QNN_HTP_ADD || operation > INFERENCE_SDK_QNN_HTP_SILU_MULTIPLY ||
        !result || !message || !capacity || static_cast<uint64_t>(rows) * width > UINT32_MAX / 2)
        return setError(message, capacity, "invalid QNN elementwise arguments");
    std::unique_ptr<InferenceSdkQnnHtpElementwise> owned(new (std::nothrow) InferenceSdkQnnHtpElementwise());
    auto *e = owned.get();
    if (!e) return setError(message, capacity, "QNN elementwise allocation failed");
    e->session = s; e->rows = rows; e->rhsRows = rhsRows; e->width = width; e->operation = operation;
    e->lhsDims = {rows, width}; e->rhsDims = {rhsRows, width}; e->outputDims = e->lhsDims;
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &e->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN elementwise graph creation failed", status);
    initTensor(e->lhs, "lhs", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, e->lhsDims.data(), 2, nullptr, 0);
    initTensor(e->rhs, "rhs", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, e->rhsDims.data(), 2, nullptr, 0);
    initTensor(e->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, e->outputDims.data(), 2, nullptr, 0);
    Qnn_Tensor_t sigmoid{}, silu{};
    initTensor(sigmoid, "sigmoid", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, e->lhsDims.data(), 2, nullptr, 0);
    initTensor(silu, "silu", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, e->lhsDims.data(), 2, nullptr, 0);
    Qnn_Tensor_t *tensors[] = {&e->lhs, &e->rhs, &e->output};
    if (operation == INFERENCE_SDK_QNN_HTP_SILU_MULTIPLY) {
        Qnn_Tensor_t *extra[] = {&sigmoid, &silu};
        for (auto *t : extra) { status = api.tensorCreateGraphTensor(e->graph, t); if (status != QNN_SUCCESS) break; }
    }
    for (auto *t : tensors) {
        if (status != QNN_SUCCESS) break;
        status = api.tensorCreateGraphTensor(e->graph, t);
    }
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN elementwise tensor creation failed", status);
    auto ref = [](const Qnn_Tensor_t &t) { Qnn_Tensor_t v{}; v.version = QNN_TENSOR_VERSION_1; v.v1.id = t.v1.id; return v; };
    auto add = [&](const char *nodeName, const char *type, Qnn_Tensor_t *inputs,
                   uint32_t inputCount, const Qnn_Tensor_t &output) {
        Qnn_Tensor_t out = ref(output); Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = nodeName; op.v1.packageName = "qti.aisw"; op.v1.typeName = type;
        op.v1.numOfInputs = inputCount; op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &out;
        return api.graphAddNode(e->graph, op);
    };
    if (operation == INFERENCE_SDK_QNN_HTP_SILU_MULTIPLY) {
        Qnn_Tensor_t x = ref(e->lhs);
        status = add("silu_sigmoid", QNN_OP_SIGMOID, &x, 1, sigmoid);
        Qnn_Tensor_t mulInputs[] = {ref(e->lhs), ref(sigmoid)};
        if (status == QNN_SUCCESS) status = add("silu_gate", QNN_OP_ELEMENT_WISE_MULTIPLY, mulInputs, 2, silu);
        Qnn_Tensor_t gateInputs[] = {ref(silu), ref(e->rhs)};
        if (status == QNN_SUCCESS) status = add("silu_multiply", QNN_OP_ELEMENT_WISE_MULTIPLY, gateInputs, 2, e->output);
    } else {
        Qnn_Tensor_t inputs[] = {ref(e->lhs), ref(e->rhs)};
        const char *type = operation == INFERENCE_SDK_QNN_HTP_ADD ? QNN_OP_ELEMENT_WISE_ADD : QNN_OP_ELEMENT_WISE_MULTIPLY;
        status = add("elementwise", type, inputs, 2, e->output);
    }
    if (status == QNN_SUCCESS) status = api.graphFinalize(e->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN elementwise graph compile failed", status);
    *result = owned.release();
    std::snprintf(message, capacity, "QNN FP16 elementwise graph ready (%u x %u, rhs_rows=%u, op=%d)",
                  rows, width, rhsRows, static_cast<int>(operation));
    return 1;
} catch (...) {
    return setError(message, capacity, "QNN elementwise graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_elementwise_execute(InferenceSdkQnnHtpElementwise *e,
        const uint16_t *lhs, const uint16_t *rhs, uint16_t *output,
        char *message, size_t capacity) {
    if (!e || !lhs || !rhs || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN elementwise execution arguments");
    e->lhs.v1.clientBuf.data = const_cast<uint16_t *>(lhs);
    e->lhs.v1.clientBuf.dataSize = e->rows * e->width * sizeof(uint16_t);
    e->rhs.v1.clientBuf.data = const_cast<uint16_t *>(rhs);
    e->rhs.v1.clientBuf.dataSize = e->rhsRows * e->width * sizeof(uint16_t);
    e->output.v1.clientBuf.data = output;
    e->output.v1.clientBuf.dataSize = e->rows * e->width * sizeof(uint16_t);
    Qnn_Tensor_t inputs[] = {e->lhs, e->rhs};
    auto status = e->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(e->graph, inputs, 2, &e->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN elementwise execution failed", status);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_elementwise_destroy(InferenceSdkQnnHtpElementwise *e) { delete e; }

extern "C" int inference_sdk_qnn_htp_gelu_create(InferenceSdkQnnHtpSession *s,
        const char *name, uint32_t rows, uint32_t width, InferenceSdkQnnHtpGelu **result,
        char *message, size_t capacity) try {
    if (result) *result = nullptr;
    if (!s || !name || !rows || !width || !result || !message || !capacity ||
        static_cast<uint64_t>(rows) * width > UINT32_MAX / 2)
        return setError(message, capacity, "invalid QNN GELU arguments");
    std::unique_ptr<InferenceSdkQnnHtpGelu> owned(new (std::nothrow) InferenceSdkQnnHtpGelu());
    auto *g = owned.get();
    if (!g) return setError(message, capacity, "QNN GELU allocation failed");
    g->session = s; g->rows = rows; g->width = width; g->dims = {rows, width};
    const auto &api = s->provider->QNN_INTERFACE_VER_NAME;
    auto status = api.graphCreate(s->context, name, nullptr, &g->graph);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN GELU graph creation failed", status);
    initTensor(g->input, "input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, g->dims.data(), 2, nullptr, 0);
    initTensor(g->output, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, g->dims.data(), 2, nullptr, 0);
    status = api.tensorCreateGraphTensor(g->graph, &g->input);
    if (status == QNN_SUCCESS) status = api.tensorCreateGraphTensor(g->graph, &g->output);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN GELU tensor creation failed", status);
    Qnn_Tensor_t in{}, out{}; in.version = out.version = QNN_TENSOR_VERSION_1;
    in.v1.id = g->input.v1.id; out.v1.id = g->output.v1.id;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT; op.v1.name = "gelu"; op.v1.packageName = "qti.aisw";
    op.v1.typeName = QNN_OP_GELU; op.v1.numOfInputs = 1; op.v1.inputTensors = &in;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = &out;
    status = api.graphAddNode(g->graph, op);
    if (status == QNN_SUCCESS) status = api.graphFinalize(g->graph, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN GELU graph compile failed", status);
    *result = owned.release(); std::snprintf(message, capacity, "QNN FP16 GELU graph ready (%u x %u)", rows, width); return 1;
} catch (...) {
    return setError(message, capacity, "QNN GELU graph creation failed with an allocation/runtime exception");
}

extern "C" int inference_sdk_qnn_htp_gelu_execute(InferenceSdkQnnHtpGelu *g,
        const uint16_t *input, uint16_t *output, char *message, size_t capacity) {
    if (!g || !input || !output || !message || !capacity)
        return setError(message, capacity, "invalid QNN GELU execution arguments");
    const uint32_t bytes = g->rows * g->width * sizeof(uint16_t);
    g->input.v1.clientBuf.data = const_cast<uint16_t *>(input); g->input.v1.clientBuf.dataSize = bytes;
    g->output.v1.clientBuf.data = output; g->output.v1.clientBuf.dataSize = bytes;
    auto status = g->session->provider->QNN_INTERFACE_VER_NAME.graphExecute(g->graph, &g->input, 1, &g->output, 1, nullptr, nullptr);
    if (status != QNN_SUCCESS) return setError(message, capacity, "QNN GELU execution failed", status);
    return 1;
}

extern "C" void inference_sdk_qnn_htp_gelu_destroy(InferenceSdkQnnHtpGelu *g) { delete g; }
