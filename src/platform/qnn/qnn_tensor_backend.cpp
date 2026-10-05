#include "qnn_tensor_backend.h"
#include "qnn_htp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
std::mutex g_mutex;
InferenceSdkQnnHtpSession *g_session = nullptr;
uint32_t g_next_graph = 1;
constexpr uint32_t kMaxPositions = 8192;
char g_error[512]{};

struct LinearEntry { const void *w; uint32_t r, i, o; InferenceSdkQnnHtpLinear *graph; };
struct BiasedLinearEntry { const void *w, *b; uint32_t r, i, o; InferenceSdkQnnHtpLinearBias *graph; };
struct NormEntry { const void *s; const void *b; uint32_t r, w; float e; bool layer; void *graph; };
struct ElementEntry { uint32_t r, w, rr; InferenceSdkQnnHtpElementwiseOp op; InferenceSdkQnnHtpElementwise *graph; };
struct GeluEntry { uint32_t r, w; InferenceSdkQnnHtpGelu *graph; };
struct EmbeddingEntry { const void *table; uint32_t r, w, v; InferenceSdkQnnHtpEmbedding *graph; };
struct RopeEntry { uint32_t h, r, w; float theta; InferenceSdkQnnHtpRope *graph; };
struct AttentionEntry { uint32_t qh, kh, qr, kr, d; bool causal; InferenceSdkQnnHtpAttention *graph; };
struct MlpEntry { const void *w1, *b1, *w2, *b2; uint32_t r, i, m, o; InferenceSdkQnnHtpLinearBias *first, *second; InferenceSdkQnnHtpGelu *gelu; };
struct QkvEntry { const void *wq, *wk, *wv, *bq, *bk, *bv; uint32_t r, i, o; InferenceSdkQnnHtpLinearBias *q, *k, *v; };

std::vector<LinearEntry> g_linears;
std::vector<BiasedLinearEntry> g_biased;
std::vector<NormEntry> g_norms;
std::vector<ElementEntry> g_elements;
std::vector<GeluEntry> g_gelu;
std::vector<EmbeddingEntry> g_embeddings;
std::vector<RopeEntry> g_ropes;
std::vector<AttentionEntry> g_attention;
std::vector<MlpEntry> g_mlps;
std::vector<QkvEntry> g_qkvs;

void graph_name(char *name, size_t capacity, const char *kind) {
    std::snprintf(name, capacity, "sdk_%s_%u", kind, g_next_graph++);
}

void clear_graphs() {
    for (auto &e : g_linears) inference_sdk_qnn_htp_linear_destroy(e.graph);
    for (auto &e : g_biased) inference_sdk_qnn_htp_linear_bias_destroy(e.graph);
    for (auto &e : g_norms) {
        if (e.layer) inference_sdk_qnn_htp_layer_norm_destroy(static_cast<InferenceSdkQnnHtpLayerNorm *>(e.graph));
        else inference_sdk_qnn_htp_rms_norm_destroy(static_cast<InferenceSdkQnnHtpRmsNorm *>(e.graph));
    }
    for (auto &e : g_elements) inference_sdk_qnn_htp_elementwise_destroy(e.graph);
    for (auto &e : g_gelu) inference_sdk_qnn_htp_gelu_destroy(e.graph);
    for (auto &e : g_embeddings) inference_sdk_qnn_htp_embedding_destroy(e.graph);
    for (auto &e : g_ropes) inference_sdk_qnn_htp_rope_destroy(e.graph);
    for (auto &e : g_attention) inference_sdk_qnn_htp_attention_destroy(e.graph);
    for (auto &e : g_mlps) {
        inference_sdk_qnn_htp_linear_bias_destroy(e.first);
        inference_sdk_qnn_htp_linear_bias_destroy(e.second);
        inference_sdk_qnn_htp_gelu_destroy(e.gelu);
    }
    for (auto &e : g_qkvs) {
        inference_sdk_qnn_htp_linear_bias_destroy(e.q);
        inference_sdk_qnn_htp_linear_bias_destroy(e.k);
        inference_sdk_qnn_htp_linear_bias_destroy(e.v);
    }
    g_linears.clear(); g_biased.clear(); g_norms.clear(); g_elements.clear();
    g_gelu.clear(); g_embeddings.clear(); g_ropes.clear(); g_attention.clear();
    g_mlps.clear(); g_qkvs.clear();
}

InferenceSdkQnnHtpLinear *linear_for(const uint16_t *w, uint32_t rows, uint32_t in, uint32_t out) {
    for (auto &e : g_linears) if (e.w == w && e.r == rows && e.i == in && e.o == out) return e.graph;
    char name[64]; graph_name(name, sizeof(name), "linear");
    InferenceSdkQnnHtpLinear *graph = nullptr;
    if (!inference_sdk_qnn_htp_linear_create(g_session, name, w, rows, in, out, &graph, g_error, sizeof(g_error))) return nullptr;
    g_linears.push_back({w, rows, in, out, graph}); return graph;
}

int run_linear(const uint16_t *input, const uint16_t *weight, uint16_t *output,
               size_t rows, size_t in, size_t out) {
    if (!g_session || !input || !weight || !output || rows > UINT32_MAX || in > UINT32_MAX || out > UINT32_MAX) return 0;
    try {
        auto *graph = linear_for(weight, static_cast<uint32_t>(rows), static_cast<uint32_t>(in), static_cast<uint32_t>(out));
        if (!graph) return 0;
        std::vector<uint16_t> temporary;
        uint16_t *target = output;
        const size_t inputBytes = rows * in * sizeof(uint16_t);
        const size_t outputBytes = rows * out * sizeof(uint16_t);
        if (input == output || (reinterpret_cast<uintptr_t>(input) < reinterpret_cast<uintptr_t>(output) + outputBytes &&
            reinterpret_cast<uintptr_t>(output) < reinterpret_cast<uintptr_t>(input) + inputBytes)) {
            temporary.resize(rows * out); target = temporary.data();
        }
        if (!inference_sdk_qnn_htp_linear_execute(graph, input, static_cast<uint32_t>(rows), target, g_error, sizeof(g_error))) return 0;
        if (!temporary.empty()) std::memcpy(output, temporary.data(), outputBytes);
        return 1;
    } catch (...) { return 0; }
}

int available() { std::lock_guard<std::mutex> lock(g_mutex); return g_session != nullptr; }

int linear_f16(const uint16_t *x, const uint16_t *w, uint16_t *y, size_t r, size_t i, size_t o) {
    std::lock_guard<std::mutex> lock(g_mutex); return run_linear(x, w, y, r, i, o);
}

int norm_f16(const uint16_t *x, const uint16_t *scale, uint16_t *y, size_t rows, size_t width, float eps) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !x || !scale || !y || rows > UINT32_MAX || width > UINT32_MAX) return 0;
    try {
        InferenceSdkQnnHtpRmsNorm *graph = nullptr;
        for (auto &e : g_norms) if (!e.layer && e.s == scale && e.r == rows && e.w == width && e.e == eps) graph = static_cast<InferenceSdkQnnHtpRmsNorm *>(e.graph);
        if (!graph) { char name[64]; graph_name(name, sizeof(name), "rms");
            if (!inference_sdk_qnn_htp_rms_norm_create(g_session, name, scale, rows, width, eps, &graph, g_error, sizeof(g_error))) return 0;
            g_norms.push_back({scale, nullptr, static_cast<uint32_t>(rows), static_cast<uint32_t>(width), eps, false, graph}); }
        std::vector<uint16_t> temp(rows * width);
        if (!inference_sdk_qnn_htp_rms_norm_execute(graph, x, rows, temp.data(), g_error, sizeof(g_error))) return 0;
        std::memcpy(y, temp.data(), temp.size() * sizeof(uint16_t)); return 1;
    } catch (...) { return 0; }
}

int layer_norm_f16(const uint16_t *x, const uint16_t *scale, const uint16_t *bias,
                   uint16_t *y, size_t rows, size_t width, float eps) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !x || !scale || !bias || !y || rows > UINT32_MAX || width > UINT32_MAX) return 0;
    try {
        InferenceSdkQnnHtpLayerNorm *graph = nullptr;
        for (auto &e : g_norms) if (e.layer && e.s == scale && e.b == bias && e.r == rows && e.w == width && e.e == eps) graph = static_cast<InferenceSdkQnnHtpLayerNorm *>(e.graph);
        if (!graph) { char name[64]; graph_name(name, sizeof(name), "layer_norm");
            if (!inference_sdk_qnn_htp_layer_norm_create(g_session, name, scale, bias, rows, width, eps, &graph, g_error, sizeof(g_error))) return 0;
            g_norms.push_back({scale, bias, static_cast<uint32_t>(rows), static_cast<uint32_t>(width), eps, true, graph}); }
        std::vector<uint16_t> temp(rows * width);
        if (!inference_sdk_qnn_htp_layer_norm_execute(graph, x, temp.data(), g_error, sizeof(g_error))) return 0;
        std::memcpy(y, temp.data(), temp.size() * sizeof(uint16_t)); return 1;
    } catch (...) { return 0; }
}

int element_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, size_t count,
                InferenceSdkQnnHtpElementwiseOp op, uint32_t rhsRows = 0, uint32_t width = 0) {
    if (!g_session || !a || !b || !y || count > UINT32_MAX) return 0;
    if (width == 0) { width = static_cast<uint32_t>(count); rhsRows = 1; }
    const uint32_t rows = static_cast<uint32_t>(count / width);
    InferenceSdkQnnHtpElementwise *graph = nullptr;
    for (auto &e : g_elements) if (e.r == rows && e.w == width && e.rr == rhsRows && e.op == op) graph = e.graph;
    if (!graph) { char name[64]; graph_name(name, sizeof(name), "elementwise");
        if (!inference_sdk_qnn_htp_elementwise_create(g_session, name, rows, width, rhsRows, op, &graph, g_error, sizeof(g_error))) return 0;
        g_elements.push_back({rows, width, rhsRows, op, graph}); }
    std::vector<uint16_t> temp(count);
    if (!inference_sdk_qnn_htp_elementwise_execute(graph, a, b, temp.data(), g_error, sizeof(g_error))) return 0;
    std::memcpy(y, temp.data(), count * sizeof(uint16_t)); return 1;
}

int residual_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, size_t n) {
    std::lock_guard<std::mutex> lock(g_mutex); try { return element_f16(a, b, y, n, INFERENCE_SDK_QNN_HTP_ADD); } catch (...) { return 0; }
}
int silu_mul_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, size_t n) {
    std::lock_guard<std::mutex> lock(g_mutex); try { return element_f16(a, b, y, n, INFERENCE_SDK_QNN_HTP_SILU_MULTIPLY); } catch (...) { return 0; }
}
int bias_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, size_t rows, size_t width) {
    std::lock_guard<std::mutex> lock(g_mutex); try { return element_f16(a, b, y, rows * width, INFERENCE_SDK_QNN_HTP_ADD, 1, static_cast<uint32_t>(width)); } catch (...) { return 0; }
}

int gelu_f16(const uint16_t *x, uint16_t *y, size_t count) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !x || !y || count == 0 || count > UINT32_MAX) return 0;
    try {
        InferenceSdkQnnHtpGelu *graph = nullptr;
        for (auto &e : g_gelu) if (e.r == count && e.w == 1) graph = e.graph;
        if (!graph) { char name[64]; graph_name(name, sizeof(name), "gelu");
            if (!inference_sdk_qnn_htp_gelu_create(g_session, name, count, 1, &graph, g_error, sizeof(g_error))) return 0;
            g_gelu.push_back({static_cast<uint32_t>(count), 1, graph}); }
        std::vector<uint16_t> temp(count);
        if (!inference_sdk_qnn_htp_gelu_execute(graph, x, temp.data(), g_error, sizeof(g_error))) return 0;
        std::memcpy(y, temp.data(), count * sizeof(uint16_t)); return 1;
    } catch (...) { return 0; }
}

int embedding_f16(const uint16_t *table, const uint32_t *ids, uint16_t *out, size_t rows, size_t width, size_t vocab) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !table || !ids || !out || rows > UINT32_MAX || width > UINT32_MAX || vocab > UINT32_MAX) return 0;
    try {
        InferenceSdkQnnHtpEmbedding *graph = nullptr;
        for (auto &e : g_embeddings) if (e.table == table && e.r == rows && e.w == width && e.v == vocab) graph = e.graph;
        if (!graph) { char name[64]; graph_name(name, sizeof(name), "embedding");
            if (!inference_sdk_qnn_htp_embedding_create(g_session, name, table, vocab, width, rows, &graph, g_error, sizeof(g_error))) return 0;
            g_embeddings.push_back({table, static_cast<uint32_t>(rows), static_cast<uint32_t>(width), static_cast<uint32_t>(vocab), graph}); }
        std::vector<int32_t> tokenIds(rows); for (size_t i = 0; i < rows; ++i) tokenIds[i] = static_cast<int32_t>(ids[i]);
        return inference_sdk_qnn_htp_embedding_execute(graph, tokenIds.data(), rows, out, g_error, sizeof(g_error));
    } catch (...) { return 0; }
}

int rope_f16(uint16_t *q, uint16_t *k, size_t seq, size_t qh, size_t kh, size_t dim, double theta, size_t offset) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !q || !k || seq == 0 || qh > UINT32_MAX || kh > UINT32_MAX || dim > UINT32_MAX || offset + seq > kMaxPositions) return 0;
    try {
        auto get = [&](uint32_t heads) -> InferenceSdkQnnHtpRope * {
            for (auto &e : g_ropes) if (e.h == heads && e.r == seq && e.w == dim && e.theta == static_cast<float>(theta)) return e.graph;
            char name[64]; graph_name(name, sizeof(name), "rope"); InferenceSdkQnnHtpRope *graph = nullptr;
            if (!inference_sdk_qnn_htp_rope_create(g_session, name, heads, seq, dim, kMaxPositions, static_cast<float>(theta), &graph, g_error, sizeof(g_error))) return nullptr;
            g_ropes.push_back({heads, static_cast<uint32_t>(seq), static_cast<uint32_t>(dim), static_cast<float>(theta), graph}); return graph;
        };
        auto *qGraph = get(static_cast<uint32_t>(qh)); auto *kGraph = get(static_cast<uint32_t>(kh));
        if (!qGraph || !kGraph) return 0;
        std::vector<uint16_t> qhMajor(seq * qh * dim), khMajor(seq * kh * dim), qOut(qhMajor.size()), kOut(khMajor.size());
        std::vector<int32_t> positions(seq); for (size_t p = 0; p < seq; ++p) positions[p] = static_cast<int32_t>(offset + p);
        auto toHeads = [&](const uint16_t *src, uint16_t *dst, size_t heads) {
            for (size_t p = 0; p < seq; ++p) for (size_t h = 0; h < heads; ++h)
                std::memcpy(dst + (h * seq + p) * dim, src + (p * heads + h) * dim, dim * sizeof(uint16_t));
        };
        auto fromHeads = [&](const uint16_t *src, uint16_t *dst, size_t heads) {
            for (size_t p = 0; p < seq; ++p) for (size_t h = 0; h < heads; ++h)
                std::memcpy(dst + (p * heads + h) * dim, src + (h * seq + p) * dim, dim * sizeof(uint16_t));
        };
        toHeads(q, qhMajor.data(), qh); toHeads(k, khMajor.data(), kh);
        if (!inference_sdk_qnn_htp_rope_execute(qGraph, qhMajor.data(), positions.data(), qOut.data(), g_error, sizeof(g_error)) ||
            !inference_sdk_qnn_htp_rope_execute(kGraph, khMajor.data(), positions.data(), kOut.data(), g_error, sizeof(g_error))) return 0;
        fromHeads(qOut.data(), q, qh); fromHeads(kOut.data(), k, kh); return 1;
    } catch (...) { return 0; }
}

int qkv_f16(const uint16_t *x, const uint16_t *wq, const uint16_t *wk, const uint16_t *wv,
            const uint16_t *bq, const uint16_t *bk, const uint16_t *bv,
            uint16_t *q, uint16_t *k, uint16_t *v, size_t rows, size_t in, size_t out) {
    if (!bq || !bk || !bv || rows > UINT32_MAX || in > UINT32_MAX || out > UINT32_MAX) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        QkvEntry *entry = nullptr;
        for (auto &e : g_qkvs) if (e.wq == wq && e.wk == wk && e.wv == wv && e.bq == bq && e.bk == bk && e.bv == bv && e.r == rows && e.i == in && e.o == out) { entry = &e; break; }
        if (!entry) {
            char name[64]; InferenceSdkQnnHtpLinearBias *qg=nullptr,*kg=nullptr,*vg=nullptr;
            graph_name(name,sizeof(name),"qkv_q"); if (!inference_sdk_qnn_htp_linear_bias_create(g_session,name,wq,bq,rows,in,out,&qg,g_error,sizeof(g_error))) return 0;
            graph_name(name,sizeof(name),"qkv_k"); if (!inference_sdk_qnn_htp_linear_bias_create(g_session,name,wk,bk,rows,in,out,&kg,g_error,sizeof(g_error))) { inference_sdk_qnn_htp_linear_bias_destroy(qg); return 0; }
            graph_name(name,sizeof(name),"qkv_v"); if (!inference_sdk_qnn_htp_linear_bias_create(g_session,name,wv,bv,rows,in,out,&vg,g_error,sizeof(g_error))) { inference_sdk_qnn_htp_linear_bias_destroy(qg); inference_sdk_qnn_htp_linear_bias_destroy(kg); return 0; }
            g_qkvs.push_back({wq,wk,wv,bq,bk,bv,static_cast<uint32_t>(rows),static_cast<uint32_t>(in),static_cast<uint32_t>(out),qg,kg,vg}); entry=&g_qkvs.back();
        }
        return inference_sdk_qnn_htp_linear_bias_execute(entry->q,x,q,g_error,sizeof(g_error)) &&
               inference_sdk_qnn_htp_linear_bias_execute(entry->k,x,k,g_error,sizeof(g_error)) &&
               inference_sdk_qnn_htp_linear_bias_execute(entry->v,x,v,g_error,sizeof(g_error));
    } catch (...) { return 0; }
}

int mlp_f16(const uint16_t *x, const uint16_t *w1, const uint16_t *b1,
            const uint16_t *w2, const uint16_t *b2, uint16_t *y,
            size_t rows, size_t in, size_t mid, size_t out) {
    if (rows > UINT32_MAX || in > UINT32_MAX || mid > UINT32_MAX || out > UINT32_MAX) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        MlpEntry *entry = nullptr;
        for (auto &e : g_mlps) if (e.w1==w1 && e.b1==b1 && e.w2==w2 && e.b2==b2 && e.r==rows && e.i==in && e.m==mid && e.o==out) { entry=&e; break; }
        if (!entry) {
            char name[64]; InferenceSdkQnnHtpLinearBias *first=nullptr,*second=nullptr; InferenceSdkQnnHtpGelu *gelu=nullptr;
            graph_name(name,sizeof(name),"vision_fc1"); if (!inference_sdk_qnn_htp_linear_bias_create(g_session,name,w1,b1,rows,in,mid,&first,g_error,sizeof(g_error))) return 0;
            graph_name(name,sizeof(name),"vision_gelu"); if (!inference_sdk_qnn_htp_gelu_create(g_session,name,rows,mid,&gelu,g_error,sizeof(g_error))) { inference_sdk_qnn_htp_linear_bias_destroy(first); return 0; }
            graph_name(name,sizeof(name),"vision_fc2"); if (!inference_sdk_qnn_htp_linear_bias_create(g_session,name,w2,b2,rows,mid,out,&second,g_error,sizeof(g_error))) { inference_sdk_qnn_htp_linear_bias_destroy(first); inference_sdk_qnn_htp_gelu_destroy(gelu); return 0; }
            g_mlps.push_back({w1,b1,w2,b2,static_cast<uint32_t>(rows),static_cast<uint32_t>(in),static_cast<uint32_t>(mid),static_cast<uint32_t>(out),first,second,gelu}); entry=&g_mlps.back();
        }
        std::vector<uint16_t> hidden(rows*mid), activated(rows*mid);
        return inference_sdk_qnn_htp_linear_bias_execute(entry->first,x,hidden.data(),g_error,sizeof(g_error)) &&
               inference_sdk_qnn_htp_gelu_execute(entry->gelu,hidden.data(),activated.data(),g_error,sizeof(g_error)) &&
               inference_sdk_qnn_htp_linear_bias_execute(entry->second,activated.data(),y,g_error,sizeof(g_error));
    } catch (...) { return 0; }
}

int argmax_f16(const uint16_t *x, size_t n, uint32_t *result) {
    if (!x || !result || n == 0 || n > UINT32_MAX) return 0;
    size_t best = 0; float bestValue = -INFINITY;
    for (size_t i=0;i<n;i++) { const uint16_t h=x[i]; const uint32_t sign=(h>>15)&1, exp=(h>>10)&31, mant=h&1023; float value;
        if (exp==0) value=std::ldexp(static_cast<float>(mant),-24); else if (exp==31) value=mant?NAN:INFINITY; else value=std::ldexp(static_cast<float>(mant+1024),static_cast<int>(exp)-25);
        if (sign) value=-value; if (value > bestValue) {bestValue=value;best=i;}}
    *result=static_cast<uint32_t>(best); return 1;
}

int attention_f16(const uint16_t *q, const uint16_t *k, const uint16_t *v, const uint16_t *mask,
                  uint16_t *out, size_t batch, size_t qh, size_t kh, size_t qr, size_t kr,
                  size_t dim, size_t vdim, float scale, int causal) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || batch != 1 || dim != vdim || qh > UINT32_MAX || kh > UINT32_MAX || qr > UINT32_MAX || kr > UINT32_MAX || dim > UINT32_MAX ||
        std::abs(scale - 1.0f/std::sqrt(static_cast<float>(dim))) > 1.0e-5f) return 0;
    try {
        AttentionEntry *entry=nullptr;
        for (auto &e:g_attention) if (e.qh==qh && e.kh==kh && e.qr==qr && e.kr==kr && e.d==dim && e.causal==(causal!=0)) {entry=&e;break;}
        if (!entry) { char name[64]; graph_name(name,sizeof(name),"attention"); InferenceSdkQnnHtpAttention *graph=nullptr;
            if (!inference_sdk_qnn_htp_attention_create(g_session,name,qh,kh,qr,kr,dim,0,causal,&graph,g_error,sizeof(g_error))) return 0;
            g_attention.push_back({static_cast<uint32_t>(qh),static_cast<uint32_t>(kh),static_cast<uint32_t>(qr),static_cast<uint32_t>(kr),static_cast<uint32_t>(dim),causal!=0,graph}); entry=&g_attention.back(); }
        return inference_sdk_qnn_htp_attention_execute_masked(entry->graph,q,k,v,mask,out,g_error,sizeof(g_error));
    } catch (...) { return 0; }
}

int attention_available() { return available(); }
void release_cache() { std::lock_guard<std::mutex> lock(g_mutex); clear_graphs(); }

const TensorComputeBackend g_compute = [] {
    TensorComputeBackend b{};
    b.name="qnn"; b.is_available=available; b.linear_f16=linear_f16;
    b.embedding_f16=embedding_f16; b.rms_norm_f16=norm_f16;
    b.layer_norm_f16=layer_norm_f16; b.bias_add_f16=bias_f16; b.gelu_f16=gelu_f16;
    b.rope_f16=rope_f16; b.silu_multiply_f16=silu_mul_f16; b.residual_add_f16=residual_f16;
    b.argmax_f16=argmax_f16; b.release_cache=release_cache;
    b.mlp_f16=mlp_f16; b.qkv_f16=qkv_f16;
    return b;
}();

const TensorAttentionBackendOps g_attention_backend = [] {
    TensorAttentionBackendOps b{};
    b.kind=TENSOR_ATTENTION_BACKEND_QNN_HTP; b.name="qnn";
    b.is_available=attention_available; b.run_f16=attention_f16;
    return b;
}();
}

extern "C" int inference_sdk_qnn_htp_backend_initialize(const char *directory,
        char *message, size_t capacity) {
    if (!directory || !message || !capacity) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_graphs();
    if (g_session) inference_sdk_qnn_htp_session_destroy(g_session);
    g_session=nullptr;
    const int ok=inference_sdk_qnn_htp_session_create(directory,&g_session,message,capacity);
    return ok;
}

extern "C" void inference_sdk_qnn_htp_backend_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex); clear_graphs();
    if (g_session) inference_sdk_qnn_htp_session_destroy(g_session); g_session=nullptr;
}
extern "C" const TensorComputeBackend *tensor_compute_qnn_htp_backend(void) { return &g_compute; }
extern "C" const TensorAttentionBackendOps *tensor_attention_qnn_htp_backend(void) { return &g_attention_backend; }
