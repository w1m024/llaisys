#pragma once

#include "../../utils.hpp"
#include "../../tensor/tensor.hpp"
#include "qwen2_session.hpp"

#include "llaisys/models/qwen2.h"

#include <vector>

namespace llaisys::models::qwen2 {

struct Qwen2Config {
    size_t nlayers;
    size_t maxseq;
    size_t nkvhead;
    size_t head_size;
    size_t hidden_size;
    size_t num_heads;
    size_t intermediate_size;
    size_t vocab_size;
    llaisysDataType_t dtype;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);

    const LlaisysQwen2Meta &meta() const { return _meta; }
    llaisysDeviceType_t device() const { return _device; }
    int device_id() const { return _device_id; }

    Qwen2Session *create_session();

    void bind_weights(const LlaisysQwen2Weights &weights);
    int64_t infer(Qwen2Session *session, const int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature, int64_t seed = -1);
    
    // Batch Inference API
    std::vector<int64_t> infer_batch(
        const std::vector<Qwen2Session*> &sessions,
        const std::vector<std::vector<int64_t>> &batch_token_ids,
        const std::vector<int> &top_ks,
        const std::vector<float> &top_ps,
        const std::vector<float> &temperatures,
        const std::vector<int64_t> &seeds
    );

private:
    void process_token(Qwen2Session *session, int64_t token_id);
    // Batch processing helper
    void process_batch(const std::vector<Qwen2Session*> &sessions, const std::vector<int64_t> &token_ids);

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    Qwen2Weights _weights;
    bool _weights_bound = false;

    float _attn_scale = 1.0f;
};

} // namespace llaisys::models::qwen2
