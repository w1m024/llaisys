#pragma once

#include <memory>
#include <vector>
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"
#include "qwen2_kvcache.hpp"

namespace llaisys::models::qwen2 {

struct Qwen2Config;

class Qwen2Session {
public:
    Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device, int device_id);
    ~Qwen2Session() = default;

    Qwen2KVCache &kv_cache() { return _kv_cache; }
    const Qwen2KVCache &kv_cache() const { return _kv_cache; }

    void reset() { _kv_cache.reset(); }

    // Intermediate buffers moved from Qwen2Model
    tensor_t _token_ids;
    tensor_t _pos_ids;
    tensor_t _hidden;
    tensor_t _attn_norm;
    tensor_t _q_proj;
    tensor_t _k_proj;
    tensor_t _v_proj;
    tensor_t _q_view;
    tensor_t _k_view;
    tensor_t _v_view;
    tensor_t _q_rope;
    tensor_t _k_rope;
    tensor_t _attn_out;
    tensor_t _attn_out_flat;
    tensor_t _attn_proj;
    tensor_t _mlp_norm;
    tensor_t _mlp_gate;
    tensor_t _mlp_up;
    tensor_t _mlp_act;
    tensor_t _mlp_down;
    tensor_t _final_norm;
    tensor_t _logits;
    tensor_t _logits_flat;
    tensor_t _max_idx;
    tensor_t _max_val;

private:
    Qwen2KVCache _kv_cache;
};

} // namespace llaisys::models::qwen2
