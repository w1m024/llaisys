#pragma once

#include "../../tensor/tensor.hpp"
#include "qwen2_kv_cache.hpp"
#include <memory>
#include <vector>

namespace llaisys::models::qwen2 {

struct Qwen2Config;

class Qwen2Session {
public:
    Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device, int device_id, std::shared_ptr<BlockManager> block_manager);
    ~Qwen2Session();
    
    const std::vector<std::shared_ptr<KVCacheBlock>>& blocks() const { return _blocks; }

    size_t seq_len() const { return _seq_len; }

    void ensure_capacity_for_next_token();
    void write_kv(int layer, tensor_t k_slice, tensor_t v_slice);
    void advance(size_t n);
    void reset();

    // Intermediate buffers (keep same)
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
    std::shared_ptr<BlockManager> _block_manager;
    std::vector<std::shared_ptr<KVCacheBlock>> _blocks;
    size_t _seq_len = 0;
};

} // namespace llaisys::models::qwen2
