#pragma once

#include "llaisys/tensor.h"
#include "qwen2_kv_cache.hpp" // New header for Block Manager
#include <memory>
#include <vector>

namespace llaisys::models::qwen2 {

struct Qwen2Config;

class Qwen2Session {
public:
    Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device, int device_id, std::shared_ptr<BlockManager> block_manager);
    ~Qwen2Session();

    // Replaces contiguous KV Cache with Block-based access
    // But for backward compatibility with existing ops (that expect tensor_t),
    // we might need a way to present it as contiguous OR modify ops.
    // For P4-4, we are moving to Paged Attention, so direct access via kv_cache() 
    // returning a Qwen2KVCache object might need change.
    
    // Let's keep the old interface but implement it using blocks underneath?
    // No, Qwen2KVCache was a simple struct holding tensors.
    // Now session holds blocks directly.
    
    const std::vector<std::shared_ptr<KVCacheBlock>>& blocks() const { return _blocks; }
    
    // Add a new token to cache (allocates block if needed)
    void append_token_kv(int layer, tensor_t k_slice, tensor_t v_slice);
    
    // Get total sequence length
    size_t seq_len() const { return _seq_len; }
    
    // Reserve slot for next token
    void ensure_capacity_for_next_token();
    
    // Write KV for specific layer to current slot
    void write_kv(int layer, tensor_t k_slice, tensor_t v_slice);
    
    // Advance sequence length
    void advance(size_t n) { _seq_len += n; }
    
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
    
    // For now, remove the old Qwen2KVCache member
    // Qwen2KVCache _kv_cache; 
};

} // namespace llaisys::models::qwen2
