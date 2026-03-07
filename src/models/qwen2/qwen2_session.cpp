#include "qwen2_session.hpp"

namespace llaisys::models::qwen2 {

Qwen2Session::Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device, int device_id, std::shared_ptr<BlockManager> block_manager)
    : _block_manager(block_manager) {
    
    // Allocate scratchpad tensors (same as before)
    _token_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    _pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    _hidden = Tensor::create({1, config.hidden_size}, config.dtype, device, device_id);
    _attn_norm = Tensor::create({1, config.hidden_size}, config.dtype, device, device_id);
    _q_proj = Tensor::create({1, config.num_heads * config.head_size}, config.dtype, device, device_id);
    _k_proj = Tensor::create({1, config.nkvhead * config.head_size}, config.dtype, device, device_id);
    _v_proj = Tensor::create({1, config.nkvhead * config.head_size}, config.dtype, device, device_id);
    
    _q_view = _q_proj->view({1, config.num_heads, config.head_size});
    _k_view = _k_proj->view({1, config.nkvhead, config.head_size});
    _v_view = _v_proj->view({1, config.nkvhead, config.head_size});
    
    _q_rope = Tensor::create({1, config.num_heads, config.head_size}, config.dtype, device, device_id);
    _k_rope = Tensor::create({1, config.nkvhead, config.head_size}, config.dtype, device, device_id);
    _attn_out = Tensor::create({1, config.num_heads, config.head_size}, config.dtype, device, device_id);
    _attn_out_flat = _attn_out->view({1, config.hidden_size});
    _attn_proj = Tensor::create({1, config.hidden_size}, config.dtype, device, device_id);
    _mlp_norm = Tensor::create({1, config.hidden_size}, config.dtype, device, device_id);
    _mlp_gate = Tensor::create({1, config.intermediate_size}, config.dtype, device, device_id);
    _mlp_up = Tensor::create({1, config.intermediate_size}, config.dtype, device, device_id);
    _mlp_act = Tensor::create({1, config.intermediate_size}, config.dtype, device, device_id);
    _mlp_down = Tensor::create({1, config.hidden_size}, config.dtype, device, device_id);
    _final_norm = Tensor::create({1, config.hidden_size}, config.dtype, device, device_id);
    _logits = Tensor::create({1, config.vocab_size}, config.dtype, device, device_id);
    _logits_flat = _logits->view({config.vocab_size});
    _max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    _max_val = Tensor::create({1}, config.dtype, device, device_id);
}

Qwen2Session::~Qwen2Session() {
    reset();
}

void Qwen2Session::reset() {
    if (_block_manager) {
        for (auto &block : _blocks) {
            _block_manager->free(block);
        }
    }
    _blocks.clear();
    _seq_len = 0;
}

void Qwen2Session::ensure_capacity_for_next_token() {
    if (_blocks.empty() || _blocks.back()->used >= _blocks.back()->size) {
        auto new_block = _block_manager->allocate();
        if (!new_block) {
            throw std::runtime_error("OOM: Failed to allocate KV Cache block");
        }
        _blocks.push_back(new_block);
    }
}

void Qwen2Session::write_kv(int layer, tensor_t k_slice, tensor_t v_slice) {
    if (_blocks.empty()) return;
    auto current_block = _blocks.back();
    size_t offset = current_block->used;
    
    // Copy data to block
    // k_slice: [1, nkvhead, head_dim]
    // block k: [size, nkvhead, head_dim] -> slice(offset, offset+1)
    
    auto k_dst = current_block->k_blocks[layer]->slice(0, offset, offset + 1);
    auto v_dst = current_block->v_blocks[layer]->slice(0, offset, offset + 1);
    
    // We assume memory copy happens here
    // For CPU, we can just copy
    // k_slice is [1, nkvhead, head_dim]
    // k_dst is [1, nkvhead, head_dim] (after slice)
    
    // Using rearrange or direct copy
    llaisys::ops::rearrange(k_dst, k_slice);
    llaisys::ops::rearrange(v_dst, v_slice);
    
    // Note: we do NOT increment used here.
    // It should be incremented after ALL layers are done for this token,
    // which happens when we call `advance(1)`.
    // But `advance` is on Session, we need to sync with Block `used`.
    
    // Wait, `used` is property of Block.
    // Session should update Block's `used` when advancing.
}

void Qwen2Session::advance(size_t n) {
    _seq_len += n;
    if (!_blocks.empty()) {
        // Increment used count on the last block
        // Assuming we only append to the last block
        _blocks.back()->used += n;
        // Check overflow?
        if (_blocks.back()->used > _blocks.back()->size) {
            // Logic error if we advanced more than capacity
            // But we only call ensure_capacity_for_next_token which ensures 1 slot.
        }
    }
}

} // namespace llaisys::models::qwen2
