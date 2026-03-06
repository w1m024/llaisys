#include "qwen2_session.hpp"
#include "qwen2_model.hpp"

namespace llaisys::models::qwen2 {

Qwen2Session::Qwen2Session(const Qwen2Config &config, llaisysDeviceType_t device, int device_id) {
    // Use config.dtype instead of hardcoded BF16 if appropriate, 
    // but previous code had LLAISYS_DTYPE_BF16. 
    // Let's trust the previous code used BF16 for KV cache optimization or standard.
    // However, usually KV cache matches model dtype. 
    // Let's use config.dtype to be safe and consistent with other tensors.
    _kv_cache.reserve(config.nlayers, config.maxseq, config.nkvhead, config.head_size, config.dtype, device, device_id);

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

} // namespace llaisys::models::qwen2
