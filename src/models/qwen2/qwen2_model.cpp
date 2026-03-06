#include "qwen2_model.hpp"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/sample/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>

namespace llaisys::models::qwen2 {
namespace {
inline tensor_t unwrap(llaisysTensor_t handle) {
    if (!handle) {
        return nullptr;
    }
    return handle->tensor;
}

void bind_layer_list(std::vector<tensor_t> &dst, llaisysTensor_t *src, size_t nlayer) {
    dst.resize(nlayer);
    if (!src) {
        for (size_t i = 0; i < nlayer; ++i) {
            dst[i] = nullptr;
        }
        return;
    }
    for (size_t i = 0; i < nlayer; ++i) {
        dst[i] = unwrap(src[i]);
    }
}
} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    CHECK_ARGUMENT(_meta.nlayer > 0, "nlayer must be > 0");
    CHECK_ARGUMENT(_meta.hs > 0, "hidden_size must be > 0");
    CHECK_ARGUMENT(_meta.nh > 0, "num_attention_heads must be > 0");
    CHECK_ARGUMENT(_meta.nkvh > 0, "num_key_value_heads must be > 0");
    CHECK_ARGUMENT(_meta.dh > 0, "head_dim must be > 0");
    CHECK_ARGUMENT(_meta.di > 0, "intermediate_size must be > 0");
    CHECK_ARGUMENT(_meta.voc > 0, "vocab_size must be > 0");
    CHECK_ARGUMENT(_meta.nh * _meta.dh == _meta.hs, "nh * dh must equal hidden_size");

    _attn_scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
}

void Qwen2Model::bind_weights(const LlaisysQwen2Weights &weights) {
    _weights.in_embed = unwrap(weights.in_embed);
    _weights.out_embed = unwrap(weights.out_embed);
    _weights.out_norm_w = unwrap(weights.out_norm_w);

    bind_layer_list(_weights.attn_norm_w, weights.attn_norm_w, _meta.nlayer);
    bind_layer_list(_weights.attn_q_w, weights.attn_q_w, _meta.nlayer);
    bind_layer_list(_weights.attn_q_b, weights.attn_q_b, _meta.nlayer);
    bind_layer_list(_weights.attn_k_w, weights.attn_k_w, _meta.nlayer);
    bind_layer_list(_weights.attn_k_b, weights.attn_k_b, _meta.nlayer);
    bind_layer_list(_weights.attn_v_w, weights.attn_v_w, _meta.nlayer);
    bind_layer_list(_weights.attn_v_b, weights.attn_v_b, _meta.nlayer);
    bind_layer_list(_weights.attn_o_w, weights.attn_o_w, _meta.nlayer);

    bind_layer_list(_weights.mlp_norm_w, weights.mlp_norm_w, _meta.nlayer);
    bind_layer_list(_weights.mlp_gate_w, weights.mlp_gate_w, _meta.nlayer);
    bind_layer_list(_weights.mlp_up_w, weights.mlp_up_w, _meta.nlayer);
    bind_layer_list(_weights.mlp_down_w, weights.mlp_down_w, _meta.nlayer);

    CHECK_ARGUMENT(_weights.in_embed != nullptr, "missing in_embed weight");
    CHECK_ARGUMENT(_weights.out_embed != nullptr, "missing out_embed weight");
    CHECK_ARGUMENT(_weights.out_norm_w != nullptr, "missing out_norm_w weight");

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        CHECK_ARGUMENT(_weights.attn_norm_w[i] != nullptr, "missing attn_norm_w");
        CHECK_ARGUMENT(_weights.attn_q_w[i] != nullptr, "missing attn_q_w");
        CHECK_ARGUMENT(_weights.attn_k_w[i] != nullptr, "missing attn_k_w");
        CHECK_ARGUMENT(_weights.attn_v_w[i] != nullptr, "missing attn_v_w");
        CHECK_ARGUMENT(_weights.attn_o_w[i] != nullptr, "missing attn_o_w");
        CHECK_ARGUMENT(_weights.mlp_norm_w[i] != nullptr, "missing mlp_norm_w");
        CHECK_ARGUMENT(_weights.mlp_gate_w[i] != nullptr, "missing mlp_gate_w");
        CHECK_ARGUMENT(_weights.mlp_up_w[i] != nullptr, "missing mlp_up_w");
        CHECK_ARGUMENT(_weights.mlp_down_w[i] != nullptr, "missing mlp_down_w");
    }

    _weights_bound = true;
}

Qwen2Session *Qwen2Model::create_session() {
    Qwen2Config config{
        _meta.nlayer,
        _meta.maxseq,
        _meta.nkvh,
        _meta.dh,
        _meta.hs,
        _meta.nh,
        _meta.di,
        _meta.voc,
        _meta.dtype
    };
    return new Qwen2Session(config, _device, _device_id);
}

int64_t Qwen2Model::infer(Qwen2Session *session, const int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature, int64_t seed) {
    CHECK_ARGUMENT(session, "session is null");
    CHECK_ARGUMENT(token_ids || ntoken == 0, "token_ids is null");
    CHECK_ARGUMENT(_weights_bound, "Model weights are not bound");
    if (ntoken == 0) {
        return _meta.end_token;
    }
    CHECK_ARGUMENT(ntoken <= _meta.maxseq, "ntoken exceeds maxseq");

    auto &kv_cache = session->kv_cache();

    if (kv_cache.seq_len() >= _meta.maxseq) {
        kv_cache.reset();
    }

    if (ntoken <= kv_cache.seq_len()) {
        kv_cache.reset();
    }

    for (size_t i = kv_cache.seq_len(); i < ntoken; ++i) {
        process_token(session, token_ids[i]);
    }

    llaisys::ops::rms_norm(session->_final_norm, session->_hidden, _weights.out_norm_w, _meta.epsilon);
    llaisys::ops::linear(session->_logits, session->_final_norm, _weights.out_embed, nullptr);

    if (top_k > 1 || top_p > 0.0f) {
        llaisys::ops::sample(session->_max_idx, session->_logits_flat, top_k, top_p, temperature, seed);
    } else {
        llaisys::ops::argmax(session->_max_idx, session->_max_val, session->_logits_flat);
    }

    auto *idx_ptr = reinterpret_cast<const int64_t *>(session->_max_idx->data());
    return idx_ptr[0];
}

void Qwen2Model::process_token(Qwen2Session *session, int64_t token_id) {
    auto &kv_cache = session->kv_cache();
    int64_t pos = static_cast<int64_t>(kv_cache.seq_len());

    session->_token_ids->load(&token_id);
    session->_pos_ids->load(&pos);

    llaisys::ops::embedding(session->_hidden, session->_token_ids, _weights.in_embed);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        llaisys::ops::rms_norm(session->_attn_norm, session->_hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        llaisys::ops::linear(session->_q_proj, session->_attn_norm, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        llaisys::ops::linear(session->_k_proj, session->_attn_norm, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        llaisys::ops::linear(session->_v_proj, session->_attn_norm, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        llaisys::ops::rope(session->_q_rope, session->_q_view, session->_pos_ids, _meta.theta);
        llaisys::ops::rope(session->_k_rope, session->_k_view, session->_pos_ids, _meta.theta);

        auto k_cache_slice = kv_cache.k(layer)->slice(0, pos, pos + 1);
        auto v_cache_slice = kv_cache.v(layer)->slice(0, pos, pos + 1);
        llaisys::ops::rearrange(k_cache_slice, session->_k_rope);
        llaisys::ops::rearrange(v_cache_slice, session->_v_view);

        auto k_cache = kv_cache.k(layer)->slice(0, 0, pos + 1);
        auto v_cache = kv_cache.v(layer)->slice(0, 0, pos + 1);

        llaisys::ops::self_attention(session->_attn_out, session->_q_rope, k_cache, v_cache, _attn_scale);
        llaisys::ops::linear(session->_attn_proj, session->_attn_out_flat, _weights.attn_o_w[layer], nullptr);
        llaisys::ops::add(session->_hidden, session->_hidden, session->_attn_proj);

        llaisys::ops::rms_norm(session->_mlp_norm, session->_hidden, _weights.mlp_norm_w[layer], _meta.epsilon);
        llaisys::ops::linear(session->_mlp_gate, session->_mlp_norm, _weights.mlp_gate_w[layer], nullptr);
        llaisys::ops::linear(session->_mlp_up, session->_mlp_norm, _weights.mlp_up_w[layer], nullptr);
        llaisys::ops::swiglu(session->_mlp_act, session->_mlp_gate, session->_mlp_up);
        llaisys::ops::linear(session->_mlp_down, session->_mlp_act, _weights.mlp_down_w[layer], nullptr);
        llaisys::ops::add(session->_hidden, session->_hidden, session->_mlp_down);
    }

    kv_cache.advance(1);
}

} // namespace llaisys::models::qwen2
