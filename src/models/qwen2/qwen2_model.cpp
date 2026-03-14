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

#include <algorithm>
#include <utility>

namespace llaisys::models::qwen2 {
namespace {
constexpr size_t kKvBlockSize = 16;

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

std::pair<tensor_t, tensor_t> gather_attention_cache(
    Qwen2Session *session,
    size_t layer,
    size_t total_tokens,
    tensor_t tail_k,
    tensor_t tail_v,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id,
    size_t nkvhead,
    size_t head_dim) {
    auto gathered_k = Tensor::create({total_tokens, nkvhead, head_dim}, dtype, device, device_id);
    auto gathered_v = Tensor::create({total_tokens, nkvhead, head_dim}, dtype, device, device_id);

    size_t copied = 0;
    for (const auto &block : session->blocks()) {
        if (block->used == 0 || copied >= total_tokens) {
            continue;
        }

        const size_t length = std::min(block->used, total_tokens - copied);
        auto k_src = block->k_blocks[layer]->slice(0, 0, length);
        auto v_src = block->v_blocks[layer]->slice(0, 0, length);
        auto k_dst = gathered_k->slice(0, copied, copied + length);
        auto v_dst = gathered_v->slice(0, copied, copied + length);

        llaisys::ops::rearrange(k_dst, k_src);
        llaisys::ops::rearrange(v_dst, v_src);
        copied += length;
    }

    if (tail_k && tail_v) {
        ASSERT(copied + 1 == total_tokens, "KV cache gather size mismatch");
        auto k_tail_dst = gathered_k->slice(0, copied, copied + 1);
        auto v_tail_dst = gathered_v->slice(0, copied, copied + 1);
        llaisys::ops::rearrange(k_tail_dst, tail_k);
        llaisys::ops::rearrange(v_tail_dst, tail_v);
    } else {
        ASSERT(copied == total_tokens, "KV cache gather size mismatch");
    }

    return {gathered_k, gathered_v};
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

    const size_t num_blocks = std::max<size_t>((_meta.maxseq + kKvBlockSize - 1) / kKvBlockSize, 1);
    _block_manager = std::make_shared<BlockManager>(
        kKvBlockSize,
        num_blocks,
        _meta.nlayer,
        _meta.nkvh,
        _meta.dh,
        _meta.dtype,
        _device,
        _device_id);
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
    return new Qwen2Session(config, _device, _device_id, _block_manager);
}

int64_t Qwen2Model::infer(Qwen2Session *session, const int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature, int64_t seed) {
    CHECK_ARGUMENT(session, "session is null");
    CHECK_ARGUMENT(token_ids || ntoken == 0, "token_ids is null");
    CHECK_ARGUMENT(_weights_bound, "Model weights are not bound");
    if (ntoken == 0) {
        return _meta.end_token;
    }
    CHECK_ARGUMENT(ntoken <= _meta.maxseq, "ntoken exceeds maxseq");

    if (session->seq_len() >= _meta.maxseq) {
        session->reset();
    }

    if (ntoken <= session->seq_len()) {
        session->reset();
    }

    for (size_t i = session->seq_len(); i < ntoken; ++i) {
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

std::vector<int64_t> Qwen2Model::infer_batch(
    const std::vector<Qwen2Session*> &sessions,
    const std::vector<std::vector<int64_t>> &batch_token_ids,
    const std::vector<int> &top_ks,
    const std::vector<float> &top_ps,
    const std::vector<float> &temperatures,
    const std::vector<int64_t> &seeds
) {
    size_t batch_size = sessions.size();
    CHECK_ARGUMENT(batch_size > 0, "batch size must be > 0");
    CHECK_ARGUMENT(batch_token_ids.size() == batch_size, "batch_token_ids size mismatch");
    CHECK_ARGUMENT(top_ks.size() == batch_size, "top_ks size mismatch");
    CHECK_ARGUMENT(top_ps.size() == batch_size, "top_ps size mismatch");
    CHECK_ARGUMENT(temperatures.size() == batch_size, "temperatures size mismatch");
    CHECK_ARGUMENT(seeds.size() == batch_size, "seeds size mismatch");
    CHECK_ARGUMENT(_weights_bound, "Model weights are not bound");

    // Determine max steps needed
    size_t max_steps = 0;
    for (const auto& tokens : batch_token_ids) {
        if (tokens.size() > max_steps) {
            max_steps = tokens.size();
        }
    }

    for (size_t step = 0; step < max_steps; ++step) {
        bool any_active = false;
        std::vector<Qwen2Session*> active_sessions;
        std::vector<int64_t> active_tokens;

        for (size_t b = 0; b < batch_size; ++b) {
            if (sessions[b]->seq_len() >= _meta.maxseq) {
                sessions[b]->reset();
            }
            if (step == 0 && batch_token_ids[b].size() <= sessions[b]->seq_len()) {
                sessions[b]->reset();
            }

            if (step < batch_token_ids[b].size()) {
                active_sessions.push_back(sessions[b]);
                active_tokens.push_back(batch_token_ids[b][step]);
                any_active = true;
            }
        }

        if (!any_active) {
            break;
        }

        process_batch(active_sessions, active_tokens);
    }
    
    // Sampling (only for the last token of each sequence)
    std::vector<int64_t> results(batch_size);
    
    for (size_t b = 0; b < batch_size; ++b) {
        // Only sample if this session processed something
        if (batch_token_ids[b].empty()) continue;

        auto *session = sessions[b];
        
        llaisys::ops::rms_norm(session->_final_norm, session->_hidden, _weights.out_norm_w, _meta.epsilon);
        llaisys::ops::linear(session->_logits, session->_final_norm, _weights.out_embed, nullptr);

        if (top_ks[b] > 1 || top_ps[b] > 0.0f) {
            llaisys::ops::sample(session->_max_idx, session->_logits_flat, top_ks[b], top_ps[b], temperatures[b], seeds[b]);
        } else {
            llaisys::ops::argmax(session->_max_idx, session->_max_val, session->_logits_flat);
        }
        
        auto *idx_ptr = reinterpret_cast<const int64_t *>(session->_max_idx->data());
        results[b] = idx_ptr[0];
    }

    return results;
}

void Qwen2Model::process_batch(const std::vector<Qwen2Session*> &sessions, const std::vector<int64_t> &token_ids) {
    size_t batch = sessions.size();
    if (batch == 0) {
        return;
    }

    auto create_batch_tensor = [&](const std::vector<size_t>& shape) {
        std::vector<size_t> batch_shape = shape;
        batch_shape.insert(batch_shape.begin(), batch);
        return Tensor::create(batch_shape, _meta.dtype, _device, _device_id);
    };

    auto batch_hidden = create_batch_tensor({1, _meta.hs});
    auto batch_token_ids = Tensor::create({batch, 1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto batch_pos_ids = Tensor::create({batch, 1}, LLAISYS_DTYPE_I64, _device, _device_id);

    for (size_t b = 0; b < batch; ++b) {
        const int64_t tid = token_ids[b];
        const int64_t pos = static_cast<int64_t>(sessions[b]->seq_len());
        if (_device == LLAISYS_DEVICE_CPU) {
            auto *t_ptr = reinterpret_cast<int64_t *>(batch_token_ids->data());
            auto *p_ptr = reinterpret_cast<int64_t *>(batch_pos_ids->data());
            t_ptr[b] = tid;
            p_ptr[b] = pos;
        }
    }

    llaisys::ops::embedding(batch_hidden, batch_token_ids, _weights.in_embed);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        auto batch_attn_norm = create_batch_tensor({1, _meta.hs});
        llaisys::ops::rms_norm(batch_attn_norm, batch_hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        auto batch_q_proj = create_batch_tensor({1, _meta.nh * _meta.dh});
        auto batch_k_proj = create_batch_tensor({1, _meta.nkvh * _meta.dh});
        auto batch_v_proj = create_batch_tensor({1, _meta.nkvh * _meta.dh});
        
        llaisys::ops::linear(batch_q_proj, batch_attn_norm, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        llaisys::ops::linear(batch_k_proj, batch_attn_norm, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        llaisys::ops::linear(batch_v_proj, batch_attn_norm, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);
        
        auto batch_q_view = batch_q_proj->view({batch, 1, _meta.nh, _meta.dh});
        auto batch_k_view = batch_k_proj->view({batch, 1, _meta.nkvh, _meta.dh});
        auto batch_v_view = batch_v_proj->view({batch, 1, _meta.nkvh, _meta.dh});
        
        auto batch_q_rope = create_batch_tensor({1, _meta.nh, _meta.dh});
        auto batch_k_rope = create_batch_tensor({1, _meta.nkvh, _meta.dh});

        llaisys::ops::rope(batch_q_rope, batch_q_view, batch_pos_ids, _meta.theta);
        llaisys::ops::rope(batch_k_rope, batch_k_view, batch_pos_ids, _meta.theta);

        auto batch_attn_out = create_batch_tensor({1, _meta.nh, _meta.dh});

        for (size_t b = 0; b < batch; ++b) {
            sessions[b]->ensure_capacity_for_next_token();

            const size_t pos = sessions[b]->seq_len();
            auto q_rope_slice = batch_q_rope->slice(0, b, b+1)->view({1, _meta.nh, _meta.dh});
            auto k_rope_slice = batch_k_rope->slice(0, b, b+1)->view({1, _meta.nkvh, _meta.dh});
            auto v_view_slice = batch_v_view->slice(0, b, b+1)->view({1, _meta.nkvh, _meta.dh});

            sessions[b]->write_kv(layer, k_rope_slice, v_view_slice);

            auto [temp_k_cache, temp_v_cache] = gather_attention_cache(
                sessions[b],
                layer,
                pos + 1,
                k_rope_slice,
                v_view_slice,
                _meta.dtype,
                _device,
                _device_id,
                _meta.nkvh,
                _meta.dh);
            auto attn_out_slice = batch_attn_out->slice(0, b, b+1)->view({1, _meta.nh, _meta.dh});
            llaisys::ops::self_attention(attn_out_slice, q_rope_slice, temp_k_cache, temp_v_cache, _attn_scale);
        }

        auto batch_attn_out_flat = batch_attn_out->view({batch, 1, _meta.hs});
        auto batch_attn_proj = create_batch_tensor({1, _meta.hs});
        
        llaisys::ops::linear(batch_attn_proj, batch_attn_out_flat, _weights.attn_o_w[layer], nullptr);
        llaisys::ops::add(batch_hidden, batch_hidden, batch_attn_proj);

        // FFN
        auto batch_mlp_norm = create_batch_tensor({1, _meta.hs});
        llaisys::ops::rms_norm(batch_mlp_norm, batch_hidden, _weights.mlp_norm_w[layer], _meta.epsilon);
        
        auto batch_mlp_gate = create_batch_tensor({1, _meta.di});
        auto batch_mlp_up = create_batch_tensor({1, _meta.di});
        auto batch_mlp_act = create_batch_tensor({1, _meta.di});
        
        llaisys::ops::linear(batch_mlp_gate, batch_mlp_norm, _weights.mlp_gate_w[layer], nullptr);
        llaisys::ops::linear(batch_mlp_up, batch_mlp_norm, _weights.mlp_up_w[layer], nullptr);
        llaisys::ops::swiglu(batch_mlp_act, batch_mlp_gate, batch_mlp_up);
        
        auto batch_mlp_down = create_batch_tensor({1, _meta.hs});
        llaisys::ops::linear(batch_mlp_down, batch_mlp_act, _weights.mlp_down_w[layer], nullptr);
        llaisys::ops::add(batch_hidden, batch_hidden, batch_mlp_down);
    }

    for (size_t b = 0; b < batch; ++b) {
        auto hidden_slice = batch_hidden->slice(0, b, b+1)->view({1, _meta.hs});
        llaisys::ops::rearrange(sessions[b]->_hidden, hidden_slice);
        sessions[b]->advance(1);
    }
}

void Qwen2Model::process_token(Qwen2Session *session, int64_t token_id) {
    const size_t pos = session->seq_len();
    const int64_t pos_i64 = static_cast<int64_t>(pos);

    session->_token_ids->load(&token_id);
    session->_pos_ids->load(&pos_i64);

    llaisys::ops::embedding(session->_hidden, session->_token_ids, _weights.in_embed);
    session->ensure_capacity_for_next_token();

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        llaisys::ops::rms_norm(session->_attn_norm, session->_hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        llaisys::ops::linear(session->_q_proj, session->_attn_norm, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        llaisys::ops::linear(session->_k_proj, session->_attn_norm, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        llaisys::ops::linear(session->_v_proj, session->_attn_norm, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        llaisys::ops::rope(session->_q_rope, session->_q_view, session->_pos_ids, _meta.theta);
        llaisys::ops::rope(session->_k_rope, session->_k_view, session->_pos_ids, _meta.theta);

        session->write_kv(layer, session->_k_rope, session->_v_view);

        auto [k_cache, v_cache] = gather_attention_cache(
            session,
            layer,
            pos + 1,
            session->_k_rope,
            session->_v_view,
            _meta.dtype,
            _device,
            _device_id,
            _meta.nkvh,
            _meta.dh);

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

    session->advance(1);
}

} // namespace llaisys::models::qwen2
