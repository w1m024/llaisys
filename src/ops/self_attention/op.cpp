#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // Q: [batch, qlen, nhead, head_dim] or [qlen, nhead, head_dim]
    // K: [batch, kvlen, nkvhead, head_dim] or [kvlen, nkvhead, head_dim]
    // V: [batch, kvlen, nkvhead, value_dim] or [kvlen, nkvhead, value_dim]
    // Out: [batch, qlen, nhead, value_dim] or [qlen, nhead, value_dim]

    size_t batch = 1;
    size_t qlen, nhead, head_dim;
    size_t kvlen, nkvhead, k_head_dim;
    size_t value_dim;

    const auto &qs = q->shape();
    const auto &ks = k->shape();
    const auto &vs = v->shape();
    const auto &os = attn_val->shape();

    if (q->ndim() == 4) {
        ASSERT(q->ndim() == 4 && k->ndim() == 4 && v->ndim() == 4 && attn_val->ndim() == 4,
               "self_attention: all tensors must be 4D (batch)");
        batch = qs[0];
        qlen = qs[1]; nhead = qs[2]; head_dim = qs[3];
        kvlen = ks[1]; nkvhead = ks[2]; k_head_dim = ks[3];
        value_dim = vs[3];

        ASSERT(ks[0] == batch && vs[0] == batch && os[0] == batch, "batch size mismatch");
        ASSERT(os[1] == qlen && os[2] == nhead && os[3] == value_dim, "attn_val shape mismatch");
        ASSERT(vs[1] == kvlen && vs[2] == nkvhead, "v shape must align with k");
    } else {
        ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
               "self_attention: all tensors must be 3D (no batch)");
        qlen = qs[0]; nhead = qs[1]; head_dim = qs[2];
        kvlen = ks[0]; nkvhead = ks[1]; k_head_dim = ks[2];
        value_dim = vs[2];

        ASSERT(os[0] == qlen && os[1] == nhead && os[2] == value_dim, "attn_val shape mismatch");
        ASSERT(vs[0] == kvlen && vs[1] == nkvhead, "v shape must align with k");
    }

    ASSERT(head_dim == k_head_dim, "q/k head_dim must match");
    ASSERT(nhead % nkvhead == 0, "nhead must be a multiple of nkvhead for head repeat");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            batch,
            qlen,
            kvlen,
            nhead,
            nkvhead,
            head_dim,
            value_dim,
            scale);
        return;
    }

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
