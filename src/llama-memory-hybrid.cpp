#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-kv-paged.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cstdlib>
#include <map>

//
// llama_memory_hybrid
//

namespace {

// number of recurrent chunk-state snapshot slots (LLAMA_KV_RS_SNAPSHOTS env,
// default 32; 0 disables the hybrid prefix sharing snapshots)
uint32_t get_n_snapshots() {
    const char * val = std::getenv("LLAMA_KV_RS_SNAPSHOTS");
    if (val == nullptr) {
        return 32;
    }
    return (uint32_t) std::max(0, std::atoi(val));
}

// build the attention-side KV cache: vllm-style paged by default, legacy
// contiguous when paged_attn is false
std::unique_ptr<llama_kv_cache> make_attn_cache(
        const llama_model & model,
        const llama_hparams & hparams,
              ggml_type   type_k,
              ggml_type   type_v,
                   bool   v_trans,
                   bool   offload,
               uint32_t   kv_size,
               uint32_t   n_pad,
               uint32_t   n_swa,
         llama_swa_type   swa_type,
               uint32_t   n_seq_max,
                   bool   unified,
                   bool   paged_attn,
    const llama_memory_hybrid::layer_filter_cb & filter_attn) {
    const llama_memory_hybrid::layer_filter_cb filter = filter_attn == nullptr
        ? [&](int32_t il) { return !hparams.is_recr(il); }
        : filter_attn;

    if (paged_attn) {
        return std::make_unique<llama_kv_paged_cache>(
                model,
                hparams,
                type_k,
                type_v,
                v_trans,
                offload,
                kv_size,
                n_seq_max,
                n_pad,
                filter);
    }
    return std::make_unique<llama_kv_cache>(
            model,
            hparams,
            type_k,
            type_v,
            v_trans,
            offload,
            unified,
            kv_size,
            n_seq_max,
            n_pad,
            n_swa,
            swa_type,
            nullptr,
            filter,
            nullptr,
            nullptr);
}

} // namespace

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                     bool   paged_attn,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(make_attn_cache(
        model, model.hparams, type_k, type_v, v_trans, offload,
        kv_size, n_pad, n_swa, swa_type, n_seq_max, unified, paged_attn, filter_attn)),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr,
        get_n_snapshots()
    )) {}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // hybrid prefix sharing: share attention blocks and recurrent state
        // snapshots for fresh sequences whose prefix was seen before
        if (!embd_all) {
            setup_prefix_sharing(balloc);
        }

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                // align the split size so prefills stop on chunk boundaries
                // where possible: recurrent snapshots are only taken at chunk
                // boundaries, so a prefill ending off-boundary cannot be shared
                // by later requests
                uint32_t n_split = n_ubatch;
                const uint32_t block_size = mem_recr->get_block_size();
                if (block_size > 0) {
                    const uint32_t remaining = balloc.get_n_tokens() - balloc.get_n_used();
                    if (remaining > block_size && remaining % block_size != 0 && remaining <= n_ubatch) {
                        n_split = (remaining / block_size) * block_size;
                    }
                }

                ubatch = balloc.split_equal(n_split, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // budget chunk hashes for running sequences that complete a block in
        // this batch: their newly-completed chunks become shareable by future
        // requests. new sequences were fully budgeted in setup_prefix_sharing;
        // overwriting with the (identical) attention-side chain hash is a no-op.
        {
            const auto * paged = dynamic_cast<const llama_kv_paged_cache *>(mem_attn.get());
            if (paged != nullptr) {
                const uint32_t block_size = paged->block_size;
                for (const auto & ub : ubatches) {
                    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                        const llama_seq_id seq = ub.seq_id[i][0];
                        const llama_pos pos = ub.pos[i];
                        if (pos >= 0 && (uint32_t) pos % block_size == block_size - 1) {
                            const uint64_t h = paged->get_block_hash(seq, (uint32_t) pos / block_size);
                            if (h != 0) {
                                mem_recr->set_chunk_hash(seq, (uint32_t) pos / block_size, h);
                            }
                        }
                    }
                }
            }
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

// fresh-sequence prefix sharing for hybrid models:
//   - attention side: share the matched blocks via share_prefix (paged cache)
//   - recurrent side: restore the R/S state from a chunk snapshot and skip
//     the per-token recomputation of the shared prefix
// the shared prefix tokens are marked as used so that the ubatch split skips
// them, keeping the recurrent ubatches equal-seqs
void llama_memory_hybrid::setup_prefix_sharing(llama_batch_allocr & balloc) {
    auto * paged = dynamic_cast<llama_kv_paged_cache *>(mem_attn.get());
    if (paged == nullptr) {
        return; // prefix sharing requires the paged attention cache
    }

    const uint32_t block_size = paged->block_size;
    mem_recr->block_size = block_size;
    mem_recr->clear_snap_restore();

    const llama_batch & batch = balloc.get_batch();

    // group the batch token indices by sequence (batch order == token order)
    std::map<llama_seq_id, std::vector<uint32_t>> seq_idx;
    std::set<llama_seq_id> coupled_seqs;
    for (uint32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] == 0) {
            continue;
        }
        if (batch.n_seq_id[i] > 1) {
            coupled_seqs.insert(batch.seq_id[i][0]);
        }
        seq_idx[batch.seq_id[i][0]].push_back(i);
    }

    for (const auto & [seq, idxs] : seq_idx) {
        const uint32_t n = (uint32_t) idxs.size();
        if (n == 0) {
            continue;
        }

        // coupled sequences (tokens shared with another seq) cannot be split
        // cleanly - skip them
        if (coupled_seqs.count(seq)) {
            continue;
        }

        // only fresh sequences with the complete contiguous token stream
        // (pos 0..n-1) present in this batch can participate
        if (batch.pos[idxs[0]] != 0) {
            continue;
        }
        bool contig = true;
        for (uint32_t k = 1; k < n; ++k) {
            if (batch.pos[idxs[k]] != batch.pos[idxs[k-1]] + 1) {
                contig = false;
                break;
            }
        }
        if (!contig || (uint32_t) batch.pos[idxs.back()] + 1 != n) {
            continue;
        }
        if (paged->seq_length(seq) != 0) {
            continue; // not a fresh sequence on the attention side
        }

        std::vector<llama_token> tokens(n);
        for (uint32_t k = 0; k < n; ++k) {
            tokens[k] = batch.token[idxs[k]];
        }

        // precompute the chunk hashes for snapshot writes (fresh sequences
        // write snapshots while processing, later requests restore from them)
        {
            std::vector<uint64_t> hashes(n / block_size);
            uint64_t prev = 0;
            for (uint32_t i = 0; i < hashes.size(); ++i) {
                prev = llama_memory_recurrent::compute_hash(prev, tokens.data() + i*block_size, block_size);
                hashes[i] = prev;
            }
            mem_recr->set_chunk_hashes(seq, std::move(hashes));
        }

        // longest prefix shared by both sides
        const uint32_t hit_len  = paged->find_prefix(tokens.data(), n);
        const int32_t  snap_len = mem_recr->find_snap_prefix(tokens.data(), n);
        const uint32_t share_len = std::min(hit_len, (uint32_t) snap_len);
        if (share_len == 0) {
            continue;
        }

        // the restore point must be a chunk boundary that actually has a
        // snapshot and does not include output tokens (outputs must be
        // computed, not skipped); the snapshot chain may have gaps (evicted
        // slots), so walk down from share_len
        uint64_t prev = 0;
        uint32_t restore_len = 0;
        int32_t slot = -1;
        bool has_output = false;
        for (uint32_t i = 0; i < share_len / block_size; ++i) {
            // mark output tokens in this chunk
            for (uint32_t k = i*block_size; k < (i + 1)*block_size; ++k) {
                if (batch.logits[idxs[k]]) {
                    has_output = true;
                    break;
                }
            }
            prev = llama_memory_recurrent::compute_hash(prev, tokens.data() + i*block_size, block_size);
            const int32_t s = mem_recr->find_snap(prev);
            if (s >= 0 && !has_output) {
                slot = s;
                restore_len = (i + 1)*block_size;
            }
        }
        if (slot < 0) {
            continue; // no usable snapshot below the attention hit
        }

        // share the attention blocks up to the restore point so that the
        // attention and recurrent sides stay aligned
        paged->share_prefix(seq, tokens.data(), restore_len);

        LLAMA_LOG_INFO("%s: seq %d shares %u tokens (attn hit %u, snap hit %d, restore %u)\n",
                __func__, seq, restore_len, hit_len, snap_len, restore_len);

        mem_recr->set_snap_restore(seq, slot);

        // skip the shared tokens during the ubatch split
        for (uint32_t k = 0; k < restore_len; ++k) {
            balloc.mark_used(idxs[k]);
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values.
    // the recurrent state is valid only at its latest position, so the combined min must
    // not report positions that the recurrent state cannot serve
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

void llama_memory_hybrid_context::flush_snapshots() {
    ctx_recr->flush_snapshots();
}

bool llama_memory_hybrid_context::needs_snapshot_sync() const {
    return ctx_recr->needs_snapshot_sync();
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_inverse() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_inverse() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_innerq_scale_inv() const {
    return ctx_attn ? ctx_attn->get_turbo_innerq_scale_inv() : nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
