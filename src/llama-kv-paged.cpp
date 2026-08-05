#include "llama-kv-paged.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>

#define XXH_INLINE_ALL
#include "xxhash.h"

//
// llama_kv_paged_cache
//

uint32_t llama_kv_paged_cache::detect_block_size(const llama_model & model, bool offload) {
    if (offload) {
        auto * dev = model.dev_layer(0);
        if (dev && ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
            return 32;
        }
    }
    return 16;
}

llama_kv_paged_cache::llama_kv_paged_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
               const layer_filter_cb & filter) :
    llama_kv_cache(
            model,
            hparams,
            type_k,
            type_v,
            v_trans,
            offload,
            /*unified*/    true,
            /*kv_size*/    kv_size,
            n_seq_max,
            n_pad,
            /*n_swa*/      0,
            /*swa_type*/   LLAMA_SWA_TYPE_NONE,
            /*mem_other*/  nullptr,
            filter,
            nullptr,
            nullptr),
    block_size(detect_block_size(model, offload)),
    n_blocks(kv_size / block_size) {

    GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "paged cache requires swa_type == NONE");
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "paged cache requires n_pos_per_embd == 1");
    GGML_ASSERT(n_blocks > 0 && "paged cache requires kv_size >= block_size");

    blocks.resize(n_blocks);
    for (uint32_t i = 0; i < n_blocks; ++i) {
        free_block_ids.push_back(i);
    }
}

uint64_t llama_kv_paged_cache::compute_hash(uint64_t prev_hash, const llama_token * tokens, uint32_t n) {
    return XXH64(tokens, (size_t) n * sizeof(llama_token), prev_hash);
}

void llama_kv_paged_cache::touch(uint32_t block_id) {
    blocks[block_id].last_used = ++lru_counter;
}

uint32_t llama_kv_paged_cache::alloc_block(llama_seq_id exclude_seq) {
    while (free_block_ids.empty()) {
        if (!cached_block_ids.empty()) {
            // evict LRU cached block (ref_count==0, hash!=0)
            uint32_t evict_id = 0;
            uint64_t min_used = std::numeric_limits<uint64_t>::max();
            bool found = false;
            for (uint32_t bid : cached_block_ids) {
                if (blocks[bid].last_used < min_used) {
                    min_used = blocks[bid].last_used;
                    evict_id = bid;
                    found = true;
                }
            }
            GGML_ASSERT(found);
            auto & blk = blocks[evict_id];
            if (blk.hash != 0) {
                auto it = hash_to_block_id.find(blk.hash);
                if (it != hash_to_block_id.end() && it->second == evict_id) {
                    hash_to_block_id.erase(it);
                }
            }
            blk.hash = 0;
            blk.token_ids.clear();
            cached_block_ids.erase(evict_id);
            free_block_ids.push_back(evict_id);
        } else {
            // no free, no cached: preempt (swap out) a block from an LRU seq.
            // loops until a block actually frees up (shared blocks only drop
            // ref_count and stay in use).
            if (!preempt_one(exclude_seq)) {
                GGML_ASSERT(false && "paged cache: out of blocks, no seq to preempt");
            }
        }
    }

    const uint32_t block_id = free_block_ids.front();
    free_block_ids.pop_front();

    auto & blk = blocks[block_id];
    GGML_ASSERT(blk.ref_count == 0);
    if (blk.hash != 0) {
        auto it = hash_to_block_id.find(blk.hash);
        if (it != hash_to_block_id.end() && it->second == block_id) {
            hash_to_block_id.erase(it);
        }
    }
    blk.ref_count = 1;
    blk.hash      = 0;
    blk.token_ids.clear();

    used_block_ids.insert(block_id);
    touch(block_id);
    return block_id;
}

bool llama_kv_paged_cache::preempt_one(llama_seq_id exclude_seq) {
    // find the LRU active seq != exclude_seq: the seq whose last block has
    // the smallest last_used timestamp
    llama_seq_id victim = -1;
    uint64_t min_used = std::numeric_limits<uint64_t>::max();
    for (auto & [sid, bt] : block_tables) {
        if (sid == exclude_seq) continue;
        if (bt.empty()) continue;
        const uint64_t lu = blocks[bt.back()].last_used;
        if (lu < min_used) {
            min_used = lu;
            victim = sid;
        }
    }
    if (victim < 0) return false;

    auto & bt = block_tables[victim];
    const uint32_t block_id = bt.back();
    const uint32_t n_filled = (uint32_t) blocks[block_id].token_ids.size();

    bt.pop_back();
    release_block(block_id);

    // clear victim's cell metadata in the released block. if the block was
    // shared (ref_count > 0 after release) other seqs keep their bits; if it
    // went to ref_count==0 the cells become empty (pos auto-reset by seq_rm).
    const uint32_t base = block_id * block_size;
    for (uint32_t off = 0; off < n_filled; ++off) {
        v_cells[0].seq_rm(base + off, victim);
    }
    return true;
}

void llama_kv_paged_cache::release_block(uint32_t block_id) {
    auto & blk = blocks[block_id];
    GGML_ASSERT(blk.ref_count > 0);
    blk.ref_count--;
    if (blk.ref_count == 0) {
        used_block_ids.erase(block_id);
        if (blk.hash != 0) {
            // keep in cached set for prefix reuse, evictable via LRU
            cached_block_ids.insert(block_id);
        } else {
            free_block_ids.push_back(block_id);
        }
    }
}

void llama_kv_paged_cache::copy_block_data(uint32_t src_block_id, uint32_t dst_block_id, uint32_t n_tokens) {
    if (n_tokens == 0) return;

    const uint32_t src_off = src_block_id * block_size;
    const uint32_t dst_off = dst_block_id * block_size;

    struct ggml_init_params gparams = { /*.mem_size=*/ 128*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context_ptr ctx(ggml_init(gparams));
    GGML_ASSERT(ctx && "paged cache: failed to create temp context for COW copy");

    for (const auto & layer : layers) {
        if (layer.k) {
            const int64_t n_embd = layer.k->ne[0];
            const size_t row_size = ggml_row_size(layer.k->type, n_embd);
            ggml_tensor * k_src = ggml_view_2d(ctx.get(), layer.k, n_embd, n_tokens, row_size, src_off * row_size);
            ggml_tensor * k_dst = ggml_view_2d(ctx.get(), layer.k, n_embd, n_tokens, row_size, dst_off * row_size);
            ggml_backend_tensor_copy(k_src, k_dst);
        }
        if (layer.v) {
            const int64_t n_embd = layer.v->ne[0];
            const size_t row_size = ggml_row_size(layer.v->type, n_embd);
            ggml_tensor * v_src = ggml_view_2d(ctx.get(), layer.v, n_embd, n_tokens, row_size, src_off * row_size);
            ggml_tensor * v_dst = ggml_view_2d(ctx.get(), layer.v, n_embd, n_tokens, row_size, dst_off * row_size);
            ggml_backend_tensor_copy(v_src, v_dst);
        }
        if (layer.k_idx) {
            const int64_t n_embd = layer.k_idx->ne[0];
            const size_t row_size = ggml_row_size(layer.k_idx->type, n_embd);
            ggml_tensor * ki_src = ggml_view_2d(ctx.get(), layer.k_idx, n_embd, n_tokens, row_size, src_off * row_size);
            ggml_tensor * ki_dst = ggml_view_2d(ctx.get(), layer.k_idx, n_embd, n_tokens, row_size, dst_off * row_size);
            ggml_backend_tensor_copy(ki_src, ki_dst);
        }
    }
}

void llama_kv_paged_cache::may_append(llama_seq_id seq_id, llama_pos pos) {
    auto & bt = block_tables[seq_id];
    const uint32_t block_idx = pos / block_size;

    // allocate new blocks up to block_idx
    while (bt.size() <= block_idx) {
        // verify position alignment: a new block starts at pos = bt.size() * block_size
        GGML_ASSERT((uint32_t) pos == bt.size() * block_size || bt.size() < block_idx);
        bt.push_back(alloc_block(seq_id));
    }

    // lazy COW: if the block is shared (ref_count > 1), copy it before writing.
    // this is a safety net - in normal operation shared blocks are full and
    // read-only, but defensive COW prevents corruption if an edge case writes
    // to a shared block.
    const uint32_t block_id = bt[block_idx];
    if (blocks[block_id].ref_count > 1) {
        const uint32_t new_id  = alloc_block(seq_id);
        const uint32_t n_filled = (uint32_t) blocks[block_id].token_ids.size();

        copy_block_data(block_id, new_id, n_filled);
        blocks[new_id].token_ids = blocks[block_id].token_ids;

        // move this seq's cell metadata from old block to new block
        const uint32_t old_base = block_id * block_size;
        const uint32_t new_base = new_id * block_size;
        for (uint32_t off = 0; off < n_filled; ++off) {
            const llama_pos p = v_cells[0].pos_get(old_base + off);
            v_cells[0].pos_set(new_base + off, p);
            v_cells[0].seq_rm(old_base + off, seq_id);
            v_cells[0].seq_add(new_base + off, seq_id);
        }

        bt[block_idx] = new_id;
        blocks[block_id].ref_count--;
    }

    touch(bt[block_idx]);
}

void llama_kv_paged_cache::hash_blocks(llama_seq_id seq_id, uint32_t start_block, uint32_t end_block) {
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) return;

    const auto & bt = it->second;
    if (start_block >= end_block || start_block >= bt.size()) return;

    // chain seed: previous block's hash, or 0 for the first block
    uint64_t prev_hash = (start_block > 0) ? blocks[bt[start_block - 1]].hash : 0;

    for (uint32_t i = start_block; i < end_block && i < bt.size(); ++i) {
        auto & blk = blocks[bt[i]];
        if (blk.token_ids.size() < block_size) break;   // incomplete block, can't hash
        if (blk.hash != 0) { prev_hash = blk.hash; continue; }   // already hashed

        blk.hash = compute_hash(prev_hash, blk.token_ids.data(), block_size);
        hash_to_block_id[blk.hash] = bt[i];
        prev_hash = blk.hash;
    }
}

void llama_kv_paged_cache::dealloc_seq(llama_seq_id seq_id) {
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) return;

    // release in reverse to match vllm (LIFO return to free pool)
    for (auto bit = it->second.rbegin(); bit != it->second.rend(); ++bit) {
        release_block(*bit);
    }
    block_tables.erase(it);
}

uint32_t llama_kv_paged_cache::cell_index(llama_seq_id seq_id, llama_pos pos) const {
    auto it = block_tables.find(seq_id);
    GGML_ASSERT(it != block_tables.end() && "paged cache: missing block_table for seq");
    const auto & bt = it->second;
    const uint32_t block_idx = pos / block_size;
    GGML_ASSERT(block_idx < bt.size() && "paged cache: position beyond block_table");
    return bt[block_idx] * block_size + (pos % block_size);
}

uint32_t llama_kv_paged_cache::seq_length(llama_seq_id seq_id) const {
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) return 0;
    // logical length = number of blocks * block_size minus the unfilled tail
    // for simplicity, track via block_table size (last block may be partial)
    const auto & bt = it->second;
    if (bt.empty()) return 0;
    const auto & last_blk = blocks[bt.back()];
    return (uint32_t) ((bt.size() - 1) * block_size + last_blk.token_ids.size());
}

bool llama_kv_paged_cache::can_append(llama_seq_id seq_id, uint32_t n_tokens) const {
    const uint32_t cur_len   = seq_length(seq_id);
    const uint32_t end_len   = cur_len + n_tokens;
    const uint32_t cur_blocks = cur_len / block_size;
    const uint32_t end_blocks = (end_len + block_size - 1) / block_size;
    if (end_blocks <= cur_blocks) return true;
    const uint32_t n_needed = end_blocks - cur_blocks;
    const uint32_t n_avail  = (uint32_t) free_block_ids.size() + (uint32_t) cached_block_ids.size();
    return n_avail >= n_needed;
}

int llama_kv_paged_cache::ensure_capacity(llama_seq_id seq_id, uint32_t n_tokens) {
    int n_freed = 0;
    while (!can_append(seq_id, n_tokens)) {
        if (!preempt_one(seq_id)) {
            break;
        }
        n_freed++;
    }
    return n_freed;
}

uint32_t llama_kv_paged_cache::find_prefix(const llama_token * tokens, uint32_t n) const {
    if (n < block_size) return 0;

    const uint32_t n_blocks_check = n / block_size;
    uint64_t prev_hash = 0;
    uint32_t matched = 0;

    for (uint32_t i = 0; i < n_blocks_check; ++i) {
        const uint64_t h = compute_hash(prev_hash, tokens + i * block_size, block_size);
        auto it = hash_to_block_id.find(h);
        if (it == hash_to_block_id.end()) break;

        const auto & blk = blocks[it->second];
        // skip freed blocks or token mismatch (hash collision guard)
        if (blk.ref_count == 0 ||
            blk.token_ids.size() != block_size ||
            memcmp(blk.token_ids.data(), tokens + i * block_size, block_size * sizeof(llama_token)) != 0) {
            break;
        }
        prev_hash = h;
        ++matched;
    }
    return matched * block_size;
}

uint32_t llama_kv_paged_cache::share_prefix(llama_seq_id seq_id, const llama_token * tokens, uint32_t n) {
    const uint32_t hit_len = find_prefix(tokens, n);
    if (hit_len == 0) return 0;

    const uint32_t n_blocks_share = hit_len / block_size;
    auto & bt = block_tables[seq_id];
    bt.reserve(n_blocks_share);

    uint64_t prev_hash = 0;
    for (uint32_t i = 0; i < n_blocks_share; ++i) {
        const uint64_t h = compute_hash(prev_hash, tokens + i * block_size, block_size);
        auto it = hash_to_block_id.find(h);
        GGML_ASSERT(it != hash_to_block_id.end() && "share_prefix: hash vanished during share");

        const uint32_t block_id = it->second;
        auto & blk = blocks[block_id];
        GGML_ASSERT(blk.ref_count > 0 && blk.token_ids.size() == block_size);

        blk.ref_count++;
        bt.push_back(block_id);

        // add the new seq_id to each cell of this block
        const uint32_t base = block_id * block_size;
        for (uint32_t off = 0; off < block_size; ++off) {
            const uint32_t cell_idx = base + off;
            // cell must already be populated (owned by source seq)
            if (!v_cells[0].seq_has(cell_idx, seq_id)) {
                v_cells[0].seq_add(cell_idx, seq_id);
            }
        }

        touch(block_id);
        prev_hash = h;
    }

    return hit_len;
}

//
// llama_memory_i
//

llama_ubatch llama_kv_paged_cache::filter_ubatch(const llama_ubatch & ub,
        const std::unordered_map<llama_seq_id, uint32_t> & hit_lens) const {
    // embd mode: never filter
    if (ub.embd != nullptr || hit_lens.empty()) {
        return ub;
    }

    // first pass: count how many tokens survive the filter
    uint32_t n_keep = 0;
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        const llama_seq_id seq_id = ub.seq_id[i][0];
        const llama_pos    pos    = ub.pos[i * ub.n_pos];
        auto it = hit_lens.find(seq_id);
        if (it != hit_lens.end() && (uint32_t) pos < it->second) {
            continue;   // shared, skip
        }
        ++n_keep;
    }

    if (n_keep == ub.n_tokens) {
        return ub;   // nothing filtered, shallow copy
    }

    llama_ubatch res = {};
    res.b_equal_seqs = 0;
    res.n_pos        = ub.n_pos;
    res.n_tokens     = n_keep;
    res.n_seq_tokens = n_keep;
    res.n_seqs       = 1;

    if (n_keep == 0) {
        res.n_seqs_unq = 0;
        return res;   // all shared, empty ubatch
    }

    auto fdata = std::make_shared<llama_ubatch::data_t>();
    fdata->token.reserve(n_keep);
    fdata->pos.reserve(n_keep * ub.n_pos);
    fdata->n_seq_id.reserve(n_keep);
    fdata->seq_id.reserve(n_keep);
    fdata->output.reserve(n_keep);

    std::set<llama_seq_id> unq_seqs;

    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        const llama_seq_id seq_id = ub.seq_id[i][0];
        const llama_pos    pos    = ub.pos[i * ub.n_pos];
        auto it = hit_lens.find(seq_id);
        if (it != hit_lens.end() && (uint32_t) pos < it->second) {
            continue;
        }

        fdata->token.push_back(ub.token[i]);
        for (uint32_t p = 0; p < ub.n_pos; ++p) {
            fdata->pos.push_back(ub.pos[i * ub.n_pos + p]);
        }
        fdata->n_seq_id.push_back(ub.n_seq_id[i]);
        for (int32_t s = 0; s < ub.n_seq_id[i]; ++s) {
            fdata->seq_id_data.push_back(ub.seq_id[i][s]);
        }
        fdata->output.push_back(ub.output[i]);
        unq_seqs.insert(seq_id);
    }

    // set seq_id pointers AFTER seq_id_data is finalized (no more realloc)
    {
        uint32_t off = 0;
        for (uint32_t i = 0; i < n_keep; ++i) {
            fdata->seq_id.push_back(fdata->seq_id_data.data() + off);
            off += fdata->n_seq_id[i];
        }
    }

    fdata->seq_id_unq.assign(unq_seqs.begin(), unq_seqs.end());

    res.token     = fdata->token.data();
    res.pos       = fdata->pos.data();
    res.n_seq_id  = fdata->n_seq_id.data();
    res.seq_id    = fdata->seq_id.data();
    res.seq_id_unq = fdata->seq_id_unq.data();
    res.output    = fdata->output.data();
    res.n_seqs_unq = (uint32_t) fdata->seq_id_unq.size();
    res.data      = fdata;

    return res;
}

llama_memory_context_ptr llama_kv_paged_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    std::vector<llama_ubatch> ubatches;
    while (true) {
        auto ubatch = balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) break;
        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    // Phase 3: hash-chain prefix sharing (multi-seq).
    // For each new seq (empty block_table) starting at pos 0, look up the
    // hash chain and share matching full blocks. Shared tokens are then
    // filtered out of the ubatches per-token so their K/V are not recomputed.
    std::set<llama_seq_id> eligible_seqs;
    for (auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_seq_id seq_id = ub.seq_id[i][0];
            const llama_pos pos = ub.pos[i * ub.n_pos];
            if (pos == 0) {
                auto bt_it = block_tables.find(seq_id);
                if (bt_it == block_tables.end() || bt_it->second.empty()) {
                    eligible_seqs.insert(seq_id);
                }
            }
        }
    }

    // collect full token sequences for eligible seqs
    std::unordered_map<llama_seq_id, std::vector<llama_token>> seq_tokens;
    for (auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_seq_id seq_id = ub.seq_id[i][0];
            if (eligible_seqs.count(seq_id)) {
                seq_tokens[seq_id].push_back(ub.token[i]);
            }
        }
    }

    // share prefix blocks for eligible seqs
    std::unordered_map<llama_seq_id, uint32_t> hit_lens;
    for (auto & [seq_id, tokens] : seq_tokens) {
        if (tokens.empty()) continue;
        // only attempt sharing if we have at least one full block
        if (tokens.size() < block_size) continue;
        uint32_t hit = share_prefix(seq_id, tokens.data(), (uint32_t) tokens.size());
        if (hit > 0) {
            hit_lens[seq_id] = hit;
        }
    }

    // filter ubatches: drop shared prefix tokens per-token (supports multi-seq)
    std::vector<llama_ubatch> filtered_ubatches;
    for (auto & ub : ubatches) {
        llama_ubatch filtered = filter_ubatch(ub, hit_lens);
        if (filtered.n_tokens > 0) {
            filtered_ubatches.push_back(std::move(filtered));
        }
    }

    // track first new block per seq for post-batch hashing
    std::unordered_map<llama_seq_id, uint32_t> first_new_block;

    slot_info_vec_t sinfos;
    sinfos.reserve(filtered_ubatches.size());

    for (auto & ubatch : filtered_ubatches) {
        slot_info sinfo;
        sinfo.s0 = 0;
        sinfo.s1 = 0;
        sinfo.resize(1);
        sinfo.strm[0] = 0;
        sinfo.idxs[0].reserve(ubatch.n_tokens);

        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const llama_seq_id seq_id = ubatch.seq_id[i][0];
            const llama_pos    pos    = ubatch.pos[i * ubatch.n_pos];

            // record first new block for this seq (before may_append grows the table)
            if (first_new_block.find(seq_id) == first_new_block.end()) {
                auto it = block_tables.find(seq_id);
                first_new_block[seq_id] = (it != block_tables.end()) ? (uint32_t) it->second.size() : 0;
            }

            may_append(seq_id, pos);

            // append token to the current block's token_ids
            auto & bt   = block_tables[seq_id];
            auto & blk  = blocks[bt.back()];
            blk.token_ids.push_back(ubatch.token[i]);

            sinfo.idxs[0].push_back(cell_index(seq_id, pos));
        }

        sinfos.push_back(std::move(sinfo));
    }

    // hash newly completed full blocks
    for (auto & [seq_id, start] : first_new_block) {
        auto it = block_tables.find(seq_id);
        if (it == block_tables.end()) continue;
        hash_blocks(seq_id, start, (uint32_t) it->second.size());
    }

    return std::make_unique<llama_kv_cache_context>(this, std::move(sinfos), std::move(filtered_ubatches));
}

llama_memory_context_ptr llama_kv_paged_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    // paged cache: no K-shift, no stream copy
    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_kv_paged_cache::get_can_shift() const {
    return false;
}

void llama_kv_paged_cache::clear(bool data) {
    // reset block pool
    for (auto & blk : blocks) {
        blk.ref_count = 0;
        blk.hash      = 0;
        blk.last_used = 0;
        blk.token_ids.clear();
    }
    free_block_ids.clear();
    used_block_ids.clear();
    cached_block_ids.clear();
    hash_to_block_id.clear();
    lru_counter = 0;
    for (uint32_t i = 0; i < n_blocks; ++i) {
        free_block_ids.push_back(i);
    }
    block_tables.clear();

    // reset cell metadata + clear data buffers
    llama_kv_cache::clear(data);
}

bool llama_kv_paged_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0) {
        // remove from all sequences
        std::vector<llama_seq_id> seqs;
        seqs.reserve(block_tables.size());
        for (auto & [sid, _] : block_tables) seqs.push_back(sid);
        for (auto sid : seqs) {
            if (!seq_rm(sid, p0, p1)) return false;
        }
        return true;
    }

    // normalize range
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    if (p0 >= p1) return true;

    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) {
        // no block_table, delegate to base for any cell cleanup
        return llama_kv_cache::seq_rm(seq_id, p0, p1);
    }

    auto & bt = it->second;
    const uint32_t seq_len = seq_length(seq_id);

    if (p0 == 0 && (uint32_t) p1 >= seq_len) {
        // full clear
        llama_kv_cache::seq_rm(seq_id, p0, p1);
        dealloc_seq(seq_id);
        return true;
    }

    // suffix removal: p0 to end
    if ((uint32_t) p1 >= seq_len) {
        const uint32_t start_block = p0 / block_size;
        // release blocks from start_block to end
        for (uint32_t i = start_block; i < bt.size(); ++i) {
            release_block(bt[i]);
        }
        bt.resize(start_block);

        llama_kv_cache::seq_rm(seq_id, p0, p1);
        return true;
    }

    // range removal within [p0, p1): not supported in Phase 1
    // (would require splitting blocks or allocating new ones for the tail)
    LLAMA_LOG_WARN("%s: paged cache does not support partial range removal [%d, %d) for seq %d\n",
            __func__, p0, p1, seq_id);
    return false;
}

void llama_kv_paged_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && seq_id_dst >= 0 && seq_id_src != seq_id_dst);

    // normalize range
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    auto it_src = block_tables.find(seq_id_src);
    if (it_src == block_tables.end() || it_src->second.empty()) {
        // nothing to copy
        llama_kv_cache::seq_cp(seq_id_src, seq_id_dst, p0, p1);
        return;
    }

    const auto & bt_src = it_src->second;
    const uint32_t src_len = seq_length(seq_id_src);
    if (p0 >= (llama_pos) src_len) {
        llama_kv_cache::seq_cp(seq_id_src, seq_id_dst, p0, p1);
        return;
    }

    // p0 > 0 requires sparse block_tables (gap before p0), not supported
    GGML_ASSERT(p0 == 0 && "paged cache seq_cp: only p0 = 0 is supported");

    const uint32_t copy_len = std::min((uint32_t) p1, src_len);
    const uint32_t n_full   = copy_len / block_size;   // full blocks to share
    const uint32_t n_part   = copy_len % block_size;   // tokens in partial tail

    // clear dst's existing block_table
    auto it_dst = block_tables.find(seq_id_dst);
    if (it_dst != block_tables.end()) {
        dealloc_seq(seq_id_dst);
    }

    auto & bt_dst = block_tables[seq_id_dst];
    bt_dst.reserve(n_full + (n_part > 0 ? 1 : 0));

    // share full blocks (read-only, never appended to)
    for (uint32_t i = 0; i < n_full && i < bt_src.size(); ++i) {
        const uint32_t block_id = bt_src[i];
        blocks[block_id].ref_count++;
        bt_dst.push_back(block_id);
        touch(block_id);
    }

    // COW the partial tail block: dst gets a private copy so it can append
    // without corrupting src's token_ids / K-V data
    if (n_part > 0 && n_full < bt_src.size()) {
        const uint32_t src_blk = bt_src[n_full];
        const uint32_t new_blk = alloc_block(seq_id_src);

        copy_block_data(src_blk, new_blk, n_part);

        auto & new_blk_ref = blocks[new_blk];
        new_blk_ref.token_ids.assign(blocks[src_blk].token_ids.begin(),
                                     blocks[src_blk].token_ids.begin() + n_part);

        bt_dst.push_back(new_blk);

        // set cell metadata for the COW block (new cells, base seq_cp won't touch them)
        const uint32_t base = new_blk * block_size;
        for (uint32_t off = 0; off < n_part; ++off) {
            const uint32_t cell_idx = base + off;
            v_cells[0].pos_set(cell_idx, n_full * block_size + off);
            v_cells[0].seq_add(cell_idx, seq_id_dst);
        }
    }

    // delegate cell bitset sharing for the shared full blocks only
    llama_kv_cache::seq_cp(seq_id_src, seq_id_dst, 0, n_full * block_size);
}

void llama_kv_paged_cache::seq_keep(llama_seq_id seq_id) {
    // delegate cell bitset to base
    llama_kv_cache::seq_keep(seq_id);

    // deallocate block_tables of all other sequences
    std::vector<llama_seq_id> to_remove;
    for (auto & [sid, _] : block_tables) {
        if (sid != seq_id) to_remove.push_back(sid);
    }
    for (auto sid : to_remove) {
        dealloc_seq(sid);
    }
}

void llama_kv_paged_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(seq_id); GGML_UNUSED(p0); GGML_UNUSED(p1); GGML_UNUSED(shift);
    GGML_ABORT("%s: paged cache does not support seq_add (K-shift)\n", __func__);
}

void llama_kv_paged_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id); GGML_UNUSED(p0); GGML_UNUSED(p1); GGML_UNUSED(d);
    GGML_ABORT("%s: paged cache does not support seq_div (K-shift)\n", __func__);
}

//
// state write/load
//

void llama_kv_paged_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    // write block pool (unique used blocks) first so state_read can reconstruct
    // block_tables before placing cells via block_table mapping
    if (seq_id < 0) {
        uint32_t n_unique = (uint32_t) used_block_ids.size();
        io.write(&n_unique, sizeof(n_unique));
        for (uint32_t bid : used_block_ids) {
            io.write(&bid, sizeof(bid));
            const auto & blk = blocks[bid];
            io.write(&blk.hash, sizeof(blk.hash));
            uint32_t n_tokens = (uint32_t) blk.token_ids.size();
            io.write(&n_tokens, sizeof(n_tokens));
            if (n_tokens > 0) {
                io.write(blk.token_ids.data(), n_tokens * sizeof(llama_token));
            }
        }

        uint32_t n_seqs = (uint32_t) block_tables.size();
        io.write(&n_seqs, sizeof(n_seqs));
        for (auto & [sid, bt] : block_tables) {
            io.write(&sid, sizeof(sid));
            uint32_t n_blocks_seq = (uint32_t) bt.size();
            io.write(&n_blocks_seq, sizeof(n_blocks_seq));
            for (uint32_t i = 0; i < n_blocks_seq; ++i) {
                io.write(&bt[i], sizeof(bt[i]));
            }
        }
    } else {
        // per-seq: write only the blocks referenced by this seq's block_table
        auto it = block_tables.find(seq_id);
        uint32_t n_blocks_seq = (it != block_tables.end()) ? (uint32_t) it->second.size() : 0;
        io.write(&n_blocks_seq, sizeof(n_blocks_seq));
        if (it != block_tables.end()) {
            for (uint32_t i = 0; i < n_blocks_seq; ++i) {
                uint32_t bid = it->second[i];
                io.write(&bid, sizeof(bid));
                const auto & blk = blocks[bid];
                io.write(&blk.hash, sizeof(blk.hash));
                uint32_t n_tokens = (uint32_t) blk.token_ids.size();
                io.write(&n_tokens, sizeof(n_tokens));
                if (n_tokens > 0) {
                    io.write(blk.token_ids.data(), n_tokens * sizeof(llama_token));
                }
            }
        }
    }

    // delegate cell metadata + tensor data to base
    llama_kv_cache::state_write(io, seq_id, flags);
}

void llama_kv_paged_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    // Step 1: reset + read block pool and block_tables so we can place cells
    // via block_table mapping (base state_read uses find_slot which conflicts)
    if (seq_id < 0) {
        clear(true);

        uint32_t n_unique;
        io.read(&n_unique, sizeof(n_unique));
        for (uint32_t i = 0; i < n_unique; ++i) {
            uint32_t bid;
            io.read(&bid, sizeof(bid));
            auto & blk = blocks[bid];
            io.read(&blk.hash, sizeof(blk.hash));
            uint32_t n_tokens;
            io.read(&n_tokens, sizeof(n_tokens));
            blk.token_ids.resize(n_tokens);
            if (n_tokens > 0) {
                io.read(blk.token_ids.data(), n_tokens * sizeof(llama_token));
            }
            if (blk.hash != 0) {
                hash_to_block_id[blk.hash] = bid;
            }
        }

        uint32_t n_seqs;
        io.read(&n_seqs, sizeof(n_seqs));
        for (uint32_t s = 0; s < n_seqs; ++s) {
            llama_seq_id sid;
            io.read(&sid, sizeof(sid));
            uint32_t n_blocks_seq;
            io.read(&n_blocks_seq, sizeof(n_blocks_seq));
            auto & bt = block_tables[sid];
            bt.resize(n_blocks_seq);
            for (uint32_t i = 0; i < n_blocks_seq; ++i) {
                uint32_t bid;
                io.read(&bid, sizeof(bid));
                bt[i] = bid;
                blocks[bid].ref_count++;
                free_block_ids.erase(std::remove(free_block_ids.begin(), free_block_ids.end(), bid), free_block_ids.end());
                used_block_ids.insert(bid);
            }
        }
    } else {
        seq_rm(seq_id, -1, -1);

        uint32_t n_blocks_seq;
        io.read(&n_blocks_seq, sizeof(n_blocks_seq));
        auto & bt = block_tables[seq_id];
        bt.resize(n_blocks_seq);
        for (uint32_t i = 0; i < n_blocks_seq; ++i) {
            uint32_t bid;
            io.read(&bid, sizeof(bid));
            bt[i] = bid;
            auto & blk = blocks[bid];
            io.read(&blk.hash, sizeof(blk.hash));
            uint32_t n_tokens;
            io.read(&n_tokens, sizeof(n_tokens));
            blk.token_ids.resize(n_tokens);
            if (n_tokens > 0) {
                io.read(blk.token_ids.data(), n_tokens * sizeof(llama_token));
            }
            blk.ref_count++;
            free_block_ids.erase(std::remove(free_block_ids.begin(), free_block_ids.end(), bid), free_block_ids.end());
            if (blk.ref_count == 1) {
                cached_block_ids.erase(bid);
                used_block_ids.insert(bid);
            }
            if (blk.hash != 0) {
                hash_to_block_id[blk.hash] = bid;
            }
        }
    }

    // Step 2: read n_stream + per-stream cell metadata (place via block_table)
    //         + tensor data
    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));
        if (cell_count == 0) {
            continue;
        }

        slot_info sinfo;
        sinfo.s0 = 0;
        sinfo.s1 = 0;
        sinfo.resize(1);
        sinfo.strm[0] = 0;
        sinfo.idxs[0].reserve(cell_count);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            // n_pos_per_embd == 1 (asserted in constructor), no ext

            std::vector<llama_seq_id> seq_ids(n_seq_id);
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                io.read(&seq_ids[j], sizeof(seq_ids[j]));
                if (seq_id >= 0 && (seq_ids[j] < 0 || (uint32_t) seq_ids[j] >= n_seq_max)) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, seq_ids[j]);
                    throw std::runtime_error("failed to restore kv cache");
                }
            }

            // place cell via block_table mapping
            const llama_seq_id primary = (seq_id >= 0) ? seq_id : seq_ids[0];
            const uint32_t cell_idx = cell_index(primary, pos);

            v_cells[0].pos_set(cell_idx, pos);
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                const llama_seq_id sid = (seq_id >= 0) ? seq_id : seq_ids[j];
                if (!v_cells[0].seq_has(cell_idx, sid)) {
                    v_cells[0].seq_add(cell_idx, sid);
                }
            }

            sinfo.idxs[0].push_back(cell_idx);
        }

        bool ok = state_read_data(io, 0, cell_count, sinfo);
        if (!ok) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}
