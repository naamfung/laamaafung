#include "llama-kv-paged.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

#define XXH_INLINE_ALL
#include "xxhash.h"

//
// llama_kv_paged_cache
//

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
            /*kv_size*/    (kv_size / 32) * 32,
            n_seq_max,
            n_pad,
            /*n_swa*/      0,
            /*swa_type*/   LLAMA_SWA_TYPE_NONE,
            /*mem_other*/  nullptr,
            filter,
            nullptr,
            nullptr),
    block_size(32),
    n_blocks(kv_size / 32) {

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

uint32_t llama_kv_paged_cache::alloc_block() {
    GGML_ASSERT(!free_block_ids.empty() && "paged cache: out of free blocks");
    const uint32_t block_id = free_block_ids.front();
    free_block_ids.pop_front();

    auto & blk = blocks[block_id];
    GGML_ASSERT(blk.ref_count == 0);
    // invalidate stale hash mapping
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
    return block_id;
}

void llama_kv_paged_cache::release_block(uint32_t block_id) {
    auto & blk = blocks[block_id];
    GGML_ASSERT(blk.ref_count > 0);
    blk.ref_count--;
    if (blk.ref_count == 0) {
        used_block_ids.erase(block_id);
        free_block_ids.push_back(block_id);
        // keep hash + token_ids until block is reused (alloc_block clears them)
    }
}

void llama_kv_paged_cache::may_append(llama_seq_id seq_id, llama_pos pos) {
    auto & bt = block_tables[seq_id];
    const uint32_t block_idx = pos / block_size;

    // allocate new blocks up to block_idx
    while (bt.size() <= block_idx) {
        // verify position alignment: a new block starts at pos = bt.size() * block_size
        GGML_ASSERT((uint32_t) pos == bt.size() * block_size || bt.size() < block_idx);
        bt.push_back(alloc_block());
    }
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

        prev_hash = h;
    }

    return hit_len;
}

//
// llama_memory_i
//

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

    // Phase 2: hash-chain prefix sharing.
    // For each new seq (empty block_table) starting at pos 0, look up the
    // hash chain and share matching full blocks. Shared tokens are then
    // skipped in the ubatches so their K/V are not recomputed.

    // determine eligible seqs: new seqs (empty block_table) starting at pos 0
    // Phase 2 safety: only share for single-seq batches to ensure shared
    // tokens form a contiguous prefix that can be cleanly sliced out.
    // In multi-seq batches, shared tokens may be interleaved with unshared
    // ones, which would cause apply_ubatch to overwrite shared cells.
    std::set<llama_seq_id> all_seqs;
    std::set<llama_seq_id> eligible_seqs;
    for (auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_seq_id seq_id = ub.seq_id[i][0];
            all_seqs.insert(seq_id);
            const llama_pos pos = ub.pos[i * ub.n_pos];
            if (pos == 0) {
                auto bt_it = block_tables.find(seq_id);
                if (bt_it == block_tables.end() || bt_it->second.empty()) {
                    eligible_seqs.insert(seq_id);
                }
            }
        }
    }
    if (all_seqs.size() > 1) {
        eligible_seqs.clear();
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

    // filter ubatches: skip shared prefix tokens (leading tokens with pos < hit_len)
    std::vector<llama_ubatch> filtered_ubatches;
    for (auto & ub : ubatches) {
        if (ub.embd != nullptr) {
            // Phase 2: skip prefix sharing for embd mode
            filtered_ubatches.push_back(ub);
            continue;
        }

        uint32_t skip = 0;
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_seq_id seq_id = ub.seq_id[i][0];
            const llama_pos    pos    = ub.pos[i * ub.n_pos];
            auto it = hit_lens.find(seq_id);
            if (it == hit_lens.end() || (uint32_t) pos >= it->second) break;
            ++skip;
        }

        if (skip == 0) {
            filtered_ubatches.push_back(ub);
        } else if (skip < ub.n_tokens) {
            // slice: keep tokens [skip, n_tokens)
            llama_ubatch sliced = ub;
            sliced.n_tokens     = ub.n_tokens - skip;
            sliced.n_seq_tokens = ub.n_seq_tokens > skip ? ub.n_seq_tokens - skip : 1;
            sliced.token        = ub.token + skip;
            sliced.pos          = ub.pos + skip * ub.n_pos;
            sliced.n_seq_id     = ub.n_seq_id + skip;
            sliced.seq_id       = ub.seq_id + skip;
            sliced.output       = ub.output + skip;
            filtered_ubatches.push_back(sliced);
        }
        // else: skip entire ubatch (all tokens shared)
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
        blk.token_ids.clear();
    }
    free_block_ids.clear();
    used_block_ids.clear();
    hash_to_block_id.clear();
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

    // Phase 1: only support copies starting at p0 = 0
    GGML_ASSERT(p0 == 0 && "paged cache seq_cp: only p0 = 0 is supported");

    const uint32_t copy_len = std::min((uint32_t) p1, src_len);
    const uint32_t n_blocks_copy = (copy_len + block_size - 1) / block_size;

    // clear dst's existing block_table
    auto it_dst = block_tables.find(seq_id_dst);
    if (it_dst != block_tables.end()) {
        dealloc_seq(seq_id_dst);
    }

    // share blocks: copy block_ids, ref_count++
    auto & bt_dst = block_tables[seq_id_dst];
    bt_dst.reserve(n_blocks_copy);
    for (uint32_t i = 0; i < n_blocks_copy && i < bt_src.size(); ++i) {
        const uint32_t block_id = bt_src[i];
        blocks[block_id].ref_count++;
        bt_dst.push_back(block_id);
    }

    // delegate cell bitset sharing to base
    llama_kv_cache::seq_cp(seq_id_src, seq_id_dst, p0, p1);
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
    // delegate cell metadata to base
    llama_kv_cache::state_write(io, seq_id, flags);

    // write block_tables for the requested seq (or all if seq_id == -1)
    if (seq_id < 0) {
        uint32_t n_seqs = (uint32_t) block_tables.size();
        io.write(&n_seqs, sizeof(n_seqs));
        for (auto & [sid, bt] : block_tables) {
            io.write(&sid, sizeof(sid));
            uint32_t n_blocks_seq = (uint32_t) bt.size();
            io.write(&n_blocks_seq, sizeof(n_blocks_seq));
            for (uint32_t i = 0; i < n_blocks_seq; ++i) {
                io.write(&bt[i], sizeof(bt[i]));
                // write block hash + token_ids for reconstruction
                const auto & blk = blocks[bt[i]];
                io.write(&blk.hash, sizeof(blk.hash));
                uint32_t n_tokens = (uint32_t) blk.token_ids.size();
                io.write(&n_tokens, sizeof(n_tokens));
                if (n_tokens > 0) {
                    io.write(blk.token_ids.data(), n_tokens * sizeof(llama_token));
                }
            }
        }
    } else {
        auto it = block_tables.find(seq_id);
        uint32_t n_blocks_seq = (it != block_tables.end()) ? (uint32_t) it->second.size() : 0;
        io.write(&n_blocks_seq, sizeof(n_blocks_seq));
        if (it != block_tables.end()) {
            for (uint32_t i = 0; i < n_blocks_seq; ++i) {
                io.write(&it->second[i], sizeof(it->second[i]));
                const auto & blk = blocks[it->second[i]];
                io.write(&blk.hash, sizeof(blk.hash));
                uint32_t n_tokens = (uint32_t) blk.token_ids.size();
                io.write(&n_tokens, sizeof(n_tokens));
                if (n_tokens > 0) {
                    io.write(blk.token_ids.data(), n_tokens * sizeof(llama_token));
                }
            }
        }
    }
}

void llama_kv_paged_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // delegate cell metadata to base
    llama_kv_cache::state_read(io, seq_id, flags);

    // read block_tables
    if (seq_id < 0) {
        // whole-cache restore: reset block pool first
        for (auto & blk : blocks) {
            blk.ref_count = 0;
            blk.hash      = 0;
            blk.token_ids.clear();
        }
        free_block_ids.clear();
        used_block_ids.clear();
        hash_to_block_id.clear();
        for (uint32_t i = 0; i < n_blocks; ++i) {
            free_block_ids.push_back(i);
        }
        block_tables.clear();

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
                uint32_t block_id;
                io.read(&block_id, sizeof(block_id));
                bt[i] = block_id;

                auto & blk = blocks[block_id];
                GGML_ASSERT(blk.ref_count == 0 && "paged cache state_read: block already in use");
                io.read(&blk.hash, sizeof(blk.hash));
                uint32_t n_tokens;
                io.read(&n_tokens, sizeof(n_tokens));
                blk.token_ids.resize(n_tokens);
                if (n_tokens > 0) {
                    io.read(blk.token_ids.data(), n_tokens * sizeof(llama_token));
                }
                blk.ref_count = 1;
                free_block_ids.erase(std::remove(free_block_ids.begin(), free_block_ids.end(), block_id), free_block_ids.end());
                used_block_ids.insert(block_id);
                if (blk.hash != 0) {
                    hash_to_block_id[blk.hash] = block_id;
                }
            }
        }
    } else {
        // per-seq restore
        auto it = block_tables.find(seq_id);
        if (it != block_tables.end()) {
            dealloc_seq(seq_id);
        }

        uint32_t n_blocks_seq;
        io.read(&n_blocks_seq, sizeof(n_blocks_seq));
        auto & bt = block_tables[seq_id];
        bt.resize(n_blocks_seq);
        for (uint32_t i = 0; i < n_blocks_seq; ++i) {
            uint32_t block_id;
            io.read(&block_id, sizeof(block_id));
            bt[i] = block_id;

            auto & blk = blocks[block_id];
            io.read(&blk.hash, sizeof(blk.hash));
            uint32_t n_tokens;
            io.read(&n_tokens, sizeof(n_tokens));
            blk.token_ids.resize(n_tokens);
            if (n_tokens > 0) {
                io.read(blk.token_ids.data(), n_tokens * sizeof(llama_token));
            }
            blk.ref_count++;
            free_block_ids.erase(std::remove(free_block_ids.begin(), free_block_ids.end(), block_id), free_block_ids.end());
            used_block_ids.insert(block_id);
            if (blk.hash != 0) {
                hash_to_block_id[blk.hash] = block_id;
            }
        }
    }
}
