#include "llama-kv-paged.h"

#include "miniz.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#define XXH_INLINE_ALL
#include "xxhash.h"

//
// llama_kv_paged_cache
//

uint32_t llama_kv_paged_cache::detect_block_size(const llama_model & model, bool offload) {
    // user override via env var (must be > 0)
    if (const char * env = std::getenv("LLAMA_KV_BLOCK_SIZE")) {
        const int v = std::atoi(env);
        if (v > 0) {
            return (uint32_t) v;
        }
    }

    // adaptive block size: aim for roughly constant K/V bytes per block.
    // wide-KV models (large n_embd_k_gqa) use finer blocks to reduce
    // partial-tail waste on the last block; narrow-KV models use coarser
    // blocks to keep the block count (and management overhead) low.
    // base remains 16 (CPU) / 32 (GPU) for the common range.
    uint32_t bs = 16;
    if (offload) {
        auto * dev = model.dev_layer(0);
        if (dev && ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
            bs = 32;
        }
    }
    const uint32_t kv_w = model.hparams.n_embd_k_gqa_max();
    if (kv_w >= 4096) {
        bs = std::max(16u, bs / 2);
    } else if (kv_w <= 128) {
        bs = std::min(64u, bs * 2);
    }
    return bs;
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
            /*n_swa*/      hparams.n_swa,
            /*swa_type*/   hparams.swa_type,
            /*mem_other*/  nullptr,
            filter,
            nullptr,
            nullptr),
    block_size(detect_block_size(model, offload)),
    n_blocks(kv_size / block_size),
    swap_compress([]() {
        const char * env = std::getenv("LLAMA_KV_SWAP_COMPRESS");
        return env && env[0] == '1';
    }()) {

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
                throw std::runtime_error("paged cache: out of blocks, no seq to preempt");
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
    // find the victim seq != exclude_seq: the lowest-priority seq first,
    // ties broken by LRU (smallest last_used of its last block)
    llama_seq_id victim = -1;
    uint64_t min_used = std::numeric_limits<uint64_t>::max();
    int32_t min_prio = std::numeric_limits<int32_t>::max();
    for (auto & [sid, bt] : block_tables) {
        if (sid == exclude_seq) continue;
        if (bt.empty()) continue;
        const auto pit = seq_priorities.find(sid);
        const int32_t prio = pit != seq_priorities.end() ? pit->second : 0;
        const uint64_t lu = blocks[bt.back()].last_used;
        if (prio < min_prio || (prio == min_prio && lu < min_used)) {
            min_prio = prio;
            min_used = lu;
            victim = sid;
        }
    }
    if (victim < 0) return false;

    preempt_count++;

    // swap out the entire victim if it has > 1 block and swap buffer has room.
    // this preserves K/V data and frees all blocks at once, avoiding recompute.
    auto & bt = block_tables[victim];
    if (bt.size() > 1 && n_swapped_tokens() + seq_length(victim) < n_blocks * block_size) {
        return swap_out(victim);
    }

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

    // NOTE: ggml_backend_tensor_copy cannot be used with view tensors here -
    // a view's buffer field is NULL (it is resolved lazily via view_src by the
    // backend, but tensor_copy checks src->buffer directly). Copy via byte
    // offsets on the underlying tensors instead.
    auto copy_range = [](ggml_tensor * tensor, size_t src_bytes, size_t dst_bytes, size_t nbytes) {
        void * tmp = malloc(nbytes);
        GGML_ASSERT(tmp && "paged cache: OOM in COW copy");
        ggml_backend_tensor_get(tensor, tmp, src_bytes, nbytes);
        ggml_backend_tensor_set(tensor, tmp, dst_bytes, nbytes);
        free(tmp);
    };

    for (const auto & layer : layers) {
        if (layer.k) {
            const size_t row_size = ggml_row_size(layer.k->type, layer.k->ne[0]);
            copy_range(layer.k, (size_t) src_off * row_size, (size_t) dst_off * row_size, row_size * n_tokens);
        }
        if (layer.v) {
            const size_t row_size = ggml_row_size(layer.v->type, layer.v->ne[0]);
            copy_range(layer.v, (size_t) src_off * row_size, (size_t) dst_off * row_size, row_size * n_tokens);
        }
        if (layer.k_idx) {
            const size_t row_size = ggml_row_size(layer.k_idx->type, layer.k_idx->ne[0]);
            copy_range(layer.k_idx, (size_t) src_off * row_size, (size_t) dst_off * row_size, row_size * n_tokens);
        }
    }
}

uint32_t llama_kv_paged_cache::cow_block(llama_seq_id seq_id, uint32_t block_idx) {
    auto & bt = block_tables[seq_id];
    const uint32_t old_id = bt[block_idx];
    GGML_ASSERT(blocks[old_id].ref_count > 1);

    const uint32_t new_id  = alloc_block(seq_id);
    const uint32_t n_filled = (uint32_t) blocks[old_id].token_ids.size();

    copy_block_data(old_id, new_id, n_filled);
    blocks[new_id].token_ids = blocks[old_id].token_ids;

    // move this seq's cell metadata from old block to new block
    const uint32_t old_base = old_id * block_size;
    const uint32_t new_base = new_id * block_size;
    for (uint32_t off = 0; off < n_filled; ++off) {
        const llama_pos p = v_cells[0].pos_get(old_base + off);
        v_cells[0].pos_set(new_base + off, p);
        v_cells[0].seq_rm(old_base + off, seq_id);
        v_cells[0].seq_add(new_base + off, seq_id);
    }

    bt[block_idx] = new_id;
    blocks[old_id].ref_count--;
    touch(new_id);
    touch(old_id);
    return new_id;
}

void llama_kv_paged_cache::may_append(llama_seq_id seq_id, llama_pos pos) {
    auto & bt = block_tables[seq_id];
    const uint32_t block_idx = pos / block_size;

    // allocate new blocks up to block_idx
    while (bt.size() <= block_idx) {
        // each new block must start exactly at pos == bt.size() * block_size.
        // non-contiguous positions would leave holes in the block chain and
        // break the block hash chain used for prefix matching
        GGML_ASSERT((uint32_t) pos == bt.size() * block_size && "paged cache: non-contiguous positions are not supported");
        bt.push_back(alloc_block(seq_id));
    }

    // lazy COW: if the block is shared (ref_count > 1), copy it before writing.
    // this is a safety net - in normal operation shared blocks are full and
    // read-only, but defensive COW prevents corruption if an edge case writes
    // to a shared block.
    const uint32_t block_id = bt[block_idx];
    if (blocks[block_id].ref_count > 1) {
        cow_block(seq_id, block_idx);
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

    if (start_block > 0 && prev_hash == 0) {
        // the previous block was never hashed (only possible with a broken
        // chain). rebuild from block 0 so stored hashes stay consistent with
        // find_prefix's chain walk.
        for (uint32_t i = 0; i < start_block; ++i) {
            const auto & pb = blocks[bt[i]];
            if (pb.token_ids.size() < block_size) {
                return;   // chain can't be rebuilt, leave blocks unhashed
            }
            prev_hash = pb.hash != 0 ? pb.hash : compute_hash(prev_hash, pb.token_ids.data(), block_size);
        }
    }

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
    seq_priorities.erase(seq_id);
}

void llama_kv_paged_cache::seq_set_priority(llama_seq_id seq_id, int32_t priority) {
    // keep the entry even for default priority so a later seq_id reuse does
    // not inherit a stale priority (dealloc_seq erases it on release)
    seq_priorities[seq_id] = priority;
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

bool llama_kv_paged_cache::is_swapped(llama_seq_id seq_id) const {
    return swapped_seqs.find(seq_id) != swapped_seqs.end();
}

uint32_t llama_kv_paged_cache::n_swapped_tokens() const {
    uint32_t total = 0;
    for (auto & [sid, entry] : swapped_seqs) {
        total += entry.n_tokens;
    }
    return total;
}

ggml_backend_t llama_kv_paged_cache::get_swap_backend(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buf) return nullptr;

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft) return nullptr;

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) return nullptr;

    auto it = swap_backends.find(dev);
    if (it != swap_backends.end()) {
        return it->second.get();
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return nullptr;

    auto & ptr = swap_backends[dev];
    ptr.reset(backend);
    return backend;
}

void llama_kv_paged_cache::swap_synchronize() {
    for (auto & backend : swap_backends_used) {
        if (backend) {
            ggml_backend_synchronize(backend);
        }
    }
    swap_backends_used.clear();
}

llama_kv_paged_cache::metrics llama_kv_paged_cache::get_metrics() const {
    metrics m;
    m.n_blocks_total   = n_blocks;
    m.n_blocks_free    = (uint32_t) free_block_ids.size();
    m.n_blocks_used    = (uint32_t) used_block_ids.size();
    m.n_blocks_cached  = (uint32_t) cached_block_ids.size();
    m.n_swapped_tokens = n_swapped_tokens();
    m.preempt_count    = preempt_count;
    m.swap_out_count   = swap_out_count;
    m.swap_in_count    = swap_in_count;
    m.block_size       = block_size;
    return m;
}

bool llama_kv_paged_cache::swap_out(llama_seq_id seq_id) {
    const int64_t t0 = ggml_time_us();
    uint32_t n_tokens = seq_length(seq_id);
    if (n_tokens == 0) return false;

    swap_entry_t entry;
    entry.n_tokens = n_tokens;
    entry.k_data.resize(layers.size());
    entry.v_data.resize(layers.size());

    // allocate CPU buffers
    for (size_t il = 0; il < layers.size(); il++) {
        auto & layer = layers[il];
        if (layer.k) {
            size_t row_size = ggml_row_size(layer.k->type, layer.k->ne[0]);
            entry.k_data[il].resize(n_tokens * row_size);
        }
        if (layer.v) {
            size_t row_size = ggml_row_size(layer.v->type, layer.v->ne[0]);
            entry.v_data[il].resize(n_tokens * row_size);
        }
    }

    // copy K/V data and collect tokens, block by block. use async copies to
    // pipeline all layers on the swap stream, then sync once at the end.
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) return false;
    auto & bt = it->second;

    uint32_t token_off = 0;
    for (uint32_t block_id : bt) {
        auto & blk = blocks[block_id];
        uint32_t n_filled = (uint32_t) blk.token_ids.size();
        uint32_t cell_start = block_id * block_size;

        for (auto t : blk.token_ids) {
            entry.tokens.push_back(t);
        }

        for (size_t il = 0; il < layers.size(); il++) {
            auto & layer = layers[il];
            if (layer.k && !entry.k_data[il].empty()) {
                size_t row_size = ggml_row_size(layer.k->type, layer.k->ne[0]);
                ggml_backend_t backend = get_swap_backend(layer.k);
                if (backend) {
                    swap_backends_used.push_back(backend);
                    ggml_backend_tensor_get_async(
                        backend, layer.k,
                        entry.k_data[il].data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                } else {
                    ggml_backend_tensor_get(
                        layer.k,
                        entry.k_data[il].data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                }
            }
            if (layer.v && !entry.v_data[il].empty()) {
                size_t row_size = ggml_row_size(layer.v->type, layer.v->ne[0]);
                ggml_backend_t backend = get_swap_backend(layer.v);
                if (backend) {
                    swap_backends_used.push_back(backend);
                    ggml_backend_tensor_get_async(
                        backend, layer.v,
                        entry.v_data[il].data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                } else {
                    ggml_backend_tensor_get(
                        layer.v,
                        entry.v_data[il].data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                }
            }
        }

        token_off += n_filled;
    }

    // wait for all async copies to complete before freeing blocks
    swap_synchronize();

    // optional lossless compression of the K/V data before storing to the
    // swap buffer (LLAMA_KV_SWAP_COMPRESS=1). reduces CPU swap memory at the
    // cost of compress/decompress CPU time; swap_in restores bit-exact data.
    if (swap_compress) {
        for (size_t il = 0; il < layers.size(); il++) {
            auto compress_layer = [](std::vector<uint8_t> & data, std::vector<size_t> & raw_sizes, size_t il_idx) {
                if (data.empty()) return;
                const size_t raw_size = data.size();
                std::vector<uint8_t> comp(mz_compressBound((mz_ulong) raw_size));
                mz_ulong comp_size = (mz_ulong) comp.size();
                if (mz_compress2(comp.data(), &comp_size, data.data(), (mz_ulong) raw_size, MZ_DEFAULT_LEVEL) == MZ_OK) {
                    comp.resize(comp_size);
                    data = std::move(comp);
                    if (raw_sizes.size() <= il_idx) raw_sizes.resize(il_idx + 1);
                    raw_sizes[il_idx] = raw_size;
                }
                // on failure keep the raw data
            };
            compress_layer(entry.k_data[il], entry.k_raw_size, il);
            compress_layer(entry.v_data[il], entry.v_raw_size, il);
        }
        entry.compressed = true;
    }

    // clear cell metadata for the seq
    for (uint32_t block_id : bt) {
        auto & blk = blocks[block_id];
        uint32_t n_filled = (uint32_t) blk.token_ids.size();
        uint32_t base = block_id * block_size;
        for (uint32_t off = 0; off < n_filled; ++off) {
            v_cells[0].seq_rm(base + off, seq_id);
        }
    }

    dealloc_seq(seq_id);
    swapped_seqs[seq_id] = std::move(entry);

    swap_out_count++;

    const int64_t t1 = ggml_time_us();
    LLAMA_LOG_INFO("paged cache: swap_out seq=%d, n_tokens=%u, %.2f ms\n",
                   seq_id, n_tokens, (t1 - t0) / 1000.0);
    return true;
}

bool llama_kv_paged_cache::swap_in(llama_seq_id seq_id) {
    auto it = swapped_seqs.find(seq_id);
    if (it == swapped_seqs.end()) return false;

    const int64_t t0 = ggml_time_us();
    auto & entry = it->second;
    uint32_t n_tokens = entry.n_tokens;

    // allocate blocks at block boundaries
    uint32_t n_blocks_needed = (n_tokens + block_size - 1) / block_size;
    for (uint32_t bi = 0; bi < n_blocks_needed; bi++) {
        may_append(seq_id, bi * block_size);
    }

    // set cell metadata for each position
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        uint32_t cell = cell_index(seq_id, pos);
        v_cells[0].pos_set(cell, pos);
        v_cells[0].seq_add(cell, seq_id);
    }

    // restore token_ids in blocks
    auto & bt = block_tables[seq_id];
    uint32_t token_off = 0;
    for (uint32_t bi = 0; bi < bt.size(); bi++) {
        auto & blk = blocks[bt[bi]];
        uint32_t n_filled = std::min(block_size, n_tokens - token_off);
        blk.token_ids.assign(
            entry.tokens.begin() + token_off,
            entry.tokens.begin() + token_off + n_filled);
        token_off += n_filled;
    }

    // copy K/V data from CPU to GPU, block by block. use async copies to
    // pipeline all layers on the swap stream, then sync once at the end.
    // if the swap data was stored compressed, decompress it once up front
    // (bit-exact, so the restored K/V matches the original).
    std::vector<std::vector<uint8_t>> k_raw, v_raw;
    if (entry.compressed) {
        k_raw.resize(layers.size());
        v_raw.resize(layers.size());
        for (size_t il = 0; il < layers.size(); il++) {
            // a layer is only decompressed when it was actually compressed
            // (k_raw_size[il] > 0); compression may have been skipped/failed
            if (!entry.k_data[il].empty() && entry.k_raw_size.size() > il && entry.k_raw_size[il] > 0) {
                k_raw[il].resize(entry.k_raw_size[il]);
                mz_ulong out_size = (mz_ulong) k_raw[il].size();
                if (mz_uncompress(k_raw[il].data(), &out_size,
                        entry.k_data[il].data(), (mz_ulong) entry.k_data[il].size()) != MZ_OK) {
                    LLAMA_LOG_ERROR("%s: K decompression failed for layer %zu\n", __func__, il);
                    return false;
                }
            }
            if (!entry.v_data[il].empty() && entry.v_raw_size.size() > il && entry.v_raw_size[il] > 0) {
                v_raw[il].resize(entry.v_raw_size[il]);
                mz_ulong out_size = (mz_ulong) v_raw[il].size();
                if (mz_uncompress(v_raw[il].data(), &out_size,
                        entry.v_data[il].data(), (mz_ulong) entry.v_data[il].size()) != MZ_OK) {
                    LLAMA_LOG_ERROR("%s: V decompression failed for layer %zu\n", __func__, il);
                    return false;
                }
            }
        }
    }

    token_off = 0;
    for (uint32_t block_id : bt) {
        uint32_t n_filled = (uint32_t) blocks[block_id].token_ids.size();
        uint32_t cell_start = block_id * block_size;

        for (size_t il = 0; il < layers.size(); il++) {
            auto & layer = layers[il];
            if (layer.k && !entry.k_data[il].empty()) {
                const std::vector<uint8_t> & k_src = (k_raw.size() > il && !k_raw[il].empty()) ? k_raw[il] : entry.k_data[il];
                size_t row_size = ggml_row_size(layer.k->type, layer.k->ne[0]);
                ggml_backend_t backend = get_swap_backend(layer.k);
                if (backend) {
                    swap_backends_used.push_back(backend);
                    ggml_backend_tensor_set_async(
                        backend, layer.k,
                        k_src.data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                } else {
                    ggml_backend_tensor_set(
                        layer.k,
                        k_src.data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                }
            }
            if (layer.v && !entry.v_data[il].empty()) {
                const std::vector<uint8_t> & v_src = (v_raw.size() > il && !v_raw[il].empty()) ? v_raw[il] : entry.v_data[il];
                size_t row_size = ggml_row_size(layer.v->type, layer.v->ne[0]);
                ggml_backend_t backend = get_swap_backend(layer.v);
                if (backend) {
                    swap_backends_used.push_back(backend);
                    ggml_backend_tensor_set_async(
                        backend, layer.v,
                        v_src.data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                } else {
                    ggml_backend_tensor_set(
                        layer.v,
                        v_src.data() + token_off * row_size,
                        cell_start * row_size, n_filled * row_size);
                }
            }
        }

        token_off += n_filled;
    }

    // wait for all async copies to complete before returning
    swap_synchronize();

    // hash full blocks for prefix sharing
    uint32_t n_full_blocks = n_tokens / block_size;
    if (n_full_blocks > 0) {
        hash_blocks(seq_id, 0, n_full_blocks);
    }

    swapped_seqs.erase(it);

    swap_in_count++;

    const int64_t t1 = ggml_time_us();
    LLAMA_LOG_INFO("paged cache: swap_in seq=%d, n_tokens=%u, %.2f ms\n",
                   seq_id, n_tokens, (t1 - t0) / 1000.0);
    return true;
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
        // skip freed blocks or token mismatch (hash collision guard). cached
        // blocks (ref_count == 0, data still resident) can be reused.
        if (blk.token_ids.size() != block_size ||
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
        GGML_ASSERT(blk.token_ids.size() == block_size);

        bt.push_back(block_id);

        const uint32_t base = block_id * block_size;
        if (blk.ref_count == 0) {
            // reuse a cached (evicted) block: its cells were fully released
            // (pos/seq cleared), so repopulate the metadata before adding this
            // sequence. the block hash/token_ids are still valid.
            blk.ref_count = 1;
            cached_block_ids.erase(block_id);
            used_block_ids.insert(block_id);
            for (uint32_t off = 0; off < block_size; ++off) {
                v_cells[0].pos_set(base + off, (llama_pos) i * block_size + off);
                v_cells[0].seq_add(base + off, seq_id);
            }
        } else {
            blk.ref_count++;
            // add the new seq_id to each cell of this block
            for (uint32_t off = 0; off < block_size; ++off) {
                const uint32_t cell_idx = base + off;
                // cell must already be populated (owned by source seq)
                if (!v_cells[0].seq_has(cell_idx, seq_id)) {
                    v_cells[0].seq_add(cell_idx, seq_id);
                }
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

    // track first modified block per seq for post-batch hashing.
    // unlike "first new block", this also covers the case where a previously
    // partial block becomes full during decode - essential for incremental
    // prefix matching (running seqs' newly completed blocks must be hashed
    // so future seqs can share them).
    std::unordered_map<llama_seq_id, uint32_t> first_modified_block;

    slot_info_vec_t sinfos;
    sinfos.reserve(filtered_ubatches.size());

    for (auto & ubatch : filtered_ubatches) {
        sinfos.push_back(process_ubatch(ubatch, first_modified_block));
    }

    // hash newly completed full blocks. hash_blocks skips already-hashed
    // blocks, so starting from first_modified_block is safe even if some
    // blocks in the range were hashed in a previous batch.
    for (auto & [seq_id, start] : first_modified_block) {
        auto it = block_tables.find(seq_id);
        if (it == block_tables.end()) continue;
        hash_blocks(seq_id, start, (uint32_t) it->second.size());
    }

    return std::make_unique<llama_kv_cache_context>(this, std::move(sinfos), std::move(filtered_ubatches));
}

llama_kv_cache::slot_info llama_kv_paged_cache::process_ubatch(
        const llama_ubatch & ubatch,
        std::unordered_map<llama_seq_id, uint32_t> & first_modified_block) {
    slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.resize(1);
    sinfo.strm[0] = 0;
    sinfo.idxs[0].reserve(ubatch.n_tokens);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        const llama_pos    pos    = ubatch.pos[i * ubatch.n_pos];

        // record the block index of the first token processed for this
        // seq. this block may already exist (partial) and become full
        // during this batch, or it may be newly allocated.
        const uint32_t block_idx = (uint32_t) pos / block_size;
        auto fmb_it = first_modified_block.find(seq_id);
        if (fmb_it == first_modified_block.end() || block_idx < fmb_it->second) {
            first_modified_block[seq_id] = block_idx;
        }

        may_append(seq_id, pos);

        // append token to the block that contains this position (contiguous
        // positions guarantee it equals the last block)
        auto & bt   = block_tables[seq_id];
        auto & blk  = blocks[bt[block_idx]];
        blk.token_ids.push_back(ubatch.token[i]);

        sinfo.idxs[0].push_back(cell_index(seq_id, pos));
    }

    return sinfo;
}

llama_kv_cache::slot_info_vec_t llama_kv_paged_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    // Non-sharing path for externally-sliced ubatches: used by
    // llama_memory_hybrid, where the recurrent layers impose their own ubatch
    // slicing, so init_batch's split/share/filter flow cannot be applied.
    // Blocks are allocated immediately from the pool; slot cell indices are
    // the physical cell offsets of each token.
    std::unordered_map<llama_seq_id, uint32_t> first_modified_block;

    slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (auto & ubatch : ubatches) {
        sinfos.push_back(process_ubatch(ubatch, first_modified_block));
    }

    // hash newly completed full blocks
    for (auto & [seq_id, start] : first_modified_block) {
        auto it = block_tables.find(seq_id);
        if (it == block_tables.end()) continue;
        hash_blocks(seq_id, start, (uint32_t) it->second.size());
    }

    return sinfos;
}

void llama_kv_paged_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // Same cell-metadata update as the base cache, but without the
    // "purge overwritten positions" pass - in the block layout the
    // pos->cell mapping is fixed per sequence (block_table), so overwriting
    // a cell never breaks positional continuity (see llama_kv_cache::apply_ubatch).
    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);
                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t ss = 0; ss < ubatch.n_seq_id[i]; ss++) {
                cells.seq_add(idx, ubatch.seq_id[i][ss]);
            }
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
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
    swapped_seqs.clear();
    swap_backends_used.clear();

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

        if (start_block >= bt.size()) {
            // p0 is beyond the current sequence length: nothing left to keep
            for (uint32_t i = 0; i < bt.size(); ++i) {
                release_block(bt[i]);
            }
            bt.clear();
            llama_kv_cache::seq_rm(seq_id, p0, p1);
            return true;
        }

        const uint32_t keep_in_block = p0 % block_size;

        if (keep_in_block != 0) {
            // p0 cuts inside a block: truncate that block (COW first if it is
            // shared with other sequences) and release everything after it
            uint32_t blk_id = bt[start_block];
            if (blocks[blk_id].ref_count > 1) {
                blk_id = cow_block(seq_id, start_block);
            }
            auto & blk = blocks[blk_id];
            const uint32_t n_filled = (uint32_t) blk.token_ids.size();
            if (n_filled > keep_in_block) {
                // drop the tail tokens and their cell metadata
                const uint32_t base = blk_id * block_size;
                for (uint32_t off = keep_in_block; off < n_filled; ++off) {
                    const uint32_t cell_idx = base + off;
                    if (v_cells[0].seq_has(cell_idx, seq_id)) {
                        v_cells[0].seq_rm(cell_idx, seq_id);
                    }
                }
                blk.token_ids.resize(keep_in_block);
                // the block hash is now stale: drop it so it gets recomputed
                if (blk.hash != 0) {
                    auto it = hash_to_block_id.find(blk.hash);
                    if (it != hash_to_block_id.end() && it->second == blk_id) {
                        hash_to_block_id.erase(it);
                    }
                    blk.hash = 0;
                }
            }
            // release everything after the truncated block
            for (uint32_t i = start_block + 1; i < bt.size(); ++i) {
                release_block(bt[i]);
            }
            bt.resize(start_block + 1);
        } else {
            // p0 is block-aligned: release blocks from start_block onward
            for (uint32_t i = start_block; i < bt.size(); ++i) {
                release_block(bt[i]);
            }
            bt.resize(start_block);
        }

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

            // ext (n_pos_per_embd > 1) is written between n_seq_id and seq_ids
            llama_kv_cell_ext ext;
            if (hparams.n_pos_per_embd() > 1) {
                io.read(&ext, sizeof(ext));
            }

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
            if (hparams.n_pos_per_embd() > 1) {
                v_cells[0].ext_set(cell_idx, ext);
            }
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
