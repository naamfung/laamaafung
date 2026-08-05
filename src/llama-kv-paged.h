#pragma once

#include "llama-kv-cache.h"

#include <cstdint>
#include <deque>
#include <set>
#include <unordered_map>
#include <vector>

// vllm-style paged KV cache.
// Inherits llama_kv_cache so all tensor management, cpy_k/get_k, mask
// building and graph code work unchanged. Only init_batch is overridden
// to allocate physical cells via a block-hash-chain instead of a
// contiguous ring buffer. Block sharing (ref-counted) replaces the
// server-layer prefix cache.
class llama_kv_paged_cache : public llama_kv_cache {
public:
    // block_size: 32 for GPU buft, 16 for CPU. Set in constructor based on buft.
    const uint32_t block_size;
    const uint32_t n_blocks;

    llama_kv_paged_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
               const layer_filter_cb & filter);

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    // state write/load - extended to persist block_tables and hashes
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // paged cache API (for future server-layer prefix lookup)
    //

    // hash-chain prefix lookup: returns number of matching full blocks
    // for the given token sequence, or 0 on miss.
    uint32_t find_prefix(const llama_token * tokens, uint32_t n) const;

    // per-seq logical length (number of tokens currently mapped)
    uint32_t seq_length(llama_seq_id seq_id) const;

private:
    struct block_t {
        int32_t  ref_count = 0;
        uint64_t hash      = 0;   // 0 = not yet hashed
        std::vector<llama_token> token_ids;   // partial until size == block_size
    };

    // block pool + allocation index
    std::vector<block_t>        blocks;
    std::deque<uint32_t>        free_block_ids;
    std::set<uint32_t>          used_block_ids;
    std::unordered_map<uint64_t, uint32_t> hash_to_block_id;

    // per-seq physical block table (block_table[seq][i] -> physical block id)
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> block_tables;

    // XXH64 chain: hash of block i = XXH64(tokens[i], seed=hash[i-1])
    // hash[-1] = 0 (chain head). Order-dependent.
    static uint64_t compute_hash(uint64_t prev_hash, const llama_token * tokens, uint32_t n);

    // allocate a fresh block from the free pool, returns physical block id
    uint32_t alloc_block();

    // release a block: ref_count-- and return to free pool on 0
    void release_block(uint32_t block_id);

    // ensure block_table[seq_id] has enough blocks for position `pos`.
    // allocates a new block when pos % block_size == 0.
    void may_append(llama_seq_id seq_id, llama_pos pos);

    // hash full blocks in [start_block, end_block) for seq_id using the
    // stored token_ids. Stores block.hash + hash_to_block_id.
    void hash_blocks(llama_seq_id seq_id, uint32_t start_block, uint32_t end_block);

    // share hash-chain-matched prefix blocks for seq_id:
    //   - lookup full blocks via hash_to_block_id
    //   - ref_count++ on hit, append to block_table[seq_id]
    //   - seq_add the new seq_id on each shared cell
    // returns number of shared tokens (multiple of block_size), 0 on miss
    uint32_t share_prefix(llama_seq_id seq_id, const llama_token * tokens, uint32_t n);

    // deallocate all blocks in block_table[seq_id], then clear it
    void dealloc_seq(llama_seq_id seq_id);

    // physical cell index for (seq_id, pos)
    uint32_t cell_index(llama_seq_id seq_id, llama_pos pos) const;
};
