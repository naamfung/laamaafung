#pragma once

#include "llama-kv-cache.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <deque>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// vllm-style paged KV cache.
// Inherits llama_kv_cache so all tensor management, cpy_k/get_k, mask
// building and graph code work unchanged. Only init_batch is overridden
// to allocate physical cells via a block-hash-chain instead of a
// contiguous ring buffer. Block sharing (ref-counted) replaces the
// server-layer prefix cache.
class llama_kv_paged_cache : public llama_kv_cache {
public:
    // block_size: 32 for GPU buft, 16 for CPU. Auto-detected in constructor.
    const uint32_t block_size;
    const uint32_t n_blocks;

    // lossless K/V compression for swap_out (LLAMA_KV_SWAP_COMPRESS=1;
    // default 0 keeps the vLLM behavior of raw copies)
    const bool swap_compress;

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

    // process one externally-sliced ubatch: allocate blocks, append tokens,
    // build cell-index slot info. shared by init_batch (after prefix
    // filtering) and prepare (hybrid path).
    slot_info process_ubatch(
            const llama_ubatch & ubatch,
            std::unordered_map<llama_seq_id, uint32_t> & first_modified_block);

    // externally-sliced ubatch processing WITHOUT prefix sharing: used by
    // llama_memory_hybrid, whose recurrent layers impose their own ubatch
    // slicing (init_batch's split/share/filter flow cannot be used there)
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches) override;

    // cell-metadata update on K/V write. Same as the base cache but WITHOUT
    // the "purge overwritten positions" pass: in the block layout the
    // pos->cell mapping is fixed per sequence (block_table), so overwriting a
    // cell never breaks positional continuity, and partial-range seq_rm is
    // not supported by the paged cache.
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    bool can_append(llama_seq_id seq_id, uint32_t n_tokens) const override;
    int  ensure_capacity(llama_seq_id seq_id, uint32_t n_tokens) override;

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

    // per-seq logical length (number of tokens currently mapped)
    uint32_t seq_length(llama_seq_id seq_id) const;

    // hash-chain prefix lookup and sharing (override of llama_memory_i)
    uint32_t find_prefix (const llama_token * tokens, uint32_t n) const override;
    uint32_t share_prefix(llama_seq_id seq_id, const llama_token * tokens, uint32_t n) override;

    // chain hash of the block at block_idx of seq_id (0 when unavailable);
    // used to budget recurrent chunk snapshots for running sequences that
    // complete a block mid-generation
    uint64_t get_block_hash(llama_seq_id seq_id, uint32_t block_idx) const;

    // swap preemption: save/restore a seq's K/V data to/from CPU memory
    bool is_swapped(llama_seq_id seq_id) const override;
    bool swap_out  (llama_seq_id seq_id) override;
    bool swap_in   (llama_seq_id seq_id) override;

    // preemption priority: higher values are retained longer under capacity
    // pressure; the lowest-priority seq is preempted first (ties by LRU)
    void seq_set_priority(llama_seq_id seq_id, int32_t priority) override;
    uint32_t n_swapped_tokens() const;

    // metrics for monitoring
    struct metrics {
        uint32_t n_blocks_total   = 0;
        uint32_t n_blocks_free    = 0;
        uint32_t n_blocks_used    = 0;
        uint32_t n_blocks_cached  = 0;
        uint32_t n_swapped_tokens = 0;
        uint64_t preempt_count    = 0;
        uint64_t swap_out_count   = 0;
        uint64_t swap_in_count    = 0;
        uint32_t block_size       = 0;
    };
    metrics get_metrics() const;

private:
    struct block_t {
        int32_t  ref_count = 0;
        uint64_t hash      = 0;   // 0 = not yet hashed
        uint64_t last_used = 0;   // LRU timestamp, higher = more recent
        std::vector<llama_token> token_ids;   // partial until size == block_size
    };

    // block pool + allocation index
    std::vector<block_t>        blocks;
    std::deque<uint32_t>        free_block_ids;     // ref_count==0, hash==0 (truly free)
    std::set<uint32_t>          used_block_ids;     // ref_count>0 (in use)
    std::set<uint32_t>          cached_block_ids;   // ref_count==0, hash!=0 (evictable)
    // hash -> one or more physical blocks with that content. multiple blocks
    // can share a hash when identical partial-tail blocks exist on different
    // sequences (the map must not collapse them into a single entry).
    std::unordered_multimap<uint64_t, uint32_t> hash_to_block_id;

    // monotonic counter for LRU ordering
    uint64_t lru_counter = 0;

    // metrics counters
    uint64_t preempt_count  = 0;
    uint64_t swap_out_count = 0;
    uint64_t swap_in_count  = 0;

    // per-seq physical block table (block_table[seq][i] -> physical block id)
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> block_tables;

    // preemption priority per sequence (default 0 = LRU-only behavior)
    std::unordered_map<llama_seq_id, int32_t> seq_priorities;

    // sequences with tokens still pending in the current batch. preemption
    // must never swap out one of these: its remaining tokens would hit an
    // empty block_table mid-batch. only sequences outside the batch (finished
    // requests holding cache) are eligible victims.
    std::unordered_set<llama_seq_id> in_flight_seqs;

    // swap storage: per-seq CPU buffers for swapped-out K/V data
    struct swap_entry_t {
        std::vector<llama_token> tokens;
        std::vector<std::vector<uint8_t>> k_data;  // per-layer (compressed or raw)
        std::vector<std::vector<uint8_t>> v_data;
        std::vector<size_t> k_raw_size;  // uncompressed size per layer (when compressed)
        std::vector<size_t> v_raw_size;
        bool compressed = false;
        uint32_t n_tokens = 0;
    };
    std::unordered_map<llama_seq_id, swap_entry_t> swapped_seqs;

    // async swap backends: one per device, lazily created. using a dedicated
    // backend gives a separate CUDA stream so swap copies don't block the
    // compute stream. all layer copies are pipelined then synchronized once,
    // reducing N syncs to 1.
    std::unordered_map<ggml_backend_dev_t, ggml_backend_ptr> swap_backends;
    std::vector<ggml_backend_t> swap_backends_used;

    ggml_backend_t get_swap_backend(const ggml_tensor * tensor);
    void swap_synchronize();

    // auto-detect block_size: 32 for GPU buft, 16 for CPU
    static uint32_t detect_block_size(const llama_model & model, bool offload);

    // XXH64 chain: hash of block i = XXH64(tokens[i], seed=hash[i-1])
    // hash[-1] = 0 (chain head). Order-dependent.
    static uint64_t compute_hash(uint64_t prev_hash, const llama_token * tokens, uint32_t n);

    // update last_used on a block (call on alloc/share/append)
    void touch(uint32_t block_id);

    // allocate a fresh block. when the free pool and cached set are both
    // empty, preempt (swap out) the tail block of the LRU active seq that is
    // not exclude_seq until a block becomes available. asserts if no seq can
    // be preempted (e.g. exclude_seq is the only active seq).
    uint32_t alloc_block(llama_seq_id exclude_seq = -1);

    // preempt one block from the LRU active seq != exclude_seq: release the
    // tail block of that seq, clear the victim's cell metadata for the
    // released cells, and truncate its block_table. returns false if no seq
    // can be preempted. note: a shared block (ref_count > 1 after release)
    // stays in use and does not immediately yield a free block - the caller
    // (alloc_block) loops until something frees up.
    bool preempt_one(llama_seq_id exclude_seq);

    // release a block: ref_count-- and on 0 move to cached set (keep hash for reuse)
    void release_block(uint32_t block_id);

    // copy K/V (and k_idx) data for n_tokens cells from src to dst block.
    // used for COW when forking a partial block in seq_cp.
    void copy_block_data(uint32_t src_block_id, uint32_t dst_block_id, uint32_t n_tokens);

    // ensure block_table[seq_id] has enough blocks for position `pos`.
    // allocates a new block when pos % block_size == 0.
    void may_append(llama_seq_id seq_id, llama_pos pos);

    // copy-on-write: duplicate the block at block_table[seq_id][block_idx]
    // into a fresh private block for seq_id (used when writing to a block
    // shared with other sequences). migrates seq_id's cell metadata to the
    // copy and decrements the old block's ref_count. returns the new block id.
    uint32_t cow_block(llama_seq_id seq_id, uint32_t block_idx);

    // zero out the K/V data of a block

    // hash full blocks in [start_block, end_block) for seq_id using the
    // stored token_ids. Stores block.hash + hash_to_block_id.
    void hash_blocks(llama_seq_id seq_id, uint32_t start_block, uint32_t end_block);

    // deallocate all blocks in block_table[seq_id], then clear it
    void dealloc_seq(llama_seq_id seq_id);

    // filter a ubatch, dropping tokens whose (seq_id, pos) is covered by
    // hit_lens (i.e. pos < hit_lens[seq_id]). returns a new ubatch with own
    // storage. if no tokens are dropped, returns a shallow copy of the input.
    llama_ubatch filter_ubatch(const llama_ubatch & ub,
            const std::unordered_map<llama_seq_id, uint32_t> & hit_lens) const;

    // physical cell index for (seq_id, pos)
    uint32_t cell_index(llama_seq_id seq_id, llama_pos pos) const;
};
