#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter,
                     uint32_t   n_snapshots = 0);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

    //
    // chunk state snapshots for hybrid prefix sharing (vLLM-style):
    //
    // the r_l/s_l tensors are extended with n_snap extra columns at the end.
    // each snapshot stores the R/S state at a chunk boundary (a multiple of
    // block_size tokens), keyed by the hash of the chunk's tokens. when a new
    // sequence shares a prefix, the state at the deepest available snapshot
    // boundary is restored by pointing the first cell's src0 at the snapshot
    // column, skipping the per-token recomputation of the shared prefix.
    //

    struct mem_snap {
        uint64_t hash      = 0;
        uint32_t n_tokens  = 0;
        uint64_t last_used = 0; // LRU timestamp
        bool     valid     = false;
    };

    // a scheduled snapshot write for the current ubatch (filled in apply())
    struct snap_write {
        int32_t  slot;     // snapshot slot index
        uint64_t hash;     // chunk hash
        uint32_t n_tokens; // tokens in the chunk
        uint32_t seq_off;  // seq offset within the ubatch; state column = head + seq_off
    };

    uint32_t n_snap     = 0; // snapshot slots (0 = disabled)
    uint32_t block_size = 0; // chunk size used for snapshots
    uint64_t snap_lru   = 0;
    std::vector<mem_snap> snaps;

    // seq -> snapshot slot to restore from (set by hybrid init_batch, used by find_slot)
    std::unordered_map<llama_seq_id, int32_t> snap_restore;

    // seq -> per-chunk hashes, chunk idx = pos / block_size (set by hybrid init_batch)
    std::unordered_map<llama_seq_id, std::vector<uint64_t>> chunk_hashes;

    static uint64_t compute_hash(uint64_t prev_hash, const llama_token * tokens, uint32_t n);

    int32_t find_snap(uint64_t hash) const;
    int32_t snap_alloc(uint64_t hash, uint32_t n_tokens);

    uint32_t snap_col(int32_t slot) const {
        return size*(1 + n_rs_seq) + slot;
    }

    bool is_snap_col(int32_t col) const {
        return col >= (int32_t) (size*(1 + n_rs_seq));
    }

    // longest prefix (multiple of block_size) for which snapshots are available
    int32_t find_snap_prefix(const llama_token * tokens, uint32_t n_tokens) const;

    void set_snap_restore(llama_seq_id seq_id, int32_t slot) {
        snap_restore[seq_id] = slot;
    }

    void clear_snap_restore() {
        snap_restore.clear();
    }

    void set_chunk_hashes(llama_seq_id seq_id, std::vector<uint64_t> && hashes) {
        chunk_hashes[seq_id] = std::move(hashes);
    }

    uint32_t get_block_size() const {
        return block_size;
    }

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    int32_t s_copy(int i) const;

    // scheduled snapshot writes for the current ubatch (non-empty only after apply())
    const std::vector<llama_memory_recurrent::snap_write> & get_snap_writes() const {
        return snap_writes;
    }

    // copy the scheduled recurrent state snapshots into the snapshot region.
    // called after the ubatch graph has been computed; a plain backend copy
    // (instead of a graph op) guarantees the write executes on every backend.
    void flush_snapshots() override;

    uint32_t snap_col(int32_t slot) const {
        return mem->snap_col(slot);
    }

    bool is_snap_col(int32_t col) const {
        return mem->is_snap_col(col);
    }

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    // filled by apply() for the current ubatch, consumed by the graph builder
    std::vector<llama_memory_recurrent::snap_write> snap_writes;

    // small no-alloc context used to build the source/destination view tensors
    // for flush_snapshots()
    struct ggml_context * ctx_tmp = nullptr;

    void schedule_snap_writes();

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
