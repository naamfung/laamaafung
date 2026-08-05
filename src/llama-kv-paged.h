#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-memory.h"
#include "llama-model.h"

#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

// Forward declarations
struct llama_hparams;
struct llama_model;
struct llama_context;
class llama_batch_allocr;

//
// Paged KV Cache Block Size Configuration
//

// Get default block size based on backend type
// GPU (CUDA/Metal/etc.): 32 (aligned with warp size for optimal performance)
// CPU: 16 (smaller block size to reduce memory footprint and fragmentation)
uint32_t llama_kv_paged_get_default_block_size(bool is_gpu);

//
// Physical Block Structure
//

struct llama_kv_paged_block {
    uint32_t ref_count = 0;
    bool is_free = true;
    
    // Reset block state
    void reset() {
        ref_count = 0;
        is_free = true;
    }
};

//
// Page Table Entry Structure
//

struct llama_kv_page_table_entry {
    uint32_t physical_block_idx;
    llama_seq_id seq_id;
};

//
// llama_kv_paged_cache Class
//

class llama_kv_paged_cache_context;

class llama_kv_paged_cache : public llama_memory_i {
friend class llama_kv_paged_cache_context;
public:
    llama_kv_paged_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                     uint32_t   kv_size_tokens, // number of tokens, converted to blocks internally
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share);

    ~llama_kv_paged_cache() override = default;

    //
    // llama_memory_i interface
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // Paged KV Cache specific API
    //

    uint32_t get_block_size() const;
    uint32_t get_n_blocks_total() const;

    // Get physical K/V tensors for a specific layer
    ggml_tensor * get_k_tensor(int32_t il) const;
    ggml_tensor * get_v_tensor(int32_t il) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    // Block configuration
    uint32_t block_size;
    uint32_t n_blocks_total;

    // KV cache parameters
    const bool v_trans;
    const bool offload;
    const uint32_t n_pad;
    const uint32_t n_swa;
    const llama_swa_type swa_type;

    // Layer filtering and sharing
    const layer_filter_cb filter_cb;
    const layer_reuse_cb reuse_cb;
    const layer_share_cb share_cb;

    // Physical blocks
    std::vector<llama_kv_paged_block> physical_blocks;

    // Free block list (list of available physical block indices)
    std::vector<uint32_t> free_blocks;

    // Page table per sequence: seq_id -> vector of physical block indices
    // The vector size is the number of logical blocks for this sequence
    // page_tables[seq_id][logical_block_idx] -> physical_block_idx
    std::map<llama_seq_id, std::vector<uint32_t>> page_tables;

    // Sequence position tracking: seq_id -> {min_pos, max_pos}
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> seq_pos_range;

    // Physical K/V tensors per layer: il -> {k_tensor, v_tensor}
    struct layer_tensors {
        ggml_tensor * k_tensor = nullptr;
        ggml_tensor * v_tensor = nullptr;
    };
    std::unordered_map<int32_t, layer_tensors> layer_tensors_map;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> ctxs_bufs;

    // Helper functions for block management
    uint32_t allocate_block(llama_seq_id seq_id);
    void release_block(llama_seq_id seq_id, uint32_t logical_block_idx);
    void ensure_blocks_for_sequence(llama_seq_id seq_id, uint32_t n_logical_blocks);

    // Helper functions for position mapping
    uint32_t pos_to_logical_block(llama_pos pos) const;
    uint32_t pos_to_block_offset(llama_pos pos) const;
    llama_pos logical_block_to_pos(uint32_t logical_block_idx) const;

    // Defragmentation helper
    void defrag_trigger();

    // Helper function to create physical K/V tensors for a layer
    void create_layer_tensors(int32_t il, ggml_type type_k, ggml_type type_v, bool v_trans);

    // Defragmentation implementation
    void defrag_trigger_impl();
};

//
// llama_kv_paged_cache_context Class
//

class llama_kv_paged_cache_context : public llama_memory_context_i {
public:
    llama_kv_paged_cache_context(llama_memory_status status);

    llama_kv_paged_cache_context(
            llama_kv_paged_cache * paged_cache,
            llama_context * lctx,
            bool do_shift,
            const std::vector<llama_ubatch> & ubatches);

    virtual ~llama_kv_paged_cache_context();

    //
    // llama_memory_context_i interface
    //

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // TurboQuant: get rotation tensors for pre-rotate-queries optimization
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // TurboQuant InnerQ: get per-channel scale_inv tensor for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

    // Paged KV Cache specific API for index generation
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

private:
    llama_memory_status status;
    llama_kv_paged_cache * paged_cache;
    llama_context * lctx;

    // Current ubatch being processed
    size_t ubatch_idx;
    std::vector<llama_ubatch> ubatches;

    // Pending shift or update info
    bool do_shift;
};
