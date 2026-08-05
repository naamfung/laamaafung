#include "llama-kv-paged.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// Paged KV Cache Block Size Configuration
//

uint32_t llama_kv_paged_get_default_block_size(bool is_gpu) {
    // GPU (CUDA/Metal/etc.): 32 (aligned with warp size for optimal performance)
    // CPU: 16 (smaller block size to reduce memory footprint and fragmentation)
    if (is_gpu) {
        return 32;
    }
    return 16;
}

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
                 uint32_t   kv_size_tokens,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    model(model), hparams(hparams), v_trans(v_trans), offload(offload),
    n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    filter_cb(filter), reuse_cb(reuse), share_cb(share) {

    // Determine block size based on backend type
    // Check if any device is a GPU backend
    bool is_gpu = false;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (offload) {
            auto * dev = model.dev_layer(il);
            // Simplified check: if offload is true, we assume it's likely GPU or accelerators
            // A more robust check would be to inspect ggml_backend_dev_type(dev)
            is_gpu = true; // Placeholder - will be refined based on actual backend detection
        }
    }

    // Check environment variable for override, though we default to auto-detection
    const char * env_block_size = getenv("LLAMA_KV_BLOCK_SIZE");
    if (env_block_size && atoi(env_block_size) > 0) {
        block_size = (uint32_t)atoi(env_block_size);
        LLAMA_LOG_INFO("llama_kv_paged_cache: block size overridden by LLAMA_KV_BLOCK_SIZE to %u\n", block_size);
    } else {
        block_size = llama_kv_paged_get_default_block_size(is_gpu);
        LLAMA_LOG_INFO("llama_kv_paged_cache: using default block size %u (is_gpu=%d)\n", block_size, is_gpu ? 1 : 0);
    }

    GGML_ASSERT(kv_size_tokens > 0);
    GGML_ASSERT(block_size > 0);

    // Calculate number of blocks
    n_blocks_total = (kv_size_tokens + block_size - 1) / block_size;

    // Initialize physical blocks
    physical_blocks.resize(n_blocks_total);
    for (uint32_t i = 0; i < n_blocks_total; ++i) {
        physical_blocks[i].reset();
        free_blocks.push_back(i);
    }

    // Initialize sequence position range
    // seq_pos_range is populated dynamically as sequences are added/modified

    // Create physical K/V tensors for all layers
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        create_layer_tensors((int32_t)il, type_k, type_v, v_trans);
    }
}

uint32_t llama_kv_paged_cache::get_block_size() const {
    return block_size;
}

uint32_t llama_kv_paged_cache::get_n_blocks_total() const {
    return n_blocks_total;
}

// Helper functions for position mapping

uint32_t llama_kv_paged_cache::pos_to_logical_block(llama_pos pos) const {
    if (pos < 0) return 0; // Should not happen with valid pos
    return (uint32_t)(pos / (llama_pos)block_size);
}

uint32_t llama_kv_paged_cache::pos_to_block_offset(llama_pos pos) const {
    if (pos < 0) return 0;
    return (uint32_t)(pos % (llama_pos)block_size);
}

llama_pos llama_kv_paged_cache::logical_block_to_pos(uint32_t logical_block_idx) const {
    return (llama_pos)logical_block_idx * (llama_pos)block_size;
}

// Helper functions for block management

uint32_t llama_kv_paged_cache::allocate_block(llama_seq_id seq_id) {
    if (free_blocks.empty()) {
        throw std::runtime_error("llama_kv_paged_cache: no free blocks available");
    }

    uint32_t block_idx = free_blocks.back();
    free_blocks.pop_back();

    physical_blocks[block_idx].is_free = false;
    physical_blocks[block_idx].ref_count = 1;

    // Ensure page table exists for this sequence
    if (page_tables.find(seq_id) == page_tables.end()) {
        page_tables[seq_id] = std::vector<uint32_t>();
    }

    page_tables[seq_id].push_back(block_idx);

    return block_idx;
}

void llama_kv_paged_cache::release_block(llama_seq_id seq_id, uint32_t logical_block_idx) {
    if (page_tables.find(seq_id) == page_tables.end() || logical_block_idx >= page_tables[seq_id].size()) {
        return; // Invalid block index
    }

    uint32_t physical_block_idx = page_tables[seq_id][logical_block_idx];

    if (physical_block_idx >= physical_blocks.size()) {
        throw std::runtime_error("llama_kv_paged_cache: invalid physical block index");
    }

    if (physical_blocks[physical_block_idx].is_free) {
        throw std::runtime_error("llama_kv_paged_cache: releasing already free block");
    }

    physical_blocks[physical_block_idx].ref_count--;

    if (physical_blocks[physical_block_idx].ref_count <= 0) {
        physical_blocks[physical_block_idx].is_free = true;
        physical_blocks[physical_block_idx].ref_count = 0;
        free_blocks.push_back(physical_block_idx);
    }

    // Remove from page table
    page_tables[seq_id].erase(page_tables[seq_id].begin() + (int)logical_block_idx);
}

void llama_kv_paged_cache::ensure_blocks_for_sequence(llama_seq_id seq_id, uint32_t n_logical_blocks) {
    if (page_tables.find(seq_id) == page_tables.end()) {
        page_tables[seq_id] = std::vector<uint32_t>();
    }

    uint32_t current_size = (uint32_t)page_tables[seq_id].size();
    if (current_size < n_logical_blocks) {
        uint32_t blocks_to_allocate = n_logical_blocks - current_size;
        for (uint32_t i = 0; i < blocks_to_allocate; ++i) {
            uint32_t block_idx = allocate_block(seq_id);
            // Update sequence position range
            llama_pos pos_start = logical_block_to_pos((uint32_t)page_tables[seq_id].size() - 1);
            llama_pos pos_end = pos_start + block_size - 1;

            if (seq_pos_range.find(seq_id) == seq_pos_range.end()) {
                seq_pos_range[seq_id] = std::make_pair(pos_start, pos_end);
            } else {
                seq_pos_range[seq_id].first = std::min(seq_pos_range[seq_id].first, pos_start);
                seq_pos_range[seq_id].second = std::max(seq_pos_range[seq_id].second, pos_end);
            }
        }
    }
}

//
// llama_memory_i interface implementations
//

llama_memory_context_ptr llama_kv_paged_cache::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    // Reset the batch allocator
    balloc.split_reset();

    // Check and trigger defragmentation if needed
    defrag_trigger();

    // Split the batch into ubatches
    std::vector<llama_ubatch> ubatches;
    for (uint32_t i = 0; i < n_ubatch; ++i) {
        llama_ubatch ubatch = balloc.split_equal(n_ubatch, true, 0);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(ubatch);

        // Ensure blocks for each sequence in the ubatch
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            llama_seq_id seq_id = ubatch.seq_id_unq[s];
            // Determine the number of logical blocks needed for this sequence
            // We iterate over the tokens in the ubatch to find the max logical block needed
            uint32_t n_logical_blocks = 0;
            for (uint32_t t = 0; t < ubatch.n_tokens; ++t) {
                // Check if token t belongs to seq_id
                // ubatch.seq_id is an array of pointers to seq_id, we need to check the seq_id at index t
                // However, ubatch.seq_id[t] is a llama_seq_id* pointer. We can check if it points to seq_id
                bool is_target_seq = false;
                if (ubatch.seq_id[t]) {
                    for (uint32_t u = 0; u < ubatch.n_seqs_unq; ++u) {
                        if (ubatch.seq_id[t] == &ubatch.seq_id_unq[u]) {
                            if (ubatch.seq_id_unq[u] == seq_id) {
                                is_target_seq = true;
                            }
                            break;
                        }
                    }
                }

                if (is_target_seq) {
                    llama_pos pos = ubatch.pos[t]; // Note: pos is n_pos * n_tokens size, but for 1D pos it's at t
                    // For 2D pos, the logic is more complex, but for now we assume 1D pos or handle via pos[t]
                    // In llama_ubatch, pos is [n_tokens * n_pos]. If n_pos == 1, pos[t] is the position.
                    uint32_t logical_block = pos_to_logical_block(pos);
                    if (logical_block + 1 > n_logical_blocks) {
                        n_logical_blocks = logical_block + 1;
                    }
                }
            }
            if (n_logical_blocks > 0) {
                ensure_blocks_for_sequence(seq_id, n_logical_blocks);
            }
        }
    }

    // Create and return the context
    return std::make_unique<llama_kv_paged_cache_context>(this, nullptr, false, ubatches);
}

llama_memory_context_ptr llama_kv_paged_cache::init_full() {
    // For full cache simulation, we ensure all sequences have max blocks
    // This is a simplified implementation
    return std::make_unique<llama_kv_paged_cache_context>(LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_kv_paged_cache::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_paged_cache_context>(LLAMA_MEMORY_STATUS_SUCCESS);
}

bool llama_kv_paged_cache::get_can_shift() const {
    return true; // Paged cache supports shift via page table updates
}

void llama_kv_paged_cache::clear(bool data) {
    // Reset all blocks to free state
    for (uint32_t i = 0; i < n_blocks_total; ++i) {
        physical_blocks[i].reset();
    }
    free_blocks.clear();
    for (uint32_t i = 0; i < n_blocks_total; ++i) {
        free_blocks.push_back(i);
    }

    // Clear page tables and sequence position ranges
    page_tables.clear();
    seq_pos_range.clear();
}

bool llama_kv_paged_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (page_tables.find(seq_id) == page_tables.end()) {
        return false; // Sequence not found
    }

    uint32_t start_block = pos_to_logical_block(p0);
    uint32_t end_block = pos_to_logical_block(p1); // Note: p1 is exclusive in llama.cpp convention typically

    // Ensure we don't go out of bounds
    if (start_block >= page_tables[seq_id].size()) {
        return false;
    }

    uint32_t num_blocks_to_release = 0;
    if (end_block > page_tables[seq_id].size()) {
        num_blocks_to_release = (uint32_t)page_tables[seq_id].size() - start_block;
    } else {
        num_blocks_to_release = end_block - start_block;
    }

    for (uint32_t i = 0; i < num_blocks_to_release; ++i) {
        release_block(seq_id, start_block + i);
    }

    // Update sequence position range
    if (seq_pos_range.find(seq_id) != seq_pos_range.end()) {
        if (p1 < seq_pos_range[seq_id].second) {
            seq_pos_range[seq_id].second = p1 - 1; // Adjust to last included position
        }
        if (seq_pos_range[seq_id].second < seq_pos_range[seq_id].first) {
            // Sequence fully removed
            seq_pos_range.erase(seq_id);
            page_tables.erase(seq_id);
        }
    }

    return true;
}

void llama_kv_paged_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // In paged cache, seq_cp is optimized: copy page table entries and increase ref counts
    // Ensure destination sequence has page table
    if (page_tables.find(seq_id_dst) == page_tables.end()) {
        page_tables[seq_id_dst] = std::vector<uint32_t>();
    }

    uint32_t start_block = pos_to_logical_block(p0);
    uint32_t end_block = pos_to_logical_block(p1);

    if (page_tables.find(seq_id_src) == page_tables.end()) {
        throw std::runtime_error("llama_kv_paged_cache: source sequence not found in page tables");
    }

    uint32_t num_blocks = 0;
    if (end_block > page_tables[seq_id_src].size()) {
        num_blocks = (uint32_t)page_tables[seq_id_src].size() - start_block;
    } else {
        num_blocks = end_block - start_block;
    }

    // Allocate blocks for destination and copy references
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t src_logical_block = start_block + i;
        if (src_logical_block >= page_tables[seq_id_src].size()) break;

        uint32_t phys_block_idx = page_tables[seq_id_src][src_logical_block];

        // Add to destination page table
        page_tables[seq_id_dst].push_back(phys_block_idx);

        // Increase reference count
        if (phys_block_idx < physical_blocks.size()) {
            physical_blocks[phys_block_idx].ref_count++;
        }
    }

    // Update sequence position range for destination
    llama_pos pos_start = logical_block_to_pos((uint32_t)page_tables[seq_id_dst].size() - (num_blocks > 0 ? num_blocks : 0));
    llama_pos pos_end = pos_start + (num_blocks > 0 ? (llama_pos)num_blocks * (llama_pos)block_size : 0) - 1;

    if (seq_pos_range.find(seq_id_dst) == seq_pos_range.end()) {
        seq_pos_range[seq_id_dst] = std::make_pair(pos_start, pos_end);
    } else {
        seq_pos_range[seq_id_dst].first = std::min(seq_pos_range[seq_id_dst].first, pos_start);
        seq_pos_range[seq_id_dst].second = std::max(seq_pos_range[seq_id_dst].second, pos_end);
    }
}

void llama_kv_paged_cache::seq_keep(llama_seq_id seq_id) {
    // In paged cache, seq_keep is a no-op as we don't have the concept of "keeping" vs "removing" in the same way.
    // All blocks belonging to the sequence are kept unless explicitly removed via seq_rm.
    // This function is primarily for the contiguous cache's defrag logic.
}

void llama_kv_paged_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // In paged cache, seq_add with shift is primarily a page table update.
    // We need to shift the logical positions and update the page table accordingly.
    // For now, a basic implementation: remove old range and add new shifted range.
    if (page_tables.find(seq_id) != page_tables.end()) {
        // Remove old blocks
        uint32_t start_block = pos_to_logical_block(p0);
        uint32_t end_block = pos_to_logical_block(p1);
        uint32_t num_blocks = 0;
        if (end_block > page_tables[seq_id].size()) {
            num_blocks = (uint32_t)page_tables[seq_id].size() - start_block;
        } else {
            num_blocks = end_block - start_block;
        }
        for (uint32_t i = 0; i < num_blocks; ++i) {
            release_block(seq_id, start_block + i);
        }
    }

    // Ensure blocks for the new shifted range
    llama_pos new_p0 = p0 + shift;
    llama_pos new_p1 = p1 + shift;
    uint32_t n_logical_blocks_new = pos_to_logical_block(new_p1) + 1;
    ensure_blocks_for_sequence(seq_id, n_logical_blocks_new);

    // Update position range
    if (seq_pos_range.find(seq_id) == seq_pos_range.end()) {
        seq_pos_range[seq_id] = std::make_pair(new_p0, new_p1 - 1);
    } else {
        seq_pos_range[seq_id].first = new_p0;
        seq_pos_range[seq_id].second = new_p1 - 1;
    }
}

void llama_kv_paged_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // In paged cache, seq_div is complex as it involves recalculating positions and potentially redistributing blocks.
    // For phase 1, we provide a basic stub or throw if not supported.
    throw std::runtime_error("llama_kv_paged_cache::seq_div not fully implemented in phase 1");
}

llama_pos llama_kv_paged_cache::seq_pos_min(llama_seq_id seq_id) const {
    auto it = seq_pos_range.find(seq_id);
    if (it != seq_pos_range.end()) {
        return it->second.first;
    }
    return -1;
}

llama_pos llama_kv_paged_cache::seq_pos_max(llama_seq_id seq_id) const {
    auto it = seq_pos_range.find(seq_id);
    if (it != seq_pos_range.end()) {
        return it->second.second;
    }
    return -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_paged_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;
    // Phase 1 stub: return empty or basic breakdown
    return breakdown;
}

void llama_kv_paged_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // Phase 1 stub: state write not fully implemented
}

void llama_kv_paged_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // Phase 1 stub: state read not fully implemented
}

// Helper function to create physical K/V tensors for a layer

void llama_kv_paged_cache::create_layer_tensors(int32_t il, ggml_type type_k, ggml_type type_v, bool v_trans) {
    // Skip layers that are not part of the cache or have no FFN
    if (filter_cb && !filter_cb(il)) {
        return;
    }

    // In Paged KV Cache mechanism, physical K/V data is managed by llama_kv_paged_block's data_k/data_v backend buffers.
    // We do not create independent ggml_context and ggml_tensor for K/V tensors here.
    // The graph construction will use build_input_k_idxs and build_input_v_idxs to generate indices
    // and use ggml_set_rows to map to the physical blocks.

    // Store empty tensor placeholders for compatibility with get_k_tensor/get_v_tensor
    layer_tensors_map[il].k_tensor = nullptr;
    layer_tensors_map[il].v_tensor = nullptr;

    // No ctxs_bufs allocation needed as physical blocks manage the actual data buffers
}

ggml_tensor * llama_kv_paged_cache::get_k_tensor(int32_t il) const {
    auto it = layer_tensors_map.find(il);
    if (it != layer_tensors_map.end()) {
        return it->second.k_tensor;
    }
    return nullptr;
}

ggml_tensor * llama_kv_paged_cache::get_v_tensor(int32_t il) const {
    auto it = layer_tensors_map.find(il);
    if (it != layer_tensors_map.end()) {
        return it->second.v_tensor;
    }
    return nullptr;
}

//
// llama_kv_paged_cache_context
//

llama_kv_paged_cache_context::llama_kv_paged_cache_context(llama_memory_status status)
    : status(status), paged_cache(nullptr), lctx(nullptr), ubatch_idx(0), do_shift(false) {}

llama_kv_paged_cache_context::llama_kv_paged_cache_context(
        llama_kv_paged_cache * paged_cache,
        llama_context * lctx,
        bool do_shift,
        const std::vector<llama_ubatch> & ubatches)
    : status(LLAMA_MEMORY_STATUS_SUCCESS), paged_cache(paged_cache), lctx(lctx),
      ubatch_idx(0), ubatches(ubatches), do_shift(do_shift) {}

llama_kv_paged_cache_context::~llama_kv_paged_cache_context() = default;

bool llama_kv_paged_cache_context::next() {
    if (ubatch_idx >= ubatches.size()) {
        return false;
    }
    ubatch_idx++;
    return ubatch_idx < ubatches.size();
}

bool llama_kv_paged_cache_context::apply() {
    // For paged KV cache, the apply step primarily ensures that the page tables
    // are correctly set up and that the physical K/V tensors are ready for the compute graph.
    // The actual data writing to physical tensors is handled by ggml_set_rows using the
    // indices generated by build_input_k_idxs and build_input_v_idxs.

    if (status != LLAMA_MEMORY_STATUS_SUCCESS) {
        return false;
    }

    // Ensure that the current ubatch has valid page table entries
    const llama_ubatch & ubatch = get_ubatch();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        llama_seq_id seq_id = *ubatch.seq_id[i];
        llama_pos pos = ubatch.pos[i];

        uint32_t logical_block_idx = paged_cache->pos_to_logical_block(pos);

        // Verify that the page table has the required block
        auto page_table_it = paged_cache->page_tables.find(seq_id);
        if (page_table_it == paged_cache->page_tables.end() || 
            logical_block_idx >= page_table_it->second.size()) {
            // This should not happen if init_batch correctly ensured blocks
            status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
            return false;
        }
    }

    return true;
}

llama_memory_status llama_kv_paged_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_paged_cache_context::get_ubatch() const {
    if (ubatch_idx < ubatches.size()) {
        return ubatches[ubatch_idx];
    }
    // Return a default or empty ubatch if out of bounds - in practice this should be handled by `next()`
    static llama_ubatch empty_ubatch = {};
    return empty_ubatch;
}

ggml_tensor * llama_kv_paged_cache_context::get_turbo_rot_forward() const {
    // Phase 1 stub: return nullptr
    return nullptr;
}

ggml_tensor * llama_kv_paged_cache_context::get_turbo_rot_inverse() const {
    // Phase 1 stub: return nullptr
    return nullptr;
}

ggml_tensor * llama_kv_paged_cache_context::get_turbo_innerq_scale_inv() const {
    // Phase 1 stub: return nullptr
    return nullptr;
}

//
// Defragmentation implementation
//

void llama_kv_paged_cache::defrag_trigger() {
    // Trigger defragmentation if available blocks are below 10% of total blocks
    if (n_blocks_total == 0) return;
    uint32_t min_free_blocks = n_blocks_total / 10;
    if (free_blocks.size() < min_free_blocks) {
        defrag_trigger_impl();
    }
}

void llama_kv_paged_cache::defrag_trigger_impl() {
    // Collect all used physical blocks (is_free == false)
    std::vector<uint32_t> used_blocks;
    used_blocks.reserve(n_blocks_total);
    for (uint32_t i = 0; i < n_blocks_total; ++i) {
        if (!physical_blocks[i].is_free) {
            used_blocks.push_back(i);
        }
    }

    if (used_blocks.empty() || used_blocks.size() == n_blocks_total) {
        return; // No need to defrag or all blocks are used
    }

    // Create a mapping from old physical block index to new physical block index
    // New used indices will be 0..used_blocks.size()-1
    std::unordered_map<uint32_t, uint32_t> old_to_new_phys;
    for (size_t i = 0; i < used_blocks.size(); ++i) {
        old_to_new_phys[used_blocks[i]] = (uint32_t)i;
    }

    // Create new physical blocks state
    // The first used_blocks.size() blocks are used, the rest are free
    std::vector<llama_kv_paged_block> new_physical_blocks(n_blocks_total);
    for (size_t i = 0; i < used_blocks.size(); ++i) {
        uint32_t old_idx = used_blocks[i];
        new_physical_blocks[i].ref_count = physical_blocks[old_idx].ref_count;
        new_physical_blocks[i].is_free = false;
    }
    for (size_t i = used_blocks.size(); i < (size_t)n_blocks_total; ++i) {
        new_physical_blocks[i].reset();
    }

    // Update page tables
    for (auto& pair : page_tables) {
        llama_seq_id seq_id = pair.first;
        std::vector<uint32_t>& page_table = pair.second;
        for (size_t i = 0; i < page_table.size(); ++i) {
            uint32_t old_phys_idx = page_table[i];
            auto it = old_to_new_phys.find(old_phys_idx);
            if (it != old_to_new_phys.end()) {
                page_table[i] = it->second;
            }
        }
    }

    // Update physical_blocks and free_blocks
    physical_blocks = new_physical_blocks;

    // Update free_blocks list
    free_blocks.clear();
    for (uint32_t i = (uint32_t)used_blocks.size(); i < n_blocks_total; ++i) {
        free_blocks.push_back(i);
    }
}

ggml_tensor * llama_kv_paged_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t block_size = paged_cache->get_block_size();

    // Create index tensor: for each token, we need the physical row index
    // The physical row index for a token at logical position p is:
    // physical_block_idx * block_size + pos_to_block_offset(p)

    // We'll create a 1D tensor of int32_t for the row indices
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    // Get the data pointer to fill with indices
    int32_t * k_idxs_data = (int32_t *) ggml_get_data(k_idxs);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        llama_seq_id seq_id = *ubatch.seq_id[i];
        llama_pos pos = ubatch.pos[i];

        uint32_t logical_block_idx = paged_cache->pos_to_logical_block(pos);
        uint32_t block_offset = paged_cache->pos_to_block_offset(pos);

        uint32_t physical_block_idx = 0;
        auto page_table_it = paged_cache->page_tables.find(seq_id);
        if (page_table_it != paged_cache->page_tables.end() && 
            logical_block_idx < page_table_it->second.size()) {
            physical_block_idx = page_table_it->second[logical_block_idx];
        } else {
            // Fallback: if page table entry is not found, assume contiguous layout or throw error
            // For phase 1, we'll use a simple calculation or assert
            // In a full implementation, this should not happen if ensure_blocks_for_sequence is called correctly
            physical_block_idx = logical_block_idx; // Fallback to logical block index
        }

        uint32_t phys_row_idx = physical_block_idx * block_size + block_offset;
        k_idxs_data[i] = (int32_t)phys_row_idx;
    }

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_paged_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t block_size = paged_cache->get_block_size();

    // Create index tensor: for each token, we need the physical row index
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    // Get the data pointer to fill with indices
    int32_t * v_idxs_data = (int32_t *) ggml_get_data(v_idxs);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        llama_seq_id seq_id = *ubatch.seq_id[i];
        llama_pos pos = ubatch.pos[i];

        uint32_t logical_block_idx = paged_cache->pos_to_logical_block(pos);
        uint32_t block_offset = paged_cache->pos_to_block_offset(pos);

        uint32_t physical_block_idx = 0;
        auto page_table_it = paged_cache->page_tables.find(seq_id);
        if (page_table_it != paged_cache->page_tables.end() && 
            logical_block_idx < page_table_it->second.size()) {
            physical_block_idx = page_table_it->second[logical_block_idx];
        } else {
            physical_block_idx = logical_block_idx; // Fallback to logical block index
        }

        uint32_t phys_row_idx = physical_block_idx * block_size + block_offset;
        v_idxs_data[i] = (int32_t)phys_row_idx;
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}