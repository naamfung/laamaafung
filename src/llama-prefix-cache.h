#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

// vllm-style prefix cache for the server layer.
// Hash-chain of fixed-size token blocks (BLOCK_SIZE), with ref-counting
// delegated to the KV cells' seq bitsets: seq_cp adds the dst seq to a
// cell's bitset, seq_rm removes it; a cell is freed only when its bitset
// becomes empty. This class is a pure server-layer index: it does NOT
// touch llama_kv_cache internals.
class llama_prefix_cache {
public:
    static constexpr uint32_t BLOCK_SIZE = 256;

    struct prefix_hit {
        llama_seq_id src_seq   = -1;
        uint32_t     n_blocks  = 0; // number of full blocks matched

        bool     empty()    const { return n_blocks == 0; }
        uint32_t n_tokens() const { return n_blocks * BLOCK_SIZE; }
    };

    // search for the longest prefix of `tokens` (up to n tokens) that is
    // fully present in the cache and owned by a single source seq.
    // returns prefix_hit with src_seq = -1 if no full block matches.
    prefix_hit find_prefix(const llama_token * tokens, uint32_t n) const;

    // register all full blocks of `tokens` (up to n tokens) as belonging to
    // `seq_id`. Idempotent: re-registering the same hash is a no-op.
    // Both source and sharer must register so the index survives either
    // being cleared.
    void register_blocks(llama_seq_id seq_id, const llama_token * tokens, uint32_t n);

    // remove all hash registrations for `seq_id` from both maps.
    // call before any KV clear (seq_rm, context shift, prompt_clear) so
    // stale entries are not served to future find_prefix calls.
    void revoke_seq(llama_seq_id seq_id);

    // convenience wrapper: just calls revoke_seq(seq_id). Range arguments
    // are ignored because the index does not track per-position ownership -
    // any KV clear within a registered range invalidates the whole chain.
    void revoke_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    void clear();

private:
    // chained block hash: block i's hash is XXH64(tokens[i], seed=hash[i-1])
    // with hash[-1] = 0. This makes the chain order-dependent: the same
    // block content at a different position yields a different hash.
    static uint64_t compute_hash(uint64_t prev_hash, const llama_token * tokens, uint32_t n);

    using seq_set  = std::unordered_set<llama_seq_id>;
    using hash_set = std::unordered_set<uint64_t>;

    // hash -> seqs that own a block with this hash
    std::unordered_map<uint64_t,      seq_set> hash_to_seqs;
    // seq  -> block hashes registered for this seq
    std::unordered_map<llama_seq_id, hash_set> seq_to_hashes;
};
