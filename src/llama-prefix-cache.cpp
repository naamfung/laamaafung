#include "llama-prefix-cache.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

llama_prefix_cache::prefix_hit llama_prefix_cache::find_prefix(const llama_token * tokens, uint32_t n) const {
    prefix_hit hit;
    if (n < BLOCK_SIZE) {
        return hit;
    }

    const uint32_t n_blocks = n / BLOCK_SIZE;
    seq_set owners; // intersection of owners across the matched chain
    uint64_t prev_hash = 0;

    for (uint32_t i = 0; i < n_blocks; i++) {
        const uint64_t h = compute_hash(prev_hash, tokens + i * BLOCK_SIZE, BLOCK_SIZE);

        auto it = hash_to_seqs.find(h);
        if (it == hash_to_seqs.end() || it->second.empty()) {
            break;
        }

        if (i == 0) {
            owners = it->second;
        } else {
            seq_set next;
            for (llama_seq_id s : owners) {
                if (it->second.count(s)) {
                    next.insert(s);
                }
            }
            owners = std::move(next);
        }

        if (owners.empty()) {
            break;
        }

        prev_hash = h;
        hit.n_blocks = i + 1;
    }

    if (hit.n_blocks > 0 && !owners.empty()) {
        hit.src_seq = *owners.begin();
    }

    return hit;
}

void llama_prefix_cache::register_blocks(llama_seq_id seq_id, const llama_token * tokens, uint32_t n) {
    if (seq_id < 0 || n < BLOCK_SIZE) {
        return;
    }

    const uint32_t n_blocks = n / BLOCK_SIZE;
    uint64_t prev_hash = 0;

    auto & hashes = seq_to_hashes[seq_id];

    for (uint32_t i = 0; i < n_blocks; i++) {
        const uint64_t h = compute_hash(prev_hash, tokens + i * BLOCK_SIZE, BLOCK_SIZE);

        hash_to_seqs[h].insert(seq_id);
        hashes.insert(h);

        prev_hash = h;
    }
}

void llama_prefix_cache::revoke_seq(llama_seq_id seq_id) {
    auto it = seq_to_hashes.find(seq_id);
    if (it == seq_to_hashes.end()) {
        return;
    }

    for (uint64_t h : it->second) {
        auto hit = hash_to_seqs.find(h);
        if (hit != hash_to_seqs.end()) {
            hit->second.erase(seq_id);
            if (hit->second.empty()) {
                hash_to_seqs.erase(hit);
            }
        }
    }

    seq_to_hashes.erase(it);
}

void llama_prefix_cache::revoke_range(llama_seq_id seq_id, llama_pos /*p0*/, llama_pos /*p1*/) {
    revoke_seq(seq_id);
}

void llama_prefix_cache::clear() {
    hash_to_seqs.clear();
    seq_to_hashes.clear();
}

uint64_t llama_prefix_cache::compute_hash(uint64_t prev_hash, const llama_token * tokens, uint32_t n) {
    return XXH64(tokens, (size_t) n * sizeof(llama_token), prev_hash);
}
