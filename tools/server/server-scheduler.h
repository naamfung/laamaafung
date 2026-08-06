#pragma once

#include "llama.h"

#include <algorithm>
#include <numeric>
#include <vector>

// Decoupled preemption/scheduling decisions for the paged KV cache
// (extracted from llama_server_context::pre_decode).
//
// Slots are prepared in priority order: higher-priority slots are restored
// and reserve capacity first, so under memory pressure the low-priority
// slots are the ones that get preempted. The per-slot priority is also
// synced to the cache via llama_memory_seq_set_priority(), which makes
// preempt_one pick the lowest-priority victim.
//
// The slot vector itself is never reordered - preparation uses a
// priority-sorted index list so downstream batch building sees the
// original slot order.
struct server_slot;

class llama_server_scheduler {
public:
    // restore swapped-out slots, sync priorities, and reserve capacity for
    // the next token of every generating slot, in priority order
    static void pre_decode_prepare(
            std::vector<server_slot> & slots,
            llama_memory_t mem) {
        // 1. restore any swapped-out slots before batch build
        for (auto & slot : slots) {
            if (!slot.is_processing()) {
                continue;
            }
            if (llama_memory_is_swapped(mem, slot.id)) {
                if (!llama_memory_swap_in(mem, slot.id)) {
                    fprintf(stderr, "server: swap_in failed for slot %d\n", slot.id);
                }
            }
        }

        // 2. sync per-slot priorities to the cache so preempt_one victim
        //    selection honors them
        for (auto & slot : slots) {
            llama_memory_seq_set_priority(mem, slot.id, slot.priority);
        }

        // 3. proactive capacity check in priority order (stable, so equal
        //    priorities keep their original order). ensures the cache has
        //    room for the next token before batch build, avoiding reactive
        //    OOM in alloc_block which would evict mid-batch.
        std::vector<size_t> order(slots.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return slots[a].priority > slots[b].priority; });

        for (size_t idx : order) {
            auto & slot = slots[idx];
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }
            const int n_freed = llama_memory_ensure_capacity(mem, slot.id, 1);
            if (n_freed > 0) {
                fprintf(stderr, "slot %d: proactive preempt: freed %d block(s) for next token\n",
                        slot.id, n_freed);
            }
        }
    }
};
