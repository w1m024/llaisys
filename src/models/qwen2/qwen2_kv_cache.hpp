#pragma once

#include "../../tensor/tensor.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <list>
#include <iostream>

namespace llaisys::models::qwen2 {

// -----------------------------------------------------------------------------
// KVCacheBlock: Represents a fixed-size block of KV cache memory
// -----------------------------------------------------------------------------
struct KVCacheBlock {
    int id;
    size_t size; // Number of tokens capacity (e.g. 16)
    size_t used; // Number of tokens currently stored
    
    // Physical storage for K and V
    // Shape: [nlayers, size, nkvhead, head_dim]
    // To simplify memory management, we can allocate one large tensor per block
    // or separate tensors per layer.
    // For PagedAttention efficiency, usually it's [num_blocks, nhead, block_size, head_dim] globally.
    // But here we manage per block.
    
    // Let's store K and V as vectors of tensors (one per layer)
    // Each tensor shape: [size, nkvhead, head_dim]
    std::vector<tensor_t> k_blocks; 
    std::vector<tensor_t> v_blocks;
    
    // For Prefix Caching: Hash of the token sequence stored in this block
    size_t content_hash = 0;
    // Pointers to next block in sequence (if we want linked list structure for prefix tree)
    // Or we manage tree externally.
    
    KVCacheBlock(int id, size_t size, size_t nlayers, size_t nkvhead, size_t head_dim, 
                 llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id)
        : id(id), size(size), used(0) {
        
        k_blocks.reserve(nlayers);
        v_blocks.reserve(nlayers);
        
        for (size_t i = 0; i < nlayers; ++i) {
            k_blocks.push_back(Tensor::create({size, nkvhead, head_dim}, dtype, device, device_id));
            v_blocks.push_back(Tensor::create({size, nkvhead, head_dim}, dtype, device, device_id));
        }
    }
    
    void reset() {
        used = 0;
        content_hash = 0;
    }
};

// -----------------------------------------------------------------------------
// BlockManager: Manages the pool of blocks and handles allocation/freeing
// -----------------------------------------------------------------------------
class BlockManager {
public:
    BlockManager(size_t block_size, size_t num_blocks, 
                 size_t nlayers, size_t nkvhead, size_t head_dim,
                 llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) 
        : _block_size(block_size) {
        
        for (size_t i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<KVCacheBlock>(
                i, block_size, nlayers, nkvhead, head_dim, dtype, device, device_id
            );
            _free_blocks.push_back(block);
            _all_blocks.push_back(block);
        }
    }
    
    std::shared_ptr<KVCacheBlock> allocate() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_free_blocks.empty()) {
            // Eviction policy needed here! (LRU)
            // For now, return null or throw if OOM
            return nullptr;
        }
        
        auto block = _free_blocks.front();
        _free_blocks.pop_front();
        block->reset();
        return block;
    }
    
    void free(std::shared_ptr<KVCacheBlock> block) {
        std::lock_guard<std::mutex> lock(_mutex);
        block->reset();
        _free_blocks.push_back(block);
    }
    
    // Prefix Caching: Find a block that matches the hash
    std::shared_ptr<KVCacheBlock> find_cached(size_t hash) {
        // Implementation of Radix Tree or Hash Map lookup
        // For P4-4, we can start with simple exact match if we store hash
        // But blocks are usually chained.
        // Let's skip complex prefix matching implementation for now 
        // and focus on the Block Allocator mechanism first.
        return nullptr;
    }

private:
    size_t _block_size;
    std::vector<std::shared_ptr<KVCacheBlock>> _all_blocks;
    std::list<std::shared_ptr<KVCacheBlock>> _free_blocks;
    std::mutex _mutex;
};

} // namespace llaisys::models::qwen2
