#include "duckdb/optimizer/lip/bloom_filter.hpp"

#include "duckdb/storage/buffer_manager.hpp"

#include <random>
#include <cmath>
#include <iostream>

namespace duckdb {
namespace {
static uint32_t CeilPowerOfTwo(uint32_t n) {
	if (n <= 1) {
		return 1;
	}
	n--;
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	return n + 1;
}

void HashColumns(DataChunk &chunk, const vector<idx_t> &cols, Vector &hashes) {
	auto count = chunk.size();
	// TODO: needed?
	// hashes.Initialize();

	VectorOperations::Hash(chunk.data[cols[0]], hashes, count);
	for (size_t j = 1; j < cols.size(); j++) {
		VectorOperations::CombineHash(hashes, chunk.data[cols[j]], count);
	}

	if (hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		hashes.Flatten(count);
	}
}
} // namespace

void BloomFilter::Initialize(ClientContext &context_p, uint32_t est_num_rows) {
	static_assert(sizeof(uint32_t) == sizeof(std::atomic<uint32_t>), "atomic<uint32_t> must be same size as uint32_t");

	context = &context_p;
	buffer_manager = &BufferManager::GetBufferManager(*context);

	uint32_t min_bits = std::max<uint32_t>(MIN_NUM_BITS, est_num_rows * MIN_NUM_BITS_PER_KEY);
	num_sectors = std::min(CeilPowerOfTwo(min_bits) >> LOG_SECTOR_SIZE, MAX_NUM_SECTORS);
	num_sectors_log = static_cast<uint32_t>(std::log2(num_sectors));

	buf_ = buffer_manager->GetBufferAllocator().Allocate(64 + num_sectors * sizeof(std::atomic<uint32_t>));
	// make sure blocks is a 64-byte aligned pointer, i.e., cache-line aligned
	insert_blocks = reinterpret_cast<std::atomic<uint32_t> *>((64ULL + reinterpret_cast<uint64_t>(buf_.get())) & ~63ULL);
	std::fill_n(insert_blocks, num_sectors, 0);
}

int BloomFilter::Lookup(DataChunk &chunk, vector<uint32_t> &results, const vector<idx_t> &bound_cols_applied, Vector &hash_staging) const {
	int count = static_cast<int>(chunk.size());
	HashColumns(chunk, bound_cols_applied, hash_staging);
	BloomFilterLookup(count, reinterpret_cast<uint64_t *>(hash_staging.GetData()), probe_blocks, results.data());
	return count;
}

void BloomFilter::Insert(DataChunk &chunk, const vector<idx_t> &bound_cols_built, Vector &hash_staging) {
	int count = static_cast<int>(chunk.size());
	HashColumns(chunk, bound_cols_built, hash_staging);
	BloomFilterInsert(count, reinterpret_cast<uint64_t *>(hash_staging.GetData()), insert_blocks);
}

void BloomFilter::Finalize() {
	if (finalized) {
		return;
	}
	finalized = true;

	probe_blocks = reinterpret_cast<uint32_t *>(insert_blocks);
	insert_blocks = nullptr;
}

} // namespace duckdb
