//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parallel/pipeline_executor.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/common/stack.hpp"

#include <functional>

namespace duckdb {
class Executor;

//! The result of executing a PipelineExecutor
enum class PipelineExecuteResult {
	//! PipelineExecutor is fully executed: the source is completely exhausted
	FINISHED,
	//! PipelineExecutor is not yet fully executed and can be called again immediately
	NOT_FINISHED,
	//! The PipelineExecutor was interrupted and should not be called again until the interrupt is handled as specified
	//! in the InterruptMode
	INTERRUPTED
};

class ExecutionBudget {
public:
	explicit ExecutionBudget(idx_t maximum) : processed(0), maximum_to_process(maximum) {
	}

public:
	bool Next() {
		if (IsDepleted()) {
			return false;
		}
		processed++;
		return true;
	}
	bool IsDepleted() const {
		return processed >= maximum_to_process;
	}

private:
	idx_t processed;
	idx_t maximum_to_process;
};

class LIPProbeInfo {
public:
	LIPProbeInfo() = delete;

	explicit LIPProbeInfo(const vector<shared_ptr<LIPBloomFilter>>& lip_filters) :
		lip_filters(lip_filters), probe_order(lip_filters.size()), bf_hit_counts(lip_filters.size()),
		bf_total_counts(lip_filters.size()), bf_hit_percentages(lip_filters.size()),
		hash_staging(LogicalType::HASH), probe_res(STANDARD_VECTOR_SIZE), probe_sel(STANDARD_VECTOR_SIZE) {
		for (idx_t i = 0; i < lip_filters.size(); i++) {
			probe_order[i] = i;
		}
	}

	//! Reference to filters being probed
	const vector<shared_ptr<LIPBloomFilter>>& lip_filters;
	//! Order in which to probe lip_filters
	vector<idx_t> probe_order;
	//! Current chunks processed in batch
	size_t chunks_processed = 0;
	//! Total chunks in batch
	size_t batch_size = 1;
	//! Hit count of each BF
	vector<size_t> bf_hit_counts;
	//! Total counts of each BF
	vector<size_t> bf_total_counts;
	//! Percentage of each BF
	vector<pair<double, idx_t>> bf_hit_percentages;

	//! staging area for hashes
	Vector hash_staging;
	//! results vector for probes (no need to clear between probes)
	vector<uint32_t> probe_res;
	//! sel vector for probe results (no need to clear between probes)
	SelectionVector probe_sel;

	//! Probe BFs in current order
	void ProbeBFs(DataChunk &chunk) {
		if (chunk.size() == 0) {
			return;
		}

		for (size_t idx = 0; idx < probe_order.size(); idx++) {
			// TODO: why is this needed???????
			probe_sel.Initialize();

			auto &filter = lip_filters[probe_order[idx]];
			// TODO: there has to be a better way to do this lmao
			//  e.g. Lookup can just return a new chunk ? or at least a sel vector
			filter->Lookup(chunk, probe_res, hash_staging);

			idx_t result_count = 0;
			for (idx_t i = 0; i < chunk.size(); i++) {
				probe_sel.set_index(result_count, i);
				result_count += probe_res[i];
				// if (probe_res[i] > 0) {
				// 	probe_sel.set_index(result_count, i);
				// 	result_count++;
				// }
			}
			bf_hit_counts[idx] += result_count;
			bf_total_counts[idx] += chunk.size();

			if (result_count == 0) {
				chunk.Slice(0, 0);
				break;
			}
			if (result_count != chunk.size()) {
				chunk.Slice(probe_sel, result_count);
			}
		}

		chunks_processed++;
		if (chunks_processed == batch_size) {
			ReorderProbes();
		}
	}

	//! On batch completion, reorder BFs such that more selective BFs are first
	void ReorderProbes() {
		D_ASSERT(chunks_processed == batch_size);

		for (size_t i = 0; i < bf_hit_counts.size(); i++) {
			size_t bf_hit_count = bf_hit_counts[i];
			size_t bf_total_count = bf_total_counts[i];
			// minumum threshold to avoid noisy reorders
			if (bf_total_count <= 32 * batch_size) {
				bf_hit_percentages[i] = {1.0, probe_order[i]};
				continue;
			}
			double bf_hit_percentage = static_cast<double>(bf_hit_count) / static_cast<double>(bf_total_count);
			bf_hit_percentages[i] = {bf_hit_percentage, probe_order[i]};
		}

		// sort by increasing hit rate
		std::sort(bf_hit_percentages.begin(), bf_hit_percentages.end());
		for (idx_t i = 0; i < bf_hit_percentages.size(); i++) {
			probe_order[i] = bf_hit_percentages[i].second;
		}

		// Reset state for next batch
		chunks_processed = 0;
		batch_size *= 2;
		for (size_t i = 0; i < bf_hit_counts.size(); i++) {
			bf_hit_counts[i] = 0;
		}
		for (size_t i = 0; i < bf_total_counts.size(); i++) {
			bf_total_counts[i] = 0;
		}
	}
};

//! The Pipeline class represents an execution pipeline
class PipelineExecutor {
public:
	PipelineExecutor(ClientContext &context, Pipeline &pipeline);

	//! Fully execute a pipeline with a source and a sink until the source is completely exhausted
	PipelineExecuteResult Execute();
	//! Execute a pipeline with a source and a sink until finished, or until max_chunks were processed from the source
	//! Returns true if execution is finished, false if Execute should be called again
	PipelineExecuteResult Execute(idx_t max_chunks);

	//! Called after depleting the source: finalizes the execution of this pipeline executor
	//! This should only be called once per PipelineExecutor.
	PipelineExecuteResult PushFinalize();

	bool RemainingSinkChunk() const;

	//! Initializes a chunk with the types that will flow out of the chunk
	void InitializeChunk(DataChunk &chunk);
	//! Execute a pipeline without a sink, and retrieve a single DataChunk
	//! Returns an empty chunk when finished.

	//! Registers the task in the interrupt_state to allow Source/Sink operators to block the task
	void SetTaskForInterrupts(weak_ptr<Task> current_task);

private:
	//! The pipeline to process
	Pipeline &pipeline;
	//! The thread context of this executor
	ThreadContext thread;
	//! The total execution context of this executor
	ExecutionContext context;

	//! Intermediate chunks for the operators
	vector<unique_ptr<DataChunk>> intermediate_chunks;
	//! Intermediate states for the operators
	vector<unique_ptr<OperatorState>> intermediate_states;

	//! The local source state
	unique_ptr<LocalSourceState> local_source_state;
	//! The local sink state (if any)
	unique_ptr<LocalSinkState> local_sink_state;
	//! The interrupt state, holding required information for sink/source operators to block
	InterruptState interrupt_state;

	//! The final chunk used for moving data into the sink
	DataChunk final_chunk;

	//! The operators that are not yet finished executing and have data remaining
	//! If the stack of in_process_operators is empty, we fetch from the source instead
	stack<idx_t> in_process_operators;
	//! Whether or not the pipeline has been finalized (used for verification only)
	bool finalized = false;
	//! Whether or not the pipeline has finished processing
	int32_t finished_processing_idx = -1;
	//! Partition info that is used by this executor
	OperatorPartitionInfo required_partition_info;

	//! Source has indicated it is exhausted
	bool exhausted_source = false;
	//! Flushing of intermediate operators has started
	bool started_flushing = false;
	//! Flushing of caching operators is done
	bool done_flushing = false;

	//! This flag is set when the pipeline gets interrupted by the Sink -> the final_chunk should be re-sink-ed.
	bool remaining_sink_chunk = false;

	//! This flag is set when the pipeline gets interrupted by NextBatch -> NextBatch should be called again and the
	//! source_chunk should be sent through the pipeline
	bool next_batch_blocked = false;

	//! Current operator being flushed
	idx_t flushing_idx;
	//! Whether the current flushing_idx should be flushed: this needs to be stored to make flushing code re-entrant
	bool should_flush_current_idx = true;

	/* LIP *******************************************************************/

	//!
	unique_ptr<LIPProbeInfo> lip_info;

private:
	void StartOperator(PhysicalOperator &op);
	void EndOperator(PhysicalOperator &op, optional_ptr<DataChunk> chunk);

	//! Reset the operator index to the first operator
	void GoToSource(idx_t &current_idx, idx_t initial_idx);
	SourceResultType FetchFromSource(DataChunk &result);

	void FinishProcessing(int32_t operator_idx = -1);
	bool IsFinished();

	//! Wrappers for sink/source calls to respective operators
	SourceResultType GetData(DataChunk &chunk, OperatorSourceInput &input);
	SinkResultType Sink(DataChunk &chunk, OperatorSinkInput &input);

	OperatorResultType ExecutePushInternal(DataChunk &input, ExecutionBudget &chunk_budget, idx_t initial_idx = 0);
	//! Pushes a chunk through the pipeline and returns a single result chunk
	//! Returns whether or not a new input chunk is needed, or whether or not we are finished
	OperatorResultType Execute(DataChunk &input, DataChunk &result, idx_t initial_index = 0);

	//! Notifies the sink that a new batch has started
	SinkNextBatchType NextBatch(DataChunk &source_chunk);

	//! Tries to flush all state from intermediate operators. Will return true if all state is flushed, false in the
	//! case of a blocked sink.
	bool TryFlushCachingOperators(ExecutionBudget &chunk_budget);

	static bool CanCacheType(const LogicalType &type);
	void CacheChunk(DataChunk &input, idx_t operator_idx);

#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	//! Debugging state: number of times blocked
	int debug_blocked_sink_count = 0;
	int debug_blocked_source_count = 0;
	int debug_blocked_combine_count = 0;
	int debug_blocked_next_batch_count = 0;
	//! Number of times the Sink/Source will block before actually returning data
	int debug_blocked_target_count = 1;
#endif
};

} // namespace duckdb
