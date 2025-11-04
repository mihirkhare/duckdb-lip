#include "duckdb/parallel/pipeline.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/tree_renderer/text_tree_renderer.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/pipeline_event.hpp"
#include "duckdb/parallel/pipeline_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <iostream>

namespace duckdb {

PipelineTask::PipelineTask(Pipeline &pipeline_p, shared_ptr<Event> event_p)
    : ExecutorTask(pipeline_p.executor, std::move(event_p)), pipeline(pipeline_p) {
}

bool PipelineTask::TaskBlockedOnResult() const {
	// If this returns true, it means the pipeline this task belongs to has a cached chunk
	// that was the result of the Sink method returning BLOCKED
	return pipeline_executor->RemainingSinkChunk();
}

const PipelineExecutor &PipelineTask::GetPipelineExecutor() const {
	return *pipeline_executor;
}

TaskExecutionResult PipelineTask::ExecuteTask(TaskExecutionMode mode) {
	if (!pipeline_executor) {
		pipeline_executor = make_uniq<PipelineExecutor>(pipeline.GetClientContext(), pipeline);
	}

	pipeline_executor->SetTaskForInterrupts(shared_from_this());

	if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
		auto res = pipeline_executor->Execute(PARTIAL_CHUNK_COUNT);

		switch (res) {
		case PipelineExecuteResult::NOT_FINISHED:
			return TaskExecutionResult::TASK_NOT_FINISHED;
		case PipelineExecuteResult::INTERRUPTED:
			return TaskExecutionResult::TASK_BLOCKED;
		case PipelineExecuteResult::FINISHED:
			break;
		}
	} else {
		auto res = pipeline_executor->Execute();
		switch (res) {
		case PipelineExecuteResult::NOT_FINISHED:
			throw InternalException("Execute without limit should not return NOT_FINISHED");
		case PipelineExecuteResult::INTERRUPTED:
			return TaskExecutionResult::TASK_BLOCKED;
		case PipelineExecuteResult::FINISHED:
			break;
		}
	}

	event->FinishTask();
	pipeline_executor.reset();
	return TaskExecutionResult::TASK_FINISHED;
}

Pipeline::Pipeline(Executor &executor_p)
    : executor(executor_p), ready(false), initialized(false), source(nullptr), sink(nullptr) {
}

ClientContext &Pipeline::GetClientContext() {
	return executor.context;
}

bool Pipeline::GetProgress(ProgressData &progress) {
	D_ASSERT(source);
	idx_t source_cardinality = MinValue<idx_t>(source->estimated_cardinality, 1ULL << 48ULL);
	if (source_cardinality < 1) {
		source_cardinality = 1;
	}
	if (!initialized) {
		progress.done = 0;
		progress.total = double(source_cardinality);
		return true;
	}
	auto &client = executor.context;

	progress = source->GetProgress(client, *source_state);
	progress.Normalize(double(source_cardinality));
	progress = sink->GetSinkProgress(client, *sink->sink_state, progress);
	return progress.IsValid();
}

void Pipeline::ScheduleSequentialTask(shared_ptr<Event> &event) {
	vector<shared_ptr<Task>> tasks;
	tasks.push_back(make_uniq<PipelineTask>(*this, event));
	event->SetTasks(std::move(tasks));
}

bool Pipeline::ScheduleParallel(shared_ptr<Event> &event) {
	// check if the sink, source and all intermediate operators support parallelism
	if (!sink->ParallelSink()) {
		return false;
	}
	if (!source->ParallelSource()) {
		return false;
	}
	for (auto &op_ref : operators) {
		auto &op = op_ref.get();
		if (!op.ParallelOperator()) {
			return false;
		}
	}
	auto partition_info = sink->RequiredPartitionInfo();
	if (partition_info.batch_index) {
		if (!source->SupportsPartitioning(OperatorPartitionInfo::BatchIndex())) {
			throw InternalException(
			    "Attempting to schedule a pipeline where the sink requires batch index but source does not support it");
		}
	}
	auto max_threads = source_state->MaxThreads();
	auto &scheduler = TaskScheduler::GetScheduler(executor.context);
	auto active_threads = NumericCast<idx_t>(scheduler.NumberOfThreads());
	if (max_threads > active_threads) {
		max_threads = active_threads;
	}
	if (sink && sink->sink_state) {
		max_threads = sink->sink_state->MaxThreads(max_threads);
	}
	if (max_threads > active_threads) {
		max_threads = active_threads;
	}
	return LaunchScanTasks(event, max_threads);
}

bool Pipeline::IsOrderDependent() const {
	auto &config = DBConfig::GetConfig(executor.context);
	if (source) {
		auto source_order = source->SourceOrder();
		if (source_order == OrderPreservationType::FIXED_ORDER) {
			return true;
		}
		if (source_order == OrderPreservationType::NO_ORDER) {
			return false;
		}
	}
	for (auto &op_ref : operators) {
		auto &op = op_ref.get();
		if (op.OperatorOrder() == OrderPreservationType::NO_ORDER) {
			return false;
		}
		if (op.OperatorOrder() == OrderPreservationType::FIXED_ORDER) {
			return true;
		}
	}
	if (!config.options.preserve_insertion_order) {
		return false;
	}
	if (sink && sink->SinkOrderDependent()) {
		return true;
	}
	return false;
}

void Pipeline::Schedule(shared_ptr<Event> &event) {
	D_ASSERT(ready);
	D_ASSERT(sink);
	Reset();
	if (!ScheduleParallel(event)) {
		// could not parallelize this pipeline: push a sequential task instead
		ScheduleSequentialTask(event);
	}
}

bool Pipeline::LaunchScanTasks(shared_ptr<Event> &event, idx_t max_threads) {
	// split the scan up into parts and schedule the parts
	if (max_threads <= 1) {
		// too small to parallelize
		return false;
	}

	// launch a task for every thread
	vector<shared_ptr<Task>> tasks;
	for (idx_t i = 0; i < max_threads; i++) {
		tasks.push_back(make_uniq<PipelineTask>(*this, event));
	}
	event->SetTasks(std::move(tasks));
	return true;
}

void Pipeline::ResetSink() {
	if (sink) {
		if (!sink->IsSink()) {
			throw InternalException("Sink of pipeline does not have IsSink set");
		}
		lock_guard<mutex> guard(sink->lock);
		if (!sink->sink_state) {
			sink->sink_state = sink->GetGlobalSinkState(GetClientContext());
		}
	}
}

void Pipeline::PrepareFinalize() {
	if (sink) {
		if (!sink->IsSink()) {
			throw InternalException("Sink of pipeline does not have IsSink set");
		}
		lock_guard<mutex> guard(sink->lock);
		if (!sink->sink_state) {
			throw InternalException("Sink of pipeline does not have sink state");
		}
		sink->PrepareFinalize(GetClientContext(), *sink->sink_state);
	}
}

void Pipeline::Reset() {
	ResetSink();
	for (auto &op_ref : operators) {
		auto &op = op_ref.get();
		lock_guard<mutex> guard(op.lock);
		if (!op.op_state) {
			op.op_state = op.GetGlobalOperatorState(GetClientContext());
		}
	}
	ResetSource(false);
	// we no longer reset source here because this function is no longer guaranteed to be called by the main thread
	// source reset needs to be called by the main thread because resetting a source may call into clients like R
	initialized = true;
}

void Pipeline::ResetSource(bool force) {
	if (source && !source->IsSource()) {
		throw InternalException("Source of pipeline does not have IsSource set");
	}
	if (force || !source_state) {
		source_state = source->GetGlobalSourceState(GetClientContext());
	}
}

void Pipeline::Ready() {
	if (ready) {
		return;
	}
	ready = true;
	std::reverse(operators.begin(), operators.end());

	/* LIP *******************************************************************/

	// Probe pipelines have a non-hash-join source
	if (!source || source->type == PhysicalOperatorType::HASH_JOIN) {
		// Hash join sources are only used for external joins, which can be
		//  ignored for LIP
		// TODO: verify above
		return;
	}

	// std::cout << "processing pipeline:\n" << ToString();

	// TODO: on failing return, go and clear all the bloom filters

	// Validate that everything in the pipeline is a hash join allowing LIP
	for (auto &op : operators) {
		switch (op.get().type) {
		case PhysicalOperatorType::HASH_JOIN: {
			// TODO: can we do LIP but ignore joins that don't support it?
			if (!op.get().Cast<PhysicalHashJoin>().join_supports_lip) {
				return;
			}
			break;
		}
		case PhysicalOperatorType::PROJECTION:
		case PhysicalOperatorType::EXPLAIN:
		case PhysicalOperatorType::EXPLAIN_ANALYZE:
		case PhysicalOperatorType::STREAMING_LIMIT:
		case PhysicalOperatorType::FILTER: {
			break;
		}
		default: {
			// TODO: are other operators ok as well?
			return;
		}
		}
	}

	// Map "index at join to probe" -> "real index at source"
	unordered_map<idx_t, idx_t> bf_probe_idxs;
	for (idx_t i = 0; i < source->types.size(); i++) {
		// std::cout << source->types[i].ToString() << ", ";
		bf_probe_idxs.emplace(i, i);
	}
	// std::cout << '\n';

	// Notify all joins that LIP should be run, and create BF pointers
	for (auto &op : operators) {
		// std::cout << '{';
		// for (auto &map : bf_probe_idxs) {
		// 	std::cout << map.first << " -> " << map.second << ", ";
		// }
		// std::cout << "}\n";

		switch (op.get().type) {
		// TODO: what other operators modify the output order?
		case PhysicalOperatorType::PROJECTION: {
			auto &proj = op.get().Cast<PhysicalProjection>();
			unordered_map<idx_t, idx_t> new_bf_probe_idxs;

			// Update output order
			for (idx_t out_idx = 0; out_idx < proj.select_list.size(); out_idx++) {
				// position in list is the output idx, value is input idx
				// TODO: other types of exprs?
				if (proj.select_list[out_idx]->type != ExpressionType::BOUND_REF) {
					continue;
				}

				auto &bound_ref = proj.select_list[out_idx].get()->Cast<BoundReferenceExpression>();
				auto it = bf_probe_idxs.find(bound_ref.index);
				if (it != bf_probe_idxs.end()) {
					new_bf_probe_idxs.emplace(out_idx, it->second);
				}
			}

			// TODO: need to clear bf_probe_idxs?
			bf_probe_idxs = new_bf_probe_idxs;
			break;
		}
		case PhysicalOperatorType::HASH_JOIN: {
			auto &join = op.get().Cast<PhysicalHashJoin>();
			unordered_map<idx_t, idx_t> new_bf_probe_idxs;
			D_ASSERT(join.join_supports_lip);
			D_ASSERT(join.bf_build.empty());
			join.pipeline_supports_lip = true;

			// Make bloom filters for all valid probe indices
			for (auto &info : join.bf_probe_build) {
				auto it = bf_probe_idxs.find(info.first);
				if (it != bf_probe_idxs.end()) {
					auto bf = make_shared_ptr<BloomFilter>();

					join.bf_build.emplace_back(info.second, bf);
					bf_probe.emplace_back(it->second, bf);
				}
			}

			// Update output order for non-join-keys
			idx_t out_idx;
			for (out_idx = 0; out_idx < join.lhs_output_columns.col_idxs.size(); out_idx++) {
				// position in list is the output idx, value is input idx
				auto it = bf_probe_idxs.find(join.lhs_output_columns.col_idxs[out_idx]);
				if (it != bf_probe_idxs.end()) {
					new_bf_probe_idxs.emplace(out_idx, it->second);
				}
			}

			// Update output order for join keys
			for (idx_t sel_idx = 0; sel_idx < join.conditions.size(); sel_idx++) {
				if (join.conditions[sel_idx].left->type != ExpressionType::BOUND_REF) {
					continue;
				}

				auto &bound_ref = join.conditions[sel_idx].left->Cast<BoundReferenceExpression>();
				auto it = bf_probe_idxs.find(bound_ref.index);
				if (it != bf_probe_idxs.end()) {
					new_bf_probe_idxs.emplace(out_idx + sel_idx, it->second);
				}
			}

			bf_probe_idxs = new_bf_probe_idxs;
			break;
		}
		default: {
			break;
		}
		}
	}

	// for (auto &info : bf_probe) {
	// 	std::cout << "probe idx: " << info.first << ", bf: " << info.second.get() << '\n';
	// }
}

void Pipeline::AddDependency(shared_ptr<Pipeline> &pipeline) {
	D_ASSERT(pipeline);
	dependencies.push_back(weak_ptr<Pipeline>(pipeline));
	pipeline->parents.push_back(weak_ptr<Pipeline>(shared_from_this()));
}

string Pipeline::ToString() const {
	TextTreeRenderer renderer;
	return renderer.ToString(*this);
}

void Pipeline::Print() const {
	Printer::Print(ToString());
}

void Pipeline::PrintDependencies() const {
	for (auto &dep : dependencies) {
		shared_ptr<Pipeline>(dep)->Print();
	}
}

vector<reference<PhysicalOperator>> Pipeline::GetOperators() {
	vector<reference<PhysicalOperator>> result;
	D_ASSERT(source);
	result.push_back(*source);
	for (auto &op : operators) {
		result.push_back(op.get());
	}
	if (sink) {
		result.push_back(*sink);
	}
	return result;
}

vector<const_reference<PhysicalOperator>> Pipeline::GetOperators() const {
	vector<const_reference<PhysicalOperator>> result;
	D_ASSERT(source);
	result.push_back(*source);
	for (auto &op : operators) {
		result.push_back(op.get());
	}
	if (sink) {
		result.push_back(*sink);
	}
	return result;
}

const vector<reference<PhysicalOperator>> &Pipeline::GetIntermediateOperators() const {
	return operators;
}

void Pipeline::ClearSource() {
	source_state.reset();
	batch_indexes.clear();
}

idx_t Pipeline::RegisterNewBatchIndex() {
	lock_guard<mutex> l(batch_lock);
	idx_t minimum = batch_indexes.empty() ? base_batch_index : *batch_indexes.begin();
	batch_indexes.insert(minimum);
	return minimum;
}

idx_t Pipeline::UpdateBatchIndex(idx_t old_index, idx_t new_index) {
	lock_guard<mutex> l(batch_lock);
	if (new_index < *batch_indexes.begin()) {
		throw InternalException("Processing batch index %llu, but previous min batch index was %llu", new_index,
		                        *batch_indexes.begin());
	}
	auto entry = batch_indexes.find(old_index);
	if (entry == batch_indexes.end()) {
		throw InternalException("Batch index %llu was not found in set of active batch indexes", old_index);
	}
	batch_indexes.erase(entry);
	batch_indexes.insert(new_index);
	return *batch_indexes.begin();
}
//===--------------------------------------------------------------------===//
// Pipeline Build State
//===--------------------------------------------------------------------===//
void PipelineBuildState::SetPipelineSource(Pipeline &pipeline, PhysicalOperator &op) {
	pipeline.source = &op;
}

void PipelineBuildState::SetPipelineSink(Pipeline &pipeline, optional_ptr<PhysicalOperator> op,
                                         idx_t sink_pipeline_count) {
	pipeline.sink = op;
	// set the base batch index of this pipeline based on how many other pipelines have this node as their sink
	pipeline.base_batch_index = BATCH_INCREMENT * sink_pipeline_count;
}

void PipelineBuildState::AddPipelineOperator(Pipeline &pipeline, PhysicalOperator &op) {
	pipeline.operators.push_back(op);
}

optional_ptr<PhysicalOperator> PipelineBuildState::GetPipelineSource(Pipeline &pipeline) {
	return pipeline.source;
}

optional_ptr<PhysicalOperator> PipelineBuildState::GetPipelineSink(Pipeline &pipeline) {
	return pipeline.sink;
}

void PipelineBuildState::SetPipelineOperators(Pipeline &pipeline, vector<reference<PhysicalOperator>> operators) {
	pipeline.operators = std::move(operators);
}

shared_ptr<Pipeline> PipelineBuildState::CreateChildPipeline(Executor &executor, Pipeline &pipeline,
                                                             PhysicalOperator &op) {
	return executor.CreateChildPipeline(pipeline, op);
}

vector<reference<PhysicalOperator>> PipelineBuildState::GetPipelineOperators(Pipeline &pipeline) {
	return pipeline.operators;
}

} // namespace duckdb
