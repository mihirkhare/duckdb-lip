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

namespace {
using duckdb::vector;
using duckdb::pair;
using duckdb::unordered_map;

using duckdb::make_shared_ptr;
using duckdb::make_pair;

using duckdb::Pipeline;

using duckdb::PhysicalOperatorType;
using duckdb::JoinType;
using duckdb::ExpressionType;
using duckdb::ExpressionClass;
using duckdb::BoundReferenceExpression;
using duckdb::PhysicalProjection;

using duckdb::JoinCondition;

using duckdb::PhysicalOperator;
using duckdb::PhysicalHashJoin;
using duckdb::PhysicalTableScan;

using duckdb::LIPBloomFilter;

class LIPPlanInfo {
public:
	LIPPlanInfo() = delete;

	explicit LIPPlanInfo(Pipeline *lip_pipeline) : lip_pipeline(lip_pipeline) {}

	bool IsLIPSupported() {
		// std::cout << "Validating pipeline:\n" << lip_pipeline->ToString();

		// Probe pipelines have a non-hash-join source
		if (!NonHashJoinSource()) {
			return false;
		}

		// std::cout << "- Valid source\n";

		// All joins must support LIP
		if (!ValidateAndBuildAllJoins()) {
			return false;
		}

		// std::cout << "- Valid joins\n";

		return true;
	}

	void LinkLIPJoins() {
		for (auto &join_data : lip_data) {
			auto &cols = join_data.first;
			auto &build_cols = cols.first;
			auto &probe_cols = cols.second;
			auto &join = join_data.second;

			auto lip_info = make_shared_ptr<LIPBloomFilter>(build_cols, probe_cols);
			join->lip_filter = lip_info;
			lip_pipeline->lip_filters.push_back(lip_info);
		}
	}

private:
	Pipeline *lip_pipeline;
	//! vector of ((build cols, probe cols), building join)
	vector<pair<pair<vector<idx_t>, vector<idx_t>>, PhysicalHashJoin *>> lip_data;

	bool NonHashJoinSource() const {
		// Hash join sources are only used for external joins, which can be
		//  ignored for LIP
		// TODO: verify above
		auto source = lip_pipeline->GetSource();
		return source && source->type != PhysicalOperatorType::HASH_JOIN;
	}

	static bool IsSelective(PhysicalOperator &op) {
		switch (op.type) {
		case PhysicalOperatorType::ORDER_BY:
			break;
		case PhysicalOperatorType::LIMIT:
			break;
		case PhysicalOperatorType::STREAMING_LIMIT:
			break;
		case PhysicalOperatorType::LIMIT_PERCENT:
			break;
		case PhysicalOperatorType::TOP_N:
			break;
		case PhysicalOperatorType::WINDOW:
			break;
		case PhysicalOperatorType::UNNEST:
			break;
		case PhysicalOperatorType::UNGROUPED_AGGREGATE:
			break;
		case PhysicalOperatorType::HASH_GROUP_BY:
			break;
		case PhysicalOperatorType::PERFECT_HASH_GROUP_BY:
			break;
		case PhysicalOperatorType::PARTITIONED_AGGREGATE:
			break;
		case PhysicalOperatorType::FILTER:
			break;
		case PhysicalOperatorType::PROJECTION: {
			return IsSelective(op.children[0]);
		}
		case PhysicalOperatorType::RESERVOIR_SAMPLE:
			break;
		case PhysicalOperatorType::STREAMING_SAMPLE:
			break;
		case PhysicalOperatorType::STREAMING_WINDOW:
			break;
		case PhysicalOperatorType::TABLE_SCAN: {
			if (!op.Cast<PhysicalTableScan>().table_filters) {
				return false;
			}
			break;
		}
		case PhysicalOperatorType::COLUMN_DATA_SCAN:
			break;
		case PhysicalOperatorType::CHUNK_SCAN:
			break;
		case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
			break;
		case PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN:
			break;
		case PhysicalOperatorType::CTE_SCAN:
			break;
		case PhysicalOperatorType::EXPRESSION_SCAN:
			break;
		case PhysicalOperatorType::POSITIONAL_SCAN:
			break;
		case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
			break;
		case PhysicalOperatorType::NESTED_LOOP_JOIN:
			break;
		case PhysicalOperatorType::HASH_JOIN:
			break;
		case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
			break;
		case PhysicalOperatorType::IE_JOIN:
			break;
		case PhysicalOperatorType::LEFT_DELIM_JOIN:
			break;
		case PhysicalOperatorType::RIGHT_DELIM_JOIN:
			break;
		case PhysicalOperatorType::POSITIONAL_JOIN:
			break;
		case PhysicalOperatorType::ASOF_JOIN:
			break;
		default: {
			return false;
		}
		}

		return true;
	}

	void BuildJoinConditions(PhysicalHashJoin *join, const unordered_map<idx_t, idx_t> &column_bindings) {
		vector<idx_t> build_cols;
		vector<idx_t> probe_cols;

		for (auto &cond : join->conditions) {
			// LIP only works on equi-joins
			if (cond.comparison != ExpressionType::COMPARE_EQUAL) {
				continue;
			}

			// LIP currently only works on directly bound references
			// TODO: can LIP work on arbitrary/other expressions?
			if (cond.left->GetExpressionClass() != ExpressionClass::BOUND_REF &&
				cond.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
				continue;
			}

			idx_t left_idx = cond.left->Cast<BoundReferenceExpression>().index;
			idx_t right_idx = cond.right->Cast<BoundReferenceExpression>().index;

			// Check that the left-hand side references the source
			auto it = column_bindings.find(left_idx);
			if (it == column_bindings.end()) {
				continue;
			}

			probe_cols.push_back(it->second);
			build_cols.push_back(right_idx);

			// std::cout << "working condition: " << condition.left->ToString() << ' ' << ExpressionTypeToString(condition.comparison) << ' ' << condition.right->ToString() << '\n';
			//
			// std::cout << "candidate probe: " << condition.left->Cast<BoundReferenceExpression>().index << '\n';
			// std::cout << "candidate build: " << condition.right->Cast<BoundReferenceExpression>().index << '\n';
		}

		// Only register as valid for LIP if at least one join condition is valid
		if (!build_cols.empty()) {
			auto cols = make_pair(build_cols, probe_cols);
			lip_data.emplace_back(cols, join);
		}
	}

	bool ValidateAndBuildJoin(PhysicalHashJoin *join, const unordered_map<idx_t, idx_t> &column_bindings) {
		// Does the join type allow us to perform LIP?
		switch (join->join_type) {
		case JoinType::INNER:
		case JoinType::RIGHT:
			break;
		default:
			return false;
		}

		// Is the build side valid for LIP?
		if (!IsSelective(join->children[1].get())) {
			// Can still do LIP on later joins, so should return true
			return true;
		}

		BuildJoinConditions(join, column_bindings);

		return true;
	}

	static void UpdateColumnBindings(PhysicalOperator &op, unordered_map<idx_t, idx_t> &column_bindings) {
		switch (op.type) {
		case PhysicalOperatorType::HASH_JOIN: {
			auto &join = op.Cast<PhysicalHashJoin>();
			unordered_map<idx_t, idx_t> new_column_bindings;

			// Maps RHS input index to equivalent LHS input
			unordered_map<idx_t, idx_t> equiv_cols;
			for (auto &cond : join.conditions) {
				if (cond.comparison != ExpressionType::COMPARE_EQUAL) {
					continue;
				}

				// TODO: can LIP work on arbitrary/other expressions?
				if (cond.left->GetExpressionClass() != ExpressionClass::BOUND_REF &&
					cond.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
					continue;
				}

				idx_t left_idx = cond.left->Cast<BoundReferenceExpression>().index;
				idx_t right_idx = cond.right->Cast<BoundReferenceExpression>().index;

				if (column_bindings.find(left_idx) == column_bindings.end()) {
					continue;
				}

				equiv_cols.emplace(right_idx, left_idx);
			}

			// Update output order for non-join-keys
			idx_t left_idx;
			for (left_idx = 0; left_idx < join.lhs_output_columns.col_idxs.size(); left_idx++) {
				// position in list is the output idx, value is input idx
				auto it = column_bindings.find(join.lhs_output_columns.col_idxs[left_idx]);
				if (it != column_bindings.end()) {
					new_column_bindings.emplace(left_idx, it->second);
				}
			}

			// Update output order for join keys
			for (idx_t right_idx = 0; right_idx < join.rhs_output_columns.col_idxs.size(); right_idx++) {
				auto it = equiv_cols.find(join.rhs_output_columns.col_idxs[right_idx]);
				if (it != equiv_cols.end()) {
					new_column_bindings.emplace(left_idx + right_idx, it->second);
				}
			}

			column_bindings = new_column_bindings;
			break;
		}
		case PhysicalOperatorType::PROJECTION: {
			auto &proj = op.Cast<PhysicalProjection>();
			unordered_map<idx_t, idx_t> new_column_bindings;

			// Update output order
			for (idx_t out_idx = 0; out_idx < proj.select_list.size(); out_idx++) {
				// position in list is the output idx, value is input idx
				// TODO: other types of exprs?
				if (proj.select_list[out_idx]->type != ExpressionType::BOUND_REF) {
					continue;
				}

				auto &bound_ref = proj.select_list[out_idx].get()->Cast<BoundReferenceExpression>();
				auto it = column_bindings.find(bound_ref.index);
				if (it != column_bindings.end()) {
					new_column_bindings.emplace(out_idx, it->second);
				}
			}

			column_bindings = new_column_bindings;
			break;
		}
		default: {
			// TODO: assert false?
		}
		}
	}

	void GetInitialColumnBindings(unordered_map<idx_t, idx_t> &column_bindings) const {
		for (idx_t i = 0; i < lip_pipeline->GetSource()->types.size(); i++) {
			column_bindings.emplace(i, i);
		}
	}

	// static void PrintColumnBindings(const unordered_map<idx_t, idx_t> &column_bindings) {
	// 	std::cout << "curr column bindings:\n";
	// 	for (auto &kv : column_bindings) {
	// 		std::cout << "- ." << kv.first << " -> " << kv.second << '\n';
	// 	}
	// }

	bool ValidateAndBuildAllJoins() {
		// Map index at current operator to index at source
		unordered_map<idx_t, idx_t> column_bindings;
		GetInitialColumnBindings(column_bindings);

		// Validate that everything in the pipeline is a hash join allowing LIP
		auto operators = lip_pipeline->GetIntermediateOperators();
		for (auto &op : operators) {
			// std::cout << "Looking at operator of type " << duckdb::PhysicalOperatorToString(op.get().type) << '\n';
			switch (op.get().type) {
			case PhysicalOperatorType::HASH_JOIN: {
				if (!ValidateAndBuildJoin(&op.get().Cast<PhysicalHashJoin>(), column_bindings)) {
					// std::cout << "Invalid join:\n" << op.get().ToString();
					return false;
				}
				UpdateColumnBindings(op.get(), column_bindings);
				break;
			}
			// These operators are fine to be in a LIP pipeline
			case PhysicalOperatorType::PROJECTION: {
				UpdateColumnBindings(op.get(), column_bindings);
				break;
			}
			case PhysicalOperatorType::EXPLAIN:
			case PhysicalOperatorType::EXPLAIN_ANALYZE:
			case PhysicalOperatorType::STREAMING_LIMIT:
			case PhysicalOperatorType::FILTER: {
				break;
			}
			default: {
				return false;
			}
			}
		}

		// At least one join must have a valid condition
		if (lip_data.empty()) {
			return false;
		}

		return true;
	}
};
}

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

	// Check if LIP is applicable, and compile linkage info if so
	LIPPlanInfo info(this);
	if (!info.IsLIPSupported()) {
		return;
	}

	// Link pipeline and hash joins with shared bloom filters
	info.LinkLIPJoins();
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
