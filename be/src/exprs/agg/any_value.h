// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <limits>
#include <type_traits>

#include "column/fixed_length_column.h"
#include "column/type_traits.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/aggregate_traits.h"
#include "gutil/casts.h"
#include "util/raw_container.h"

namespace starrocks {

/**
 * ANY_VALUE 的语义很简单，在每个 group 中选择一行返回。
 * 中间结果通过 AnyValueAggregateData 描述，只需要记录当前是否已经有结果以及对应的数据是什么即可，
 * AnyValueAggregateData 为每种数据类型进行了特化，实现上几乎一致
 *
 * 具体的计算逻辑非常简单，这部分通过 AnyValueElement 实现
 * 利用 AnyValueElement 实现 AggregateFunction 所需要的接口即可
 *
 * @tparam LT
 */
template <LogicalType LT>
struct AnyValueAggregateData {
    using T = AggDataValueType<LT>;

    T result;
    bool has_value = false;

    void reset() {
        result = T{};
        has_value = false;
    }
};

template <LogicalType LT, typename State>
struct AnyValueElement {
    using RefType = AggDataRefType<LT>;

    void operator()(State& state, RefType right) const {
        if (UNLIKELY(!state.has_value)) {
            AggDataTypeTraits<LT>::assign_value(state.result, right);
            state.has_value = true;
        }
    }
};

/**
 * any_value 聚合函数实现
 * @tparam LT
 * @tparam State AnyValueAggregateData<LT>
 * @tparam OP AnyValueElement<LT, AnyValueAggregateData<LT>>
 * @tparam T
 */
template <LogicalType LT, typename State, class OP, typename T = RunTimeCppType<LT>, typename = guard::Guard>
class AnyValueAggregateFunction final
        : public AggregateFunctionBatchHelper<State, AnyValueAggregateFunction<LT, State, OP, T>> {
public:
    using InputColumnType = RunTimeColumnType<LT>;

    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override {
        // this->data(state) 函数调用会强制转换成 State类型 => AnyValueAggregateData<LT>
        // .reset() 实际上是 AnyValueAggregateData.reset
        this->data(state).reset();
    }

    // 逐行读取数据，不断更新 state 中保存的中间结果。
    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        DCHECK(!columns[0]->is_nullable());
        const auto& column = down_cast<const InputColumnType&>(*columns[0]);
        // get_row_ref 获取第 row_num 数据， 返回类型是 AggDataRefType<LT>
        OP()(this->data(state), AggDataTypeTraits<LT>::get_row_ref(column, row_num));
    }

    void update_batch_single_state(FunctionContext* ctx, size_t chunk_size, const Column** columns,
                                   AggDataPtr __restrict state) const override {
        update(ctx, columns, state, 0);
    }

    // 通常用在多阶段聚合中，读取已经算好的部分中间结果，合并计算，更新 state 中的数据。
    void merge(FunctionContext* ctx, const Column* column, AggDataPtr __restrict state, size_t row_num) const override {
        DCHECK(!column->is_nullable());
        const auto& input_column = down_cast<const InputColumnType&>(*column);
        OP()(this->data(state), AggDataTypeTraits<LT>::get_row_ref(input_column, row_num));
    }

    // 多阶段的聚合可能会通过多个节点执行，计算的中间结果需要跨网络传输，这个方法用来实现序列化的逻辑。
    // Column->append(state)
    void serialize_to_column(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* to) const override {
        DCHECK(!to->is_nullable());
        AggDataTypeTraits<LT>::append_value(down_cast<InputColumnType*>(to), this->data(state).result);
    }

    // For streaming aggregation, we directly convert column data to serialize format
    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {
        *dst = src[0];
    }

    // 把中间结果转成最终对用户返回的结果。比如求和函数，直接返回中间结果保存的 sum 即可，而平均值函数，需要返回 sum/count。
    void finalize_to_column(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* to) const override {
        DCHECK(!to->is_nullable());
        AggDataTypeTraits<LT>::append_value(down_cast<InputColumnType*>(to), this->data(state).result);
    }

    // Insert current aggregation state into dst column from start to end
    // For aggregation window functions
    void get_values(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* dst, size_t start,
                    size_t end) const override {
        DCHECK_GT(end, start);
        InputColumnType* column = down_cast<InputColumnType*>(dst);
        for (size_t i = start; i < end; ++i) {
            AggDataTypeTraits<LT>::append_value(column, this->data(state).result);
        }
    }

    std::string get_name() const override { return "any_value"; }
};

} // namespace starrocks
