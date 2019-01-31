/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_PARALLEL_OP_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_PARALLEL_OP_INL_H_

#include "common.h"
#include "../../tensor/indexing_op.h"


namespace mxnet {
namespace op {

struct ParallelOpParam {
  int parallel_input_size;
};

template<typename xpu>
void SgParallelOpForward(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<NDArray>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<NDArray>& outputs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  std::unordered_set<std::string> output_node_name;
  for(auto entry : sym.outputs) {
    if(entry.node->op()) {
    output_node_name.insert(entry.node->attrs.name);
    }
  }

  int parallel_op_size = output_node_name.size();
  int input_size = sym.ListInputNames(nnvm::Symbol::kAll).size()/parallel_op_size;
  int output_size = sym.ListOutputNames().size()/parallel_op_size;

  int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  #pragma omp parallel for num_threads(omp_threads)
  for(int i = 0; i < parallel_op_size; i++) {
    const std::vector<NDArray>each_input(inputs.begin()+i*input_size,inputs.begin()+(i+1)*input_size);
    const std::vector<NDArray>each_output(outputs.begin()+i*output_size,outputs.begin()+(i+1)*output_size);
    SparseEmbeddingOpForwardEx<xpu>(attrs,ctx,each_input,req,each_output);
  }
}

// enum MKLDNNConvOpOutputs { kOut, kMin, kMax };

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_INL_H_

