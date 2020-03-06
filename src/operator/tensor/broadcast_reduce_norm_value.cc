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

/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_norm_value.cc
 * \brief CPU Implementation of broadcast and reduce norm functions based on value.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(NormParam);

bool MKLDNNLpNormCompute(const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs,
                         int ord) {
  auto& in_data = inputs[0];
  int axis =  in_data.shape_.ndim() - 1;                
  auto& shape = in_data.shape_;
  float* pin_data = (float*)in_data.dptr<float>();

  const size_t last_dim = shape[axis];
  auto stride = outputs[0].Size();
  float* pout_data = (float*)outputs[0].dptr<float>();

  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (size_t i = 0; i < stride; i++)
  {
      pout_data[i] = 0;
  }

  #pragma omp parallel for collapse(2) num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for(size_t i = 0; i < stride; i++) {
    for (size_t j = 0; j < last_dim; j++) {
      pout_data[i] += std::abs(pin_data[i*last_dim + j]);
    }
  }
  return true;                  
}

bool MKLDNNLpNormGradCompute(const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs,
                             int ord) {
                           
  auto&  in_grad = inputs[0];
  auto&  in_data = inputs[1];
  auto shape = in_grad.shape_;
  float* pin_grad = (float*)in_grad.dptr<float>();
  float* psrc_data = (float*)in_data.dptr<float>();

  auto&  out_grad = outputs[0];
  int axis =  out_grad.shape_.ndim() - 1;                
  const size_t last_dim = out_grad.shape_[axis];
  auto oSize = out_grad.Size();
  float* pout_grad = (float*)out_grad.dptr<float>();

  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for(size_t i = 0; i < oSize; i++) {
      int idx = i/last_dim;
      pout_grad[i] = psrc_data[i] >0 ? pin_grad[idx] : -pin_grad[idx];
  }

  // #pragma omp parallel for collapse(2) num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  // for(size_t i = 0; i < stride; i++) {
  //   for (size_t j = 0; j < last_dim; j++) {
  //     pout_grad[i*last_dim + j] = psrc_data[i*last_dim + j] >0 ? pin_grad[i] : -pin_grad[i];
  //   }
  // }
  return true;                  
}

template<>
void L2NormComputeEx<cpu>(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  struct timeval start,stop;
  gettimeofday(&start,NULL);

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const NormParam& param = nnvm::get<NormParam>(attrs.parsed);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const NDArrayStorageType istype = inputs[0].storage_type();
  const mxnet::TShape axis = param.axis.has_value() ? param.axis.value() : mxnet::TShape(0, -1);
  if ((istype == kRowSparseStorage || istype == kCSRStorage) && axis.ndim() == 0 &&
       param.ord == 2) {
    // l2 norm on the entire array
    L2NormComputeSparseImpl<cpu>(s, inputs[0], req[0], outputs[0].data());
  } else if (istype == kCSRStorage && axis.ndim() == 1 && (axis[0] == 0 || axis[0] == 1) &&
             !param.keepdims && param.ord == 2) {
    // l2 norm on a particular axis
    NDArray output = outputs[0];
    ReduceCsrImpl<cpu, sq_sum, false>(s, ctx, inputs[0], req[0], &output, axis);
    CHECK_EQ(outputs[0].storage_type(), kDefaultStorage);
    SqRootForL2<cpu>(ctx, req[0], outputs[0].data());
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
   gettimeofday(&stop,NULL);
   LOG(INFO)<<inputs[0].shape()<< " oshape "<< outputs[0].shape()<< "L2Ex cost time  ms "<<(stop.tv_sec-start.tv_sec)*1000+(stop.tv_usec-start.tv_usec)/1000.0 ;

}

NNVM_REGISTER_OP(norm)
MXNET_ADD_SPARSE_OP_ALIAS(norm)
.describe(R"code(Computes the norm on an NDArray.

This operator computes the norm on an NDArray with the specified axis, depending
on the value of the ord parameter. By default, it computes the L2 norm on the entire
array. Currently only ord=2 supports sparse ndarrays.

Examples::

  x = [[[1, 2],
        [3, 4]],
       [[2, 2],
        [5, 6]]]

  norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
                            [5.3851647 6.3245554]]

  norm(x, ord=1, axis=1) = [[4., 6.],
                            [7., 8.]]

  rsp = x.cast_storage('row_sparse')

  norm(rsp) = [5.47722578]

  csr = x.cast_storage('csr')

  norm(csr) = [5.47722578]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NormParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NormShape)
.set_attr<nnvm::FInferType>("FInferType", NormType)
.set_attr<FInferStorageType>("FInferStorageType", LpNormStorageType)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{ "_backward_norm" })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", LpNormCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", L2NormComputeEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "The input")
.add_arguments(NormParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_norm)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NormParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", LpNormGradCompute<cpu>);


}  // namespace op
}  // namespace mxnet
