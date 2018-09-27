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
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
#include <mkl.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sys/time.h>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../linalg.h"
#include "../../common/utils.h"
#include "../rnn_impl.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpResource {kTempSpace};
enum FullyConnectedOpOutputs {kOut};
}  // fullc

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool flatten;
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");
  }
};

template<typename xpu, typename DType>
void FCForward(const OpContext &ctx, const FullyConnectedParam &param,
               const std::vector<TBlob> &in_data, const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[fullc::kOut] == kNullOp) return;
  CHECK_EQ(req[fullc::kOut], kWriteTo);
  // TODO(bing): check the BLAS Handle, be careful
  // maybe need blas handle from context
  // TODO(bing): judge shape to remove flatten op
  Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
  CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
  const TShape& ishape = in_data[fullc::kData].shape_;
  const TShape& oshape = out_data[fullc::kOut].shape_;

  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
  Tensor<xpu, 2, DType> data, out;
  if (!param.flatten) {
    data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
    out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
  } else {
    data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
  }

  CHECK_EQ(data.shape_[1], wmat.shape_[1])
    << "Incomplete weight tensor detected: weight.data().shape[1] != prod(data.data().shape[1:])."
       " This is not supported by FCForward. If weight is in row_sparse format,"
       " please make sure all row ids are present.";
  // Legacy approach shown here for comparison:
  //   out = dot(data, wmat.T());
  linalg_gemm(data, wmat, out, false, true, s);
  if (!param.no_bias) {
    Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get_with_shape<xpu, 1, DType>(
      Shape1(wmat.shape_[0]), s);
    CHECK_EQ(bias.shape_[0], wmat.shape_[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";
    out += repmat(bias, data.size(0));
  }
}


//data uint8
template<typename xpu, typename DType>
void FCForward_int8(const OpContext &ctx, const FullyConnectedParam &param,
               const std::vector<TBlob> &in_data, const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data, bool bCalTime, bool bCache, FILE* pFC, long* fc_mkl_time, long* fc_q_time, long* fc_dq_time, long* fc_gemm_time, long* fc_gemm_call, long* fc_max_time, long* fc_scale_time, long* fc_sum_time, long* fc_copyoffset_time, MKL_UINT8* data_int8, MKL_INT8* wmat_int8, MKL_INT32* wmat_sum_int8, MKL_INT32* out_int8) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[fullc::kOut] == kNullOp) return;
  CHECK_EQ(req[fullc::kOut], kWriteTo);
  // TODO(bing): check the BLAS Handle, be careful
  // maybe need blas handle from context
  // TODO(bing): judge shape to remove flatten op
  Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
  CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
  const TShape& ishape = in_data[fullc::kData].shape_;
  const TShape& oshape = out_data[fullc::kOut].shape_;

  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
  Tensor<xpu, 2, DType> data, out;
  if (!param.flatten) {
    data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
    out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
  } else {
    data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
  }

  CHECK_EQ(data.shape_[1], wmat.shape_[1])
    << "Incomplete weight tensor detected: weight.data().shape[1] != prod(data.data().shape[1:])."
       " This is not supported by FCForward. If weight is in row_sparse format,"
       " please make sure all row ids are present.";
  // Legacy approach shown here for comparison:
  //   out = dot(data, wmat.T());

  struct timeval start, end;
  long costtime;  
  if(bCalTime) {
    gettimeofday(&start, NULL );
    //  LOG(INFO) << "start.tv_sec:" << start.tv_sec << " start.tv_usec:" << start.tv_usec;
  }
/*
  MKL_INT8* data_int8 = reinterpret_cast<MKL_INT8* >
      (mkl_calloc(data.shape_[0] * data.shape_[1], sizeof(MKL_INT8), 64));
*/
//  MKL_INT8* wmat2_int8 = NULL;
// reinterpret_cast<MKL_INT8* >
//      (mkl_calloc(wmat.shape_[0] * wmat.shape_[1], sizeof(MKL_INT8), 64));

/*
//  size_t sum_size = wmat.shape_[0];
//  MKL_INT32* wmat_sum_int8 = reinterpret_cast<MKL_INT32* >
//      (mkl_calloc(sum_size, sizeof(MKL_INT32), 64));
  MKL_INT32* out_int8 = reinterpret_cast<MKL_INT32* >
  (mkl_calloc(out.shape_[0] * out.shape_[1], sizeof(MKL_INT32), 64));
*/
  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasTrans;
  CBLAS_LAYOUT layout = CblasRowMajor;
  MKL_INT m = (MKL_INT)data.shape_[0];
  MKL_INT n = (MKL_INT)wmat.shape_[0];
  MKL_INT k = (MKL_INT)wmat.shape_[2];
  MKL_INT lda = 0, ldb = 0, ldc = 0;
  lda = k;
  ldb = k;
  ldc = n;
  DType alpha = 1.0;
  DType beta = 1.0;
  MKL_INT  ao = 0, bo = 0;
  MKL_INT co = 0;

  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_mkl_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_mkl_time:" << (float)(*fc_mkl_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
/*
  float factor_lr = quantilize(data.dptr_, wmat.dptr_, reinterpret_cast<int>(m),
      reinterpret_cast<int>(n), reinterpret_cast<int>(k),
  data_int8, wmat_int8, wmat_sum_int8, true, true);
*/
/*
  float factor_lr = quantilize(data.dptr_, wmat.dptr_, reinterpret_cast<int>(m),
      reinterpret_cast<int>(n), reinterpret_cast<int>(k),
  data_int8, wmat_int8, wmat_sum_int8, out_int8, true, true);
*/

  float factor_r = 0.0f;
  char buf[1024];


  if (!bCache && pFC != NULL) {
    fgets(buf,sizeof(buf),pFC);
    factor_r = atof(buf);
  }
  if (factor_r <= 0.0f) {
    factor_r = 127 / getmax(wmat.dptr_, reinterpret_cast<int>(k) * reinterpret_cast<int>(n));
  }
  if (bCache) {
    sprintf(buf, "%f\r\n", factor_r);
    fputs(buf, pFC);
  }



// get detailed time
  float factor_l = 63 / getmax(data.dptr_, reinterpret_cast<int>(m) * reinterpret_cast<int>(k));
//  factor_r = 127 / getmax(wmat.dptr_, reinterpret_cast<int>(k) * reinterpret_cast<int>(n));
  
  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_max_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_max_time:" << (float)(*fc_max_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
  scale_data(data.dptr_, reinterpret_cast<int>(m) * reinterpret_cast<int>(k), factor_l, data_int8, 128);
  scale_data(wmat.dptr_, reinterpret_cast<int>(k) * reinterpret_cast<int>(n), factor_r, wmat_int8, 0);
/*
  LOG(INFO) << "correct wmat_int8[1]:" << (int)wmat_int8[1];
  char* addr = wmat_int8;
  LOG(INFO) << "get wmat_int8[1] from address:" << (int)addr[1];
*/

/*

  if (!bCache && pFC != NULL) {
    //fgets(buf, 8, pFC);
    fread(buf, 9, 1, pFC);
    wmat2_int8 = (MKL_INT8*)buf;
    //memcpy(wmat2_int8, buf, 8);
  } else {
    wmat2_int8 = reinterpret_cast<MKL_INT8* >
      (mkl_calloc(wmat.shape_[0] * wmat.shape_[1], sizeof(MKL_INT8), 64));
  }
  if (!bCache && pFC == NULL) {
    scale_data(wmat.dptr_, reinterpret_cast<int>(k) * reinterpret_cast<int>(n), factor_r, wmat2_int8, 0);
  }

  if (bCache) {
    scale_data(wmat.dptr_, reinterpret_cast<int>(k) * reinterpret_cast<int>(n), factor_r, wmat2_int8, 0);
    // MKL_INT8* wmat_int8
    char test_address[8];
    //LOG(INFO) << "wmat_int8:" << wmat_int8;
    memcpy(test_address, wmat2_int8, 8);
    sprintf(buf, "%s\r\n", test_address);
    fputs(buf, pFC);

    //sprintf(buf, "%s\r\n", test_address);
    //MKL_INT8* testdata = (MKL_INT8*)buf;
    //LOG(INFO) << "wmat2_int8[0]:" << (int)wmat2_int8[0];
    //LOG(INFO) << "testdata[0]:" << (int)testdata[0];


  }
*/
  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_scale_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_scale_time:" << (float)(*fc_scale_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
  prepare_sum_data(wmat_int8, reinterpret_cast<int>(n), reinterpret_cast<int>(k), out_int8, true, 128);
  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_sum_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_sum_time:" << (float)(*fc_sum_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
  copyoffset(out_int8, m, n);
  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_copyoffset_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_copyoffset_time:" << (float)(*fc_copyoffset_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
  float factor_lr = factor_l * factor_r;
// get detailed time

/*
  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_q_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_q_time:" << (float)(*fc_q_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
*/

/*
  cblas_gemm_s8u8s32(layout, trans_a, trans_b, CblasRowOffset,
    m, n, k, alpha, data_int8, lda, ao, wmat_int8, ldb, bo, beta,
    out_int8, ldc, wmat_sum_int8);
*/
  cblas_gemm_s8u8s32(layout, trans_a, trans_b, CblasFixOffset,
    m, n, k, alpha, data_int8, lda, ao, wmat_int8, ldb, bo, beta,
    out_int8, ldc, &co);

  if(bCalTime) {
    (*fc_gemm_call)++;
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_gemm_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_gemm_time:" << (float)(*fc_gemm_time)/1000 << "ms" << " fc_gemm_call:" << (*fc_gemm_call) << " m:" << m << " n:" << n << " k:" << k;
    gettimeofday(&start, NULL );
  }

  dequantilize(out_int8, out.shape_[0] * out.shape_[1], factor_lr, out.dptr_);

  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_dq_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_dq_time:" << (float)(*fc_dq_time)/1000 << "ms";
    gettimeofday(&start, NULL );
  }
/*
  mkl_free(data_int8);
  mkl_free(wmat_int8);
//  mkl_free(wmat_sum_int8);
  mkl_free(out_int8);
*/

  if(bCalTime) {
    gettimeofday(&end, NULL );
    //  LOG(INFO) << "end.tv_sec:" << end.tv_sec << " end.tv_usec:" << end.tv_usec;
    if (end.tv_sec == start.tv_sec) {
      costtime = end.tv_usec - start.tv_usec;
    } else {
      costtime = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec - start.tv_usec;
    }
    (*fc_mkl_time) += costtime;
    LOG(INFO) << "costtime:" << (float)costtime/1000 << "ms" << " fc_mkl_time:" << (float)(*fc_mkl_time)/1000 << "ms";
  }

  if (!param.no_bias) {
    Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get_with_shape<xpu, 1, DType>(
      Shape1(wmat.shape_[0]), s);
    CHECK_EQ(bias.shape_[0], wmat.shape_[0])
      << "Incomplete bias tensor detected: bias.data().shape[1] != weight.data().shape[0]."
         " This is not supported by FCForward. If bias is in row_sparse format, please"
         " make sure all row ids are present.";
    out += repmat(bias, data.size(0));
  }
}

template<typename xpu, typename DType>
void FCBackward(const OpContext &ctx, const FullyConnectedParam &param,
                const std::vector<TBlob> &out_grad, const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req, const std::vector<TBlob> &in_grad) {
  using namespace mshadow;
  using namespace mshadow::expr;
  // TODO(bing): check the BLAS Handle, be careful
  //  maybe need blas handle from context
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TShape& ishape = in_data[fullc::kData].shape_;
  const TShape& oshape = out_grad[fullc::kOut].shape_;

  Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
  Tensor<xpu, 2, DType> data, grad, gdata;
  if (!param.flatten) {
    data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
    grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
    gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
  } else {
    data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
  }

#if defined(__CUDACC__)
  CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";
#endif
  //  backprop
  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  // gradient of weight
  Tensor<xpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<xpu, 2, DType>(s);
  // Legacy approach shown here for comparison:
  //   out = Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));
  linalg_gemm(grad, data, gwmat, true, false, s, req[fullc::kWeight]);
  // gradient of bias
  if (!param.no_bias) {
    Tensor<xpu, 1, DType> gbias = in_grad[fullc::kBias].get<xpu, 1, DType>(s);
    Assign(gbias, req[fullc::kBias], sum_rows(grad));
  }
  // gradient of data
  // Legacy approach shown here for comparison:
  //   Assign(gdata, req[fullc::kData], dot(grad, wmat));
  linalg_gemm(grad, wmat, gdata, false, false, s, req[fullc::kData]);
}

template<typename xpu>
void FullyConnectedCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), in_expected);
  CHECK_EQ(outputs.size(), 1U);
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCForward<xpu, float>(ctx, param, inputs, req, outputs);
    break;
  case mshadow::kFloat64:
    FCForward<xpu, double>(ctx, param, inputs, req, outputs);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
}

//data uint8
template<typename xpu>
void FullyConnectedCompute_int8(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs, bool bCalTime, bool bCache, FILE* pFC, long* fc_mkl_time, long* fc_q_time, long* fc_dq_time, long* fc_gemm_time, long* fc_gemm_call, long* fc_max_time, long* fc_scale_time, long* fc_sum_time, long* fc_copyoffset_time, MKL_UINT8* data_int8, MKL_INT8* wmat_int8, MKL_INT32* wmat_sum_int8, MKL_INT32* out_int8) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), in_expected);
  CHECK_EQ(outputs.size(), 1U);
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCForward_int8<xpu, float>(ctx, param, inputs, req, outputs, bCalTime, bCache, pFC, fc_mkl_time, fc_q_time, fc_dq_time, fc_gemm_time, fc_gemm_call, fc_max_time, fc_scale_time, fc_sum_time, fc_copyoffset_time, data_int8, wmat_int8, wmat_sum_int8, out_int8);
    break;
  case mshadow::kFloat64:
    FCForward_int8<xpu, double>(ctx, param, inputs, req, outputs, bCalTime, bCache, pFC, fc_mkl_time, fc_q_time, fc_dq_time, fc_gemm_time, fc_gemm_call, fc_max_time, fc_scale_time, fc_sum_time, fc_copyoffset_time, data_int8, wmat_int8, wmat_sum_int8, out_int8);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
}

template<typename xpu>
void FullyConnectedGradCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), out_expected);
  CHECK_EQ(req.size(), out_expected);

  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  int dtype = inputs[0].type_flag_;

  switch (dtype) {
  case mshadow::kFloat32:
    FCBackward<xpu, float>(ctx, param, out_grad, in_data, req, outputs);
    break;
  case mshadow::kFloat64:
    FCBackward<xpu, double>(ctx, param, out_grad, in_data, req, outputs);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_FULLY_CONNECTED_INL_H_
