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
 * \file mkldnn_convolution.cc
 * \brief
 * \author Da Zheng; Rong Zhang (rong.a.zhang@intel.com)
*/
#if MXNET_USE_MKLDNN == 1

#include "../convolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

static mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const ConvolutionParam& param, bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
  }

  mkldnn::memory::dims dilates{0, 0};
  if (param.dilate.ndim() == 2) {
    dilates[0] = param.dilate[0] - 1;
    dilates[1] = param.dilate[1] - 1;
  }
  return GetConvFwdPd(param.dilate.ndim(), bias, is_train, data_md, weight_md, out_md
    , strides, dilates, padding);
}


static mkldnn::convolution_backward_data::primitive_desc GetConvBwdData(
    const ConvolutionParam& param, const NDArray &data, const NDArray &weights,
    const NDArray &output, const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
  }
  if (param.dilate.ndim() == 0) {
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    }
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, dilates, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
  }
}

static mkldnn::convolution_backward_weights::primitive_desc GetConvBwdWeights(
    const ConvolutionParam& param, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
  }

  if (param.dilate.ndim() == 0) {
    if (bias == nullptr) {
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
          data_md, weight_md, bias_md, out_md, strides, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    }
  } else {
    mkldnn::memory::dims dilates{0, 0};
    if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    }
    if (bias == nullptr) {
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
                                                      data_md, weight_md, bias_md, out_md,
                                                      strides, dilates, padding, padding,
                                                      mkldnn::padding_kind::zero);
      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    }
  }
}

class MKLDNNConvForward {
  std::shared_ptr<mkldnn::convolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const ConvolutionParam& param, bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvFwdImpl(param, is_train, data, weights, bias, output)) {
  }

  ~MKLDNNConvForward() {}

  void SetDataHandle(const nnvm::NodeAttrs &attrs,
      const OpContext &ctx, const std::vector<NDArray> &in_data,
      const std::vector<OpReqType> &req, const std::vector<NDArray> &out_data);

  void Execute();
};  // class MKLDNNConvForward

void MKLDNNConvForward::SetDataHandle(const nnvm::NodeAttrs &attrs,
    const OpContext &ctx, const std::vector<NDArray> &in_data,
    const std::vector<OpReqType> &req, const std::vector<NDArray> &out_data) {

    const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
    auto data_mem = in_data[conv::kData].GetMKLDNNDataReorder(this->fwd_pd.src_primitive_desc());
    const mkldnn::memory *weight_mem;
    if (ctx.is_train) {
      // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
      // to the default format for now.
      if (in_data[conv::kWeight].IsMKLDNN()) {
        const_cast<NDArray &>(in_data[conv::kWeight]).Reorder2Default();
      }
      weight_mem = GetWeights(in_data[conv::kWeight], this->fwd_pd.weights_primitive_desc(),
                              param.num_group);
    } else {
      // For inference, we want to reorder the weight array so we don't need to
      // reorder data every time.
      const_cast<NDArray &>(in_data[conv::kWeight]).Reorder(
          this->fwd_pd.weights_primitive_desc());
      weight_mem = in_data[conv::kWeight].GetMKLDNNData();
    }
    auto out_mem = CreateMKLDNNMem(out_data[conv::kOut], this->fwd_pd.dst_primitive_desc(),
                                   req[conv::kOut]);
    const mkldnn::memory *bias_mem = nullptr;
    auto output = out_mem.second;
    if (this->data == nullptr) {
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.src_primitive_desc(), data_mem->get_data_handle()));
    } else {
      this->data->set_data_handle(data_mem->get_data_handle());
    }

    if (this->weight == nullptr) {
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.weights_primitive_desc(), weight_mem->get_data_handle()));
    } else {
      this->weight->set_data_handle(weight_mem->get_data_handle());
    }

    if (this->out == nullptr) {
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output->get_data_handle()));
    } else {
      this->out->set_data_handle(output->get_data_handle());
    }

    if (!param.no_bias) {
      bias_mem = in_data[conv::kBias].GetMKLDNNDataReorder(this->fwd_pd.bias_primitive_desc());
      if (this->bias == nullptr) {
        this->bias = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                fwd_pd.bias_primitive_desc(), bias_mem->get_data_handle()));
      } else {
        this->bias->set_data_handle(bias_mem->get_data_handle());
      }
      if (this->fwd == nullptr) {
        this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
            new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                            mkldnn::primitive::at(*this->weight),
                                            mkldnn::primitive::at(*this->bias),
                                            *this->out));
      }
    } else {
      if (this->fwd == nullptr) {
        this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
          new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                          mkldnn::primitive::at(*this->weight),
                                          *this->out));
      }
    }

    CommitOutput(out_data[conv::kOut], out_mem);
}

void MKLDNNConvForward::Execute() {
    MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
    MKLDNNStream::Get()->Submit();
}

typedef MKLDNNParamOpSign<ConvolutionParam> MKLDNNConvSignature;

static inline MKLDNNConvForward &GetConvFwd(
    const nnvm::NodeAttrs& attrs, bool is_train,
    const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, MKLDNNOpHash> fwds;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MKLDNNConvSignature key(param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConvForward fwd(param, is_train, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}


void MKLDNNConvolutionForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray> &in_data,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);

  MKLDNNConvForward &fwd = GetConvFwd(attrs,
      ctx.is_train, in_data[conv::kData], in_data[conv::kWeight],
      param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);
  fwd.SetDataHandle(attrs, ctx, in_data, req, out_data);
  fwd.Execute();
}

void MKLDNNConvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwdImpl(param, ctx.is_train,
      inputs[conv::kData + 1], inputs[conv::kWeight + 1],
      param.no_bias ? nullptr : &inputs[conv::kBias + 1], inputs[conv::kOut]);

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  mkldnn::convolution_backward_data::primitive_desc bwdData_pd
    = GetConvBwdData(param, inputs[conv::kData + 1], inputs[conv::kWeight + 1],
        inputs[conv::kOut], fwd_pd);
  auto out_grad_mem = inputs[conv::kOut].GetMKLDNNDataReorder(
      bwdData_pd.diff_dst_primitive_desc());
  if (req[conv::kData]) {
    auto weight_mem = GetWeights(inputs[conv::kWeight + 1], bwdData_pd.weights_primitive_desc(),
                                 param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[conv::kData],
        bwdData_pd.diff_src_primitive_desc(), req[conv::kData]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_data(bwdData_pd,
          *out_grad_mem, *weight_mem, *in_grad_mem.second));
    CommitOutput(in_grad[conv::kData], in_grad_mem);
  }
  if (req[conv::kWeight]) {
    mkldnn::convolution_backward_weights::primitive_desc bwdWeights_pd
        = GetConvBwdWeights(param, inputs[conv::kData + 1], inputs[conv::kWeight + 1],
                            param.no_bias ? nullptr : &inputs[conv::kBias + 1],
                            inputs[conv::kOut], fwd_pd);
    if (bwdData_pd.diff_dst_primitive_desc() != bwdWeights_pd.diff_dst_primitive_desc())
      out_grad_mem = inputs[conv::kOut].GetMKLDNNDataReorder(
          bwdWeights_pd.diff_dst_primitive_desc());
    auto data_mem = inputs[conv::kData + 1].GetMKLDNNDataReorder(
        bwdWeights_pd.src_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(in_grad[conv::kWeight],
                                                 bwdWeights_pd.diff_weights_primitive_desc(),
                                                 req[conv::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_weights(
              bwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second));
    } else {
      in_grad_bias = CreateMKLDNNMem(in_grad[conv::kBias],
                                     bwdWeights_pd.diff_bias_primitive_desc(),
                                     req[conv::kBias]);
      MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_weights(
              bwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second,
              *in_grad_bias.second));
      CommitOutput(in_grad[conv::kBias], in_grad_bias);
    }
    CommitOutput(in_grad[conv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1



