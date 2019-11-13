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
 * \file mkldnn_layer_norm.cc
 * \brief integrate mkldnn layer norm forward
 */
#if MXNET_USE_MKLDNN == 1
#include "../layer_norm-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
namespace mxnet {
namespace op {

typedef ParamOpSign<LayerNormParam> MKLDNNLayerNormSignature;
const static bool always_output_mean_var(true);
static mkldnn::layer_normalization_forward::primitive_desc GetLayerNormFwdDescImpl(
               const LayerNormParam& param, bool is_train,
               const mkldnn::memory &input_mem) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto flags = mkldnn::normalization_flags::use_scale_shift;
  float epsilon = param.eps;
  bool output_mean_var = param.output_mean_var;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto prop = (is_train || output_mean_var || always_output_mean_var)
                 ? mkldnn::prop_kind::forward_training
                 : mkldnn::prop_kind::forward_inference;
  auto desc = mkldnn::layer_normalization_forward::desc(prop, data_md, epsilon, flags);
  return mkldnn::layer_normalization_forward::primitive_desc(desc, cpu_engine);
}

class MKLDNNLayerNormFwd {
  std::shared_ptr<const mkldnn::memory> weight_m;
  std::shared_ptr<const mkldnn::memory> mean_m;
  std::shared_ptr<const mkldnn::memory> var_m;
  std::shared_ptr<mkldnn::layer_normalization_forward> fwd_;
 public:
  const mkldnn::layer_normalization_forward::primitive_desc fwd_pd;

  MKLDNNLayerNormFwd(const LayerNormParam& param, bool is_train,
                         const int axis, const mkldnn::memory &mem): fwd_pd(
                         GetLayerNormFwdDescImpl(param, is_train, mem)) {
    weight_m.reset(new mkldnn::memory(fwd_pd.weights_desc(), CpuEngine::Get()->get_engine()));
    if (is_train || param.output_mean_var || always_output_mean_var) {
      mean_m.reset(new mkldnn::memory(fwd_pd.mean_desc(), CpuEngine::Get()->get_engine()));
      var_m.reset(new mkldnn::memory(fwd_pd.variance_desc(), CpuEngine::Get()->get_engine()));
    }
    fwd_ = std::make_shared<mkldnn::layer_normalization_forward>(fwd_pd);
  }

  const inline mkldnn::layer_normalization_forward &GetFwd() const {
    return *fwd_;
  }

  const mkldnn::memory &GetWeight() const {
    return *weight_m;
  }
  const mkldnn::memory &GetMean() const {
    return *mean_m;
  }
  const mkldnn::memory &GetVar() const {
    return *var_m;
  }

};

static MKLDNNLayerNormFwd &GetLayerNormForward(const LayerNormParam& param,
                                                       const OpContext &ctx,
                                                       const NDArray &in_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
    std::unordered_map<MKLDNNLayerNormSignature, MKLDNNLayerNormFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL
    std::unordered_map<MKLDNNLayerNormSignature, MKLDNNLayerNormFwd, OpHash> fwds;
#endif
  MKLDNNLayerNormSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(in_data);

  //  mkldnn layer norm always use the last dim as axis.
  int axis = in_data.shape().ndim() - 1;

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto in_mem = *(in_data.GetMKLDNNData());
    MKLDNNLayerNormFwd fwd(param, ctx.is_train, axis, in_mem);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}


void MKLDNNLayerNormForward(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<NDArray> &in_data,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &out_data) {
  const LayerNormParam &param = nnvm::get<LayerNormParam>(attrs.parsed);

  NDArray idata = in_data[layernorm::kData];
  const NDArray &gamma    = in_data[layernorm::kGamma];
  const NDArray &beta     = in_data[layernorm::kBeta];

  NDArray odata = out_data[layernorm::kOut];
  if (in_data[layernorm::kData].IsView() && in_data[layernorm::kData].IsMKLDNNData()) {
    idata = in_data[layernorm::kData].Reorder2Default();
  }
  auto input_mem = idata.GetMKLDNNData();

  MKLDNNLayerNormFwd &fwd = GetLayerNormForward(param, ctx, idata);
  const mkldnn::memory &weight_mem = fwd.GetWeight();
  float* weight_buf = reinterpret_cast<float *>(weight_mem.get_data_handle());
  int ndim = idata.shape().ndim();
  nnvm::dim_t channels_ = idata.shape()[ndim-1];
  CHECK(weight_mem.get_desc().get_size() == channels_ * sizeof(float) * 2);
  float* weight_ptr = gamma.data().dptr<float>();
  float* bias_ptr = beta.data().dptr<float>();
  memcpy(weight_buf, weight_ptr, sizeof(weight_buf[0]) * channels_);
  memcpy(&weight_buf[channels_], bias_ptr, sizeof(weight_buf[0]) * channels_);
  auto fwd_pd = fwd.fwd_pd;
  auto out_mem = CreateMKLDNNMem(out_data[layernorm::kOut],
                                 fwd_pd.dst_desc(), req[layernorm::kOut]);
  
  mkldnn_args_map_t args = {
    {MKLDNN_ARG_SRC, *input_mem},
    {MKLDNN_ARG_SCALE_SHIFT, weight_mem},
    {MKLDNN_ARG_DST, *out_mem.second}
  };
  if(ctx.is_train || param.output_mean_var || always_output_mean_var) {
    args[MKLDNN_ARG_MEAN] = fwd.GetMean();
    args[MKLDNN_ARG_VARIANCE] = fwd.GetVar();
  }

  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), args);
  CommitOutput(out_data[layernorm::kOut], out_mem);
  stream->Submit();

  if(ctx.is_train || param.output_mean_var || always_output_mean_var) {
    mkldnn::stream s(CpuEngine::Get()->get_engine());
    mkldnn::memory::desc from_desc = fwd.GetMean().get_desc();
    mxnet::TShape tshape(from_desc.data.ndims, -1);
    for (int i = 0; i < from_desc.data.ndims; i++) {
      tshape[i] = from_desc.data.dims[i];
    }
    mkldnn_format_tag_t format = GetDefaultFormat(from_desc.data.ndims);
    mkldnn::memory::dims dims(from_desc.data.dims, from_desc.data.dims + from_desc.data.ndims);
    mkldnn::memory::data_type cpp_type =
        static_cast<mkldnn::memory::data_type>(from_desc.data.data_type);
    mkldnn::memory::desc data_md(dims, cpp_type,
        static_cast<mkldnn::memory::format_tag>(format));

    mkldnn::memory def_mean_mem(data_md, CpuEngine::Get()->get_engine(), out_data[layernorm::kMean].GetMKLDNNData()->get_data_handle());
    mkldnn::reorder reorder(fwd.GetMean(), def_mean_mem);
    auto mean_mem = fwd.GetMean();
    reorder.execute(s, mean_mem,  def_mean_mem);

    mkldnn::memory def_var_mem(data_md, CpuEngine::Get()->get_engine(), out_data[layernorm::kStd].GetMKLDNNData()->get_data_handle());
    auto var_mem = fwd.GetVar();
    reorder.execute(s, var_mem,  def_var_mem);

    size_t size = out_data[layernorm::kStd].shape().Size();
    float* pVar =  static_cast<float*>(out_data[layernorm::kStd].GetMKLDNNData()->get_data_handle());
    for (size_t i = 0; i < size; i++)
    {
      pVar[i] = math::sqrt( pVar[i] + param.eps);
    }
    
  }
}
}   // namespace op
}   // namespace mxnet
#endif

