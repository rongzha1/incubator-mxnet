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
static mkldnn::layer_normalization_forward::primitive_desc _GetFwd(
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
                         _GetFwd(param, is_train, mem)) {
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
#if !defined(_MSC_VER)
#pragma omp simd
#endif
    for (size_t i = 0; i < size; i++)
    {
      pVar[i] = math::sqrt( pVar[i] + param.eps);
    }
    
  }
}

class MKLDNNLayerNormBwd {
  std::shared_ptr<mkldnn::layer_normalization_backward> bwd;
  const std::shared_ptr<mkldnn::memory> weight_m;
  const std::shared_ptr<mkldnn::memory> gradw_m;

 public:
  const mkldnn::layer_normalization_backward::primitive_desc bwd_pd;
  const mkldnn::layer_normalization_forward::primitive_desc fwd_pd;

  explicit MKLDNNLayerNormBwd(const mkldnn::layer_normalization_backward::primitive_desc &_bwd_pd,
                              const mkldnn::layer_normalization_forward::primitive_desc &_fwd_pd)
      : weight_m(new mkldnn::memory(_bwd_pd.weights_desc(), CpuEngine::Get()->get_engine())),
        gradw_m(new mkldnn::memory(_bwd_pd.diff_weights_desc(), CpuEngine::Get()->get_engine())),
        bwd_pd(_bwd_pd),
        fwd_pd(_fwd_pd) {
    bwd.reset(new mkldnn::layer_normalization_backward(_bwd_pd));
  }

  const mkldnn::memory &GetWeight() const { return *weight_m; }

  const mkldnn::memory &GetGradw() const { return *gradw_m; }

  const mkldnn::layer_normalization_backward &GetBwd() const { return *bwd; }
};

inline static mkldnn::layer_normalization_backward::primitive_desc _GetBwd(
                                   const LayerNormParam &param,
                                   const mkldnn::memory &data_mem,
                                   const mkldnn::memory &diff_mem,
                                   mkldnn::layer_normalization_forward::primitive_desc& fwd_pd) {
  auto data_md    = data_mem.get_desc();
  auto diff_md    = diff_mem.get_desc();
  auto engine     = CpuEngine::Get()->get_engine();

  mkldnn::layer_normalization_backward::desc  lnBwd_desc(
                                mkldnn::prop_kind::backward,
                                diff_md,
                                data_md,
                                param.eps,
                                mkldnn::normalization_flags::use_scale_shift);
  fwd_pd = _GetFwd(param, true, data_mem);                              
  return mkldnn::layer_normalization_backward::primitive_desc(lnBwd_desc,
                                                              engine,
                                                              fwd_pd);
}

static MKLDNNLayerNormBwd &GetLNBackward(
    const LayerNormParam &param, const OpContext &ctx, const NDArray &in_data,
    const mkldnn::memory &in_mem, const NDArray &diff_data,
    const mkldnn::memory &diff_mem) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNLayerNormSignature, MKLDNNLayerNormBwd, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNLayerNormSignature, MKLDNNLayerNormBwd, OpHash> bwds;
#endif
  MKLDNNLayerNormSignature key(param);
  key.AddSign(in_data);
  key.AddSign(diff_data);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    mkldnn::layer_normalization_forward::primitive_desc fwd_pd;
    auto bwd_pd = _GetBwd(param, in_mem, diff_mem, fwd_pd);
    MKLDNNLayerNormBwd bwd(bwd_pd, fwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNLayerNormBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
  const LayerNormParam &param = nnvm::get<LayerNormParam>(attrs.parsed);
  // inputs order: ograd , data, gamma, mean, var
  size_t offset = 0;
  const NDArray &diff = inputs[offset++];
  const NDArray &data = inputs[offset++];
  const NDArray &gamma = inputs[offset++];
  const NDArray &mean = inputs[offset++];
  const NDArray &var = inputs[offset++];

  offset = 0;
  const NDArray &grad_data = outputs[offset++];
  const NDArray &grad_gamma = outputs[offset++];
  const NDArray &grad_beta = outputs[offset++];
  LOG(INFO) <<" diff shape is "<<diff.shape() <<" data "<<data.shape()<<" gamma "<< gamma.shape()
  <<" mean "<<mean.shape()<< " var " << var.shape()<<" grad_data "<<grad_data.shape()<<" grad_gamma "<<grad_gamma.shape();
  auto data_mem  = data.GetMKLDNNData();
  auto diff_mem  = diff.GetMKLDNNData();
#if 0
  float* p1 = (float*)diff.GetMKLDNNData()->get_data_handle();
  float* p2 = (float*)data.GetMKLDNNData()->get_data_handle();
  float* p3 = (float*)gamma.GetMKLDNNData()->get_data_handle();
  float* p4 = (float*)mean.GetMKLDNNData()->get_data_handle();
  float* p5 = (float*)var.GetMKLDNNData()->get_data_handle();

  for (size_t i = 0; i < 10; i++)
  {
    LOG(INFO) << i<<" data " <<p2[i];
    LOG(INFO) << i<<" Ograd " <<p1[i];
  }
  
  for (size_t i = 0; i < 5; i++)
  {
    LOG(INFO) << i<<" gamma " <<p3[i];
    LOG(INFO) << i<<" mean " <<p4[i];
    LOG(INFO) << i<<" var " <<p5[i];
  }
#endif  
  // MKLDNN batchnorm should run on special layouts. If one of them isn't, we
  // should reorder them.
  // if (data.IsDefaultData())
  //   data_mem = data.GetMKLDNNDataReorder(diff_mem->get_desc());
  // else if (diff.IsDefaultData())
  //   diff_mem = diff.GetMKLDNNDataReorder(data_mem->get_desc());

  auto &bwd = GetLNBackward(param, ctx, data, *data_mem, diff, *diff_mem);
//  auto gradi_mem = const_cast<NDArray &>(grad_data).CreateMKLDNNData(data_mem->get_desc());

  
  float *weight_buf = reinterpret_cast<float *>(bwd.GetWeight().get_data_handle());
  nnvm::dim_t channels_ = gamma.shape()[0];
  for (int i = 0; i < channels_; i++) {
    weight_buf[i] = (gamma.data().dptr<float>())[i];   // weight
  }
  for (int i = 0; i < channels_; i++) {
    weight_buf[channels_ + i] = 0;  // no input bias in layer norm
  }

  size_t size = var.shape().Size();
  float* pVar =  static_cast<float*>(var.GetMKLDNNData()->get_data_handle());
    for (size_t i = 0; i < size; i++)
    {
      pVar[i] = pVar[i]*pVar[i] - param.eps;
    }
  // auto mean_mem = const_cast<NDArray &>(mean).CreateMKLDNNData(bwd.fwd_pd.mean_desc());
  // auto var_mem = const_cast<NDArray &>(var).CreateMKLDNNData(bwd.fwd_pd.variance_desc());

    mkldnn::stream s(CpuEngine::Get()->get_engine());
    
    mkldnn::memory def_mean_mem(bwd.fwd_pd.mean_desc(), CpuEngine::Get()->get_engine());
    mxnet::TShape tshape(mean.shape().ndim()-1, -1);
    for (int i = 0; i < mean.shape().ndim() - 1; i++) {
      tshape[i] = mean.shape()[i];
    }

    // auto mean_mem = mean.MKLDNNDataReshape(tshape);
    auto mean_mem = const_cast<mkldnn::memory*>(mean.MKLDNNDataReshape(tshape).GetMKLDNNData());
    mkldnn::reorder reorder(*mean_mem, def_mean_mem);
    reorder.execute(s, *mean_mem,  def_mean_mem);

    mkldnn::memory def_var_mem(bwd.fwd_pd.variance_desc(), CpuEngine::Get()->get_engine());
    auto var_mem =  const_cast<mkldnn::memory*>(var.MKLDNNDataReshape(tshape).GetMKLDNNData());
    reorder.execute(s, *var_mem,  def_var_mem);

#if 0
  float *pW = (float*)(bwd.GetWeight().get_data_handle());
for (size_t i = 0; i < channels_*2; i++)
{
LOG(INFO) <<i<<" out weight is  "<< pW[i];
}
#endif

  auto gradi_mem = CreateMKLDNNMem(outputs[layernorm::kOut],
                                 bwd.bwd_pd.diff_src_desc(), req[layernorm::kOut]);

  mkldnn_args_map_t net_args;
  net_args[MKLDNN_ARG_SRC] = *data_mem;
  net_args[MKLDNN_ARG_DIFF_SRC] = *gradi_mem.second;
  net_args[MKLDNN_ARG_SCALE_SHIFT] = bwd.GetWeight();
  net_args[MKLDNN_ARG_DIFF_SCALE_SHIFT] = bwd.GetGradw();
  net_args[MKLDNN_ARG_DIFF_DST] = *diff_mem;
  net_args[MKLDNN_ARG_MEAN] = def_mean_mem;
  net_args[MKLDNN_ARG_VARIANCE] = def_var_mem;
  // training but no input mean and variance

  MKLDNNStream::Get()->RegisterPrimArgs(bwd.GetBwd(), net_args);
  CommitOutput(outputs[layernorm::kOut], gradi_mem);
  MKLDNNStream::Get()->Submit();

  // copy data from gradw_mem to output[1] and output[2]
  float *gw_buf = reinterpret_cast<float *>(bwd.GetGradw().get_data_handle());
  if(req[layernorm::kGamma] == kAddTo) {
  for (int i = 0; i < channels_; i++) {
    (grad_gamma.data().dptr<float>())[i] += gw_buf[i];
//    LOG(INFO) <<i<< " gamma addto "<< (grad_gamma.data().dptr<float>())[i];
  }
  } else {
  for (int i = 0; i < channels_; i++) {
    (grad_gamma.data().dptr<float>())[i] = gw_buf[i];
//    LOG(INFO) <<i<< " gamma "<< (grad_gamma.data().dptr<float>())[i];
  }
  }

  if(req[layernorm::kGamma] == kAddTo) {
  for (int i = 0; i < channels_; i++) {
    (grad_beta.data().dptr<float>())[i] += gw_buf[i + channels_];
//    LOG(INFO) <<i<< " beta "<< (grad_beta.data().dptr<float>())[i];
  }
  } else {
  for (int i = 0; i < channels_; i++) {
    (grad_beta.data().dptr<float>())[i] = gw_buf[i + channels_];
//    LOG(INFO) <<i<< " beta "<< (grad_beta.data().dptr<float>())[i];
  }
  }
#if 0
float *pOut_data = (float*)outputs[0].GetMKLDNNData()->get_data_handle();
for (size_t i = 0; i < 10; i++)
{
LOG(INFO) <<i<<" out_data "<< pOut_data[i];
}
#endif
}

}   // namespace op
}   // namespace mxnet
#endif

