#include "parallel_op-inl.h"

namespace mxnet {
namespace op {

class SgParallelOperator {
 public:
  explicit SgParallelOperator(const nnvm::NodeAttrs &attrs)
      : initalized_(false),
        subgraph_sym_(*attrs.subgraphs[0]),
        param_(nnvm::get<ParallelOpParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph Parallel OP only supports "
                  "inference computation.";
  }

 private:
  bool initalized_;
  nnvm::Symbol subgraph_sym_;
  ParallelOpParam param_;
  /*
  std::shared_ptr<MKLDNNConvForward> fwd_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  float cached_data_min_;
  float cached_data_max_;
  float cached_sum_min_;
  float cached_sum_max_;
  size_t weight_ver_;
  size_t bias_ver_;
  std::vector<float> weight_scales_;
  bool inplace_;
  */
};

void SgParallelOperator::Forward(const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {

                   
}
/*
static void SgParallelOpForward(const OpStatePtr &state_ptr,
                                  const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
  SgParallelOperator &op = state_ptr.get_state<SgParallelOperator>();
  op.Forward(ctx, inputs, req, outputs);
}
*/


static void SgParallelOpParamParser(nnvm::NodeAttrs *attrs) {
}

NNVM_REGISTER_OP(SgParallel_op)
.describe(R"code(SgParallel_op)code" ADD_FILELINE)
.set_num_inputs(DefaultSubgraphOpNumInputs)
.set_num_outputs(DefaultSubgraphOpNumOutputs)
.set_attr_parser(SgParallelOpParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames", DefaultSubgraphOpListInputs)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", DefaultSubgraphOpListOutputs)
//.set_attr<nnvm::FListOutputNames>("FListOutputNames", DefaultSubgraphOpListOutputs)
// .set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNConvState)
.set_attr<nnvm::FInferShape>("FInferShape", DefaultSubgraphOpShape)
.set_attr<nnvm::FInferType>("FInferType", DefaultSubgraphOpType)
.set_attr<FInferStorageType>("FInferStorageType", DefaultSubgraphOpStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SgParallelOpForward<cpu>)
//.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgParallelOpForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args");
//.set_attr<nnvm::FInplaceOption>("FInplaceOption", SgMKLDNNConvInplaceOption)
//.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNConvQuantizedOp)
//.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
//.set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgMKLDNNAvoidQuantizeInput);

}  // namespace op
}  // namespace mxnet

