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

#if MXNET_USE_MKLDNN == 1
#include <sstream>
#include "../common.h"
#include "../subgraph_property.h"
#include "../../nn/fully_connected-inl.h"
#include "../../nn/activation-inl.h"
#include "../../tensor/mkldnn/mkldnn_parallel_embedding.h"

namespace mxnet {
namespace op {

#define EMBEDDING_NODE_NAME "Embedding"

class SgMKLDNNParallelEmbeddingSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kEmbedding,
    kSuccess,
  };

 private:
  bool disable_all;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;

 public:
  explicit SgMKLDNNParallelEmbeddingSelector(int dis_all)
      : disable_all(dis_all) {}

  bool Select(const nnvm::Node &n) override {
    if ((!disable_all) && n.op() && n.op()->name == "Concat") {
      status = kStart;
      matched_list.clear();
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
      if (disable_all) return false;
      if (status == kFail || status == kSuccess || new_node.is_variable())
          return false;
      bool ret = false;
      switch (status) {
      case kStart:
          //The Assumption is only base on W&D which all embedding occur at the beginning and output to 1 concat node
          if (new_node.op()->name == EMBEDDING_NODE_NAME) {
              matched_list.push_back(&new_node);
              status = kEmbedding; // > 2 embedding
              ret = true;
          }
          else{
              return false;
          }

          break;
      case kEmbedding:
          if (new_node.op()->name == EMBEDDING_NODE_NAME) {
              matched_list.push_back(&new_node);
              ret = true;
          }
          else {
              status = kSuccess;
              return false;
          }

          break;
      default:
      {
          status = kSuccess;
          break;
      }
      }
      if (!ret) {
          while (matched_list.back() != &n) {
              matched_list.pop_back();
          }
          status = kSuccess;
      }
      return ret;

  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
      return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status != kSuccess) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }
};
template <typename T>
static std::string int_vector_to_attr(T v) {
    std::stringstream ss;
    ss << "[";
    int i = 0;
    for (; i < v.size()-1; i++) {
        ss << v[i] << ",";        
    }
    ss << v[i];
    ss << "]";
    return ss.str();
}
class SgMKLDNNParallelEmbeddingProperty : public SubgraphProperty {
private:
 public:
  SgMKLDNNParallelEmbeddingProperty() {
    disable_all = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_OPT", 0);
    if (disable_all) {
      LOG(INFO) << "MKLDNN Parallel Embedding is disabled.";
    } else {
      LOG(INFO) << "Start to execute MKLDNN Parallel Embedding optimization pass.";
    }       
  }
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgMKLDNNParallelEmbeddingProperty>();
  }

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const SubgraphSelectorPtr& subgraph_selector,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr pe = nnvm::Node::Create();
    std::vector<nnvm::NodePtr> emb_nodes;
    nnvm::NodePtr concat_node = nullptr;

    DFSVisit(sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &op_name = node->op()->name;
      //The Assumption is only base on W&D which all embedding occur at the beginning and output to 1 concat node
      if (op_name == EMBEDDING_NODE_NAME) {
          emb_nodes.push_back(node);
      }
      else if (emb_nodes.size() != 0 && concat_node == nullptr)
      {
          if (op_name != "Concat") {
              std::cout << "!!Parallel Embedding Node following: " << op_name << std::endl;
          }
          concat_node = node;
      }      
    });
    CHECK_NOTNULL(emb_nodes.size() != 0);
    pe->attrs.name = "ParallelEmbedding_0";
    pe->attrs.op = Op::Get("ParallelEmbedding");

    CHECK(pe->attrs.op);

    std::vector<nnvm::NodePtr>::iterator it;
    //Assumption:  subgraph use DFS
    std::vector<int> v_in_dims; 
    std::vector<int> v_out_dims;
    std::vector<int> v_types;
    std::vector<bool> v_sparse_grads;
    for (it = emb_nodes.begin(); it != emb_nodes.end(); it++) {
        nnvm::NodePtr em_node = *it;
        const EmbeddingParam &param = nnvm::get<EmbeddingParam>(em_node->attrs.parsed);
        v_in_dims.push_back(param.input_dim);
        v_out_dims.push_back(param.output_dim);
        v_types.push_back(param.dtype);
        v_sparse_grads.push_back(param.sparse_grad);
    }

    pe->attrs.dict["input_dims"] = int_vector_to_attr<std::vector<int> >(v_in_dims);
    pe->attrs.dict["output_dims"] = int_vector_to_attr<std::vector<int> >(v_out_dims);
    pe->attrs.dict["dtypes"] = int_vector_to_attr<std::vector<int> >(v_types);
    pe->attrs.dict["num_args"] = std::to_string(emb_nodes.size());
    pe->attrs.dict["sparse_grads"] = int_vector_to_attr<std::vector<bool> >(v_sparse_grads);
    pe->op()->attr_parser(&(pe->attrs));
    uint32_t e_idx = 0;
    for (int i=0; i < concat_node->inputs.size(); i++) {
        nnvm::NodeEntry& entry = concat_node->inputs[i];
        if (entry.node->op() && entry.node->op()->name == EMBEDDING_NODE_NAME) {
            concat_node->inputs[i] = nnvm::NodeEntry{ pe, e_idx, 0};
            ++e_idx;
        }
    }
    return concat_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector =
        std::make_shared<SgMKLDNNParallelEmbeddingSelector>(disable_all);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }
  void ConnectSubgraphInputs(
      const nnvm::NodePtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
      nnvm::NodePtr para_embdding = nullptr;
      std::vector<int> concat_non_embedding_idxs;
      for (int i = 0; i < n->inputs.size(); i++) {
          nnvm::NodePtr& n_input = n->inputs[i].node;
          std::string op_name = "";
          if (n_input->op()) op_name = n_input->op()->name;
          if(!para_embdding && op_name == "ParallelEmbedding") {
              para_embdding = n->inputs[i].node;
          }
          else {
              concat_non_embedding_idxs.push_back(i);
          }
      }
      CHECK_NOTNULL(para_embdding);
      int non_embedding_idx = 0;
      uint32_t slice_channel_idx = 0;
      for (int i = 0; i < orig_input_entries->size(); i++) {
          nnvm::NodeEntry &entry = (*orig_input_entries)[i];
          std::string entry_name = "";
          if (entry.node->op()) entry_name = entry.node->op()->name;
          if (entry_name != "slice" && entry_name != "SliceChannel") {
              para_embdding->inputs.push_back(nnvm::NodeEntry{ entry.node, 0, 0 });
          }
          else if (entry_name == "SliceChannel") {
              para_embdding->inputs.push_back(nnvm::NodeEntry{ entry.node, slice_channel_idx++, 0 });
          }
          else { //Slice
              n->inputs[concat_non_embedding_idxs[non_embedding_idx++]] = nnvm::NodeEntry{ entry.node, 0, 0 };
          }
      }
  }
 private:
  int disable_all;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_PARALLEL_EMBEDDING, SgMKLDNNParallelEmbeddingProperty);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
