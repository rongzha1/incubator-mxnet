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

#include "common.h"
#include "subgraph_property.h"


namespace mxnet {
namespace op {
class SgParallelOpSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
  kSelect,
    kEnd,
  };

 private:
  bool disable_all;
  SelectStatus status;
  std::vector<const nnvm::Node *> matched_list;
  std::string sleceted_op_name;
  static std::unordered_set<std::string> parallel_op_whitelist;
 public:
  SgParallelOpSelector(int dis_all)
      : disable_all(dis_all) {}

  bool Select(const nnvm::Node &n) override {
  LOG(INFO) << "Coming in function Select for node: "<<n.attrs.name;
    if((!disable_all) && n.op()) {
      status = kStart;
      matched_list.clear();
      return true;
    }
    return false;

  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    LOG(INFO) << "SelectInput n"<<n.attrs.name<<" new_node "<< new_node.attrs.name << " status "<<status;
    if (new_node.is_variable()) {
      return false;
    } else {
      bool select_new_node = false;
    switch (status) {
      case kStart:
      {
      std::unordered_set<std::string>::const_iterator got = 
        parallel_op_whitelist.find (new_node.op()->name);
      if(got != parallel_op_whitelist.end()) {
              status = kSelect;
        select_new_node = true;
        sleceted_op_name = new_node.op()->name;
        matched_list.push_back(&new_node);
      }
      }
      break;
      case kSelect:
      if(new_node.op()->name == sleceted_op_name) {
        select_new_node = true;
        matched_list.push_back(&new_node);
      }
      break;
      case kEnd:
      break;
    default:
          status = kFail;
      return false;
    }
    return select_new_node; 
  }  
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
  LOG(INFO) << "SelectOutput n"<<n.attrs.name<<" new_node "<< new_node.attrs.name;
  if(status == kSelect) {
    status = kEnd;
  }   
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
      
  LOG(INFO) << "Filter candidates size "<<candidates.size() ;
  
    if (status != kEnd) {
      return std::vector<nnvm::Node *>(0);
    } else {
        //TODO:add condition function to decided whether to parallel
    return candidates;
    }
  }
};

std::unordered_set<std::string> SgParallelOpSelector::
  parallel_op_whitelist = {"Embedding"};

class SgParallelOpProperty : public SubgraphProperty {
 public:
  SgParallelOpProperty() {
    disable_all = dmlc::GetEnv("MXNET_DISABLE_PARALLEL_OP_ALL", 0);

    if (disable_all) {
      LOG(INFO) << "Parallel OP optimization pass is disabled.";
    } else {
      LOG(INFO) << "Start to execute Parallel OP optimization pass.";
    }
  }
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgParallelOpProperty>();
  }
  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
  LOG(INFO)<<"CreateSubgraphNode sym size " << sym.outputs.size();
  //output node for parallel op
    auto last_node = sym.outputs[0].node;
  std::string last_node_name = last_node->op()->name;
  std::string parallel_op_name;
  int parallel_op_num = 0;
    DFSVisit(sym.outputs, [&](const nnvm::NodePtr &node) {
    auto &op_name = node->attrs.name;
    LOG(INFO)<<" CreateSubgraphNode sym outputs name "<<op_name;
      if (node->is_variable()) return;
    if(node->op()->name != last_node_name) {
        parallel_op_name = node->op()->name;
    parallel_op_num ++;
    }
    });
  
    nnvm::NodePtr n = nnvm::Node::Create();
    std::ostringstream node_name;
    node_name << "sg_parallel_";
  node_name << parallel_op_name << std::to_string(subgraph_id);
  
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("SgParallel_op");
    CHECK(n->attrs.op);
//    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
//    n->op()->attr_parser(&(n->attrs));
    uint32_t e_idx = 0;
    for (int i=0; i < last_node->inputs.size(); i++) {
        nnvm::NodeEntry& entry = last_node->inputs[i];
        if (entry.node->op() && entry.node->op()->name == parallel_op_name) {
            last_node->inputs[i] = nnvm::NodeEntry{ n, e_idx, 0};
            ++e_idx;
        }
    }
      
    return last_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgParallelOpSelector>(
        disable_all);
  
    LOG(INFO)<<"CreateSubgraphSelector " ;
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      *output_entries->at(i) = nnvm::NodeEntry{n, 0, 0};
    }
    LOG(INFO)<<"ConnectSubgraphOutputs " <<n->attrs.name;
  }

  void ConnectSubgraphInputs(
      const nnvm::NodePtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
  LOG(INFO)<<"ConnectSubgraphInputs "<<n->op()->name;

  for (int i = 0; i < n->inputs.size(); i++) {
    nnvm::NodePtr& n_input = n->inputs[i].node;
    LOG(INFO) << "inputs name " << n_input->attrs.name;
  }

    for (int i = 0; i < input_entries->size(); i++) {
//      nnvm::NodeEntry *pEntry = (*input_entries)[i];
//    LOG(INFO) << "input_entries name " << pEntry->node->attrs.name;
  }
    for (int i = 0; i < orig_input_entries->size(); i++) {
      nnvm::NodeEntry &origin_entry = (*orig_input_entries)[i];
    LOG(INFO) << "origin_entry name " << origin_entry.node->attrs.name;
  }
  }

 private:
  int disable_all;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(PARALLEL_OP, SgParallelOpProperty);

}  // namespace op
}  // namespace mxnet

