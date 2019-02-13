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
  
    if ((status != kEnd && status != kSelect) || (candidates.size() < 2)) {
      return std::vector<nnvm::Node *>(0);
    } else {
        //TODO:add condition function to decided whether to parallel
    std::vector<nnvm::Node *> ret;
    for (auto i : matched_list) {
      auto non_const_i = const_cast<nnvm::Node *>(i);
      if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
        candidates.end()) {
      ret.push_back(non_const_i);
      }
    }
    return ret;
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
                                           const SubgraphSelectorPtr& subgraph_selector,
                                           const int subgraph_id = 0) const {
    LOG(INFO)<<"CreateSubgraphNode sym size " << sym.outputs.size();
    //output node for parallel op
    auto last_node = sym.outputs[0].node;
    std::string op_name = last_node->op()->name;
    int parallel_op_num = sym.outputs.size();
  
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.name = "sg_parallel_" + op_name + std::to_string(subgraph_id);
    n->attrs.op = Op::Get("SgParallel_op");
    CHECK(n->attrs.op); 
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
  //    n->op()->attr_parser(&(n->attrs));
#if 0
//    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    uint32_t e_idx = 0;
    for (int i=0; i < last_node->inputs.size(); i++) {
        nnvm::NodeEntry& entry = last_node->inputs[i];
        if (entry.node->op() && entry.node->op()->name == op_name) {
      LOG(INFO)<<"parallel embedding op inputs number  "<<n->inputs.size();
      for(int j = 0; j < entry.node->inputs.size(); j++) {
              nnvm::NodeEntry& parallel_op_Entry = entry.node->inputs[j];
        LOG(INFO)<<"add entry "<<parallel_op_Entry.node->attrs.name;
        n->inputs.emplace_back(parallel_op_Entry);
      }
            last_node->inputs[i] = nnvm::NodeEntry{ n, e_idx, 0};
            ++e_idx;
        }
    }
#endif
    return n;
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
  LOG(INFO)<<"ConnectSubgraphOutputs " <<n->attrs.name;
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
    LOG(INFO)<<"output_entries size " <<output_entries->size();
    LOG(INFO)<<"output_entries name " <<(*output_entries)[i]->node->attrs.name;
      *output_entries->at(i) = nnvm::NodeEntry{n, i, 0};
    }
  }

  void ConnectSubgraphInputs(
      const nnvm::NodePtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
    LOG(INFO)<<"ConnectSubgraphInputs "<<n->op()->name;

    for (int i = 0; i < input_entries->size(); i++) {
    nnvm::NodeEntry *pEntry = (*input_entries)[i];
    LOG(INFO) << "input_entries name " << pEntry->node->attrs.name;
  }
    for (int i = 0; i < orig_input_entries->size(); i++) {
    nnvm::NodeEntry &origin_entry = (*orig_input_entries)[i];
    LOG(INFO) << "origin_entry name " << origin_entry.node->attrs.name;
  }
  
  for(auto entry : *orig_input_entries) {
    n->inputs.emplace_back(entry);  
  }

#if 0
  
  std::string parallel_op_name;
    
  for (auto entry : n->inputs) {    
      if(entry.node->op()) {
        std::string op_name = entry.node->op()->name;
    if(op_name != n->op()->name) {
      parallel_op_name = op_name;
    }
    }
  }
  

    std::ostringstream node_name;
    node_name << "sg_parallel_";
    node_name << parallel_op_name;
    nnvm::NodePtr parallel_n = nnvm::Node::Create();
    parallel_n->attrs.name = node_name.str();
    parallel_n->attrs.op = Op::Get("SgParallel_op");

    uint32_t e_idx = 0;
  int origin_idx = 0;
    for (int i=0; i < n->inputs.size(); i++) {
      nnvm::NodeEntry& entry = n->inputs[i];
        if (entry.node->op() && entry.node->op()->name == parallel_op_name) {
          for(int j = 0; j < entry.node->inputs.size(); j++) {
      parallel_n->inputs.emplace_back((*orig_input_entries)[origin_idx++]);
      }
      n->inputs[i] = nnvm::NodeEntry{ parallel_n, e_idx, 0};
    } else {
          n->inputs[i] = (*orig_input_entries)[origin_idx++];
    }
    }
  for (int i = 0; i < n->inputs.size(); i++) {
    nnvm::NodePtr& n_input = n->inputs[i].node;
    LOG(INFO) << "inputs name " << n_input->attrs.name;
  if(n->inputs[i].node->op()) {
    LOG(INFO)<<"   its op name is "<<n->inputs[i].node->op()->name;

    for(auto input_node_entry : n->inputs[i].node->inputs) {
      LOG(INFO)<<"      and its op entry name "<<input_node_entry.node->attrs.name;
    }
      
  }
  }
#endif
  
  }

 private:
  int disable_all;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(PARALLEL_OP, SgParallelOpProperty);

}  // namespace op
}  // namespace mxnet


