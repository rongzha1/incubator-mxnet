#ifndef __MKLDNN_UTILS_H__
#define __MKLDNN_UTILS_H__

#include <mkldnn.hpp>

static inline void print_dims(mkldnn::memory::desc desc, const std::string name = "") {
  printf("%s: ", name.c_str());
  for (int i = 0; i < desc.data.ndims; i++)
    printf("%d, ", desc.data.dims[i]);
  printf("\n");
}

static inline void print_dims(mkldnn::memory::primitive_desc pd, const std::string name = "") {
  print_dims(pd.desc(), name);
}

static inline void print_dims(const mkldnn::memory &mem, const std::string name = "") {
  print_dims(mem.get_primitive_desc(), name);
}

static inline void print_dims(const nnvm::TShape &shape, const std::string name = "") {
  printf("%s: ", name.c_str());
  for (size_t i = 0; i < shape.ndim(); i++)
    printf("%ld, ", shape[i]);
  printf("\n");
}

static inline void print_dims(const mxnet::NDArray &arr, const std::string name = "") {
  print_dims(arr.shape(), name);
}

static inline void print(const mxnet::NDArray &arr, const std::string name = "") {
  print_dims(arr, name);
  float *data = (float *) arr.data().dptr_;
  for (size_t i = 0; i < arr.shape().Size(); i++)
    printf("%f, ", data[i]);
  printf("\n");
}

static inline void print_diff(const mxnet::NDArray &arr1, const mxnet::NDArray &arr2) {
  print_dims(arr1);
  float *data1 = (float *) arr1.data().dptr_;
  float *data2 = (float *) arr2.data().dptr_;
  for (size_t i = 0; i < arr1.shape().Size(); i++)
    printf("%g, ", data1[i] - data2[i]);
  printf("\n");
}

static inline bool similar_array(const mxnet::NDArray &arr1, const mxnet::NDArray &arr2, float tol) {
  float *data1 = (float *) arr1.data().dptr_;
  float *data2 = (float *) arr2.data().dptr_;
  if (arr1.shape().Size() != arr2.shape().Size())
    return false;
  for (size_t i = 0; i < arr1.shape().Size(); i++)
    if (std::abs(data1[i] - data2[i]) > tol)
      return false;
  return true;
}

static int get_type_size(int dtype) {
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    return sizeof(DType);
  });
  return -1;
}

static inline mxnet::NDArray copy(const mxnet::NDArray &arr) {
  mxnet::NDArray new_arr(arr.shape(), arr.ctx(), false, arr.dtype());
  memcpy(new_arr.data().dptr_, arr.data().dptr_,
         arr.shape().Size() * get_type_size(arr.dtype()));
  return new_arr;
}

class TimeCount {
  std::unordered_map<std::string, size_t> nansecs;
  std::chrono::time_point<std::chrono::system_clock> start;
  std::string name;

  std::string filter_name(const std::string &name) {
    auto idx = name.find("unit");
    std::string str = idx == std::string::npos ? name : name.substr(idx + 6);
    idx = str.find_first_of("0123456789");
    if (idx != std::string::npos)
      str = str.erase(idx, 1);
    return str;
  }
 public:
  TimeCount(const std::string &name) {
    this->name = name;
  }

  ~TimeCount() {
    std::map<std::string, size_t> tmp(nansecs.begin(), nansecs.end());
    for (auto it = tmp.begin(); it != tmp.end(); it++)
      std::cerr << it->first << " takes " << it->second / 1000000 << " ms" << std::endl;
  }

  void Start(const std::string &name = "") {
    start = std::chrono::system_clock::now();
  }

  void End(const std::string &name = "") {
    auto end = std::chrono::system_clock::now();
    auto ns = end - start;
    std::string new_name = filter_name(name);
    auto it = nansecs.find(new_name);
    if (it == nansecs.end())
      nansecs.insert(std::pair<std::string, size_t>(new_name, ns.count()));
    else
      it->second += ns.count();
  }
};

#endif

