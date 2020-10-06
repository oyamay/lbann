////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#define LBANN_GRU_LAYER_INSTANTIATE
#include "lbann/layers/learning/gru.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/hash.hpp"
#include <layers.pb.h>

namespace lbann {


// ---------------------------------------------
// Life cycle
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>::gru_layer(
  lbann_comm* comm,
  size_t hidden_size,
  size_t num_layers)
  : data_type_layer<TensorDataType>(comm),
    m_hidden_size{hidden_size},
    m_num_layers{num_layers} {
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>::gru_layer(const gru_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_hidden_size{other.m_hidden_size},
    m_num_layers{other.m_num_layers} {
#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED
  // No need to copy cuDNN tensor descriptors, cuDNN workspaces, or
  // CUDA graphs. They are setup in forward and backward prop
  // functions, as needed.
  /// @todo Copy @c m_rnn_cudnn_desc
  m_cuda_graph_forward_prop_cache.clear();
  m_cuda_graph_backward_prop_cache.clear();
#endif // LBANN_GRU_LAYER_GPU_SUPPORTED
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType, Layout, Device>& gru_layer<TensorDataType, Layout, Device>
::operator=(const gru_layer& other) {
  data_type_layer<TensorDataType>::operator=(other);
  m_hidden_size = other.m_hidden_size;
  m_num_layers = other.m_num_layers;
#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED
  // No need to copy cuDNN tensor descriptors, cuDNN workspaces, or
  // CUDA graphs. They are setup in forward and backward prop
  // functions, as needed.
  /// @todo Copy @c m_rnn_cudnn_desc
  m_cuda_graph_forward_prop_cache.clear();
  m_cuda_graph_backward_prop_cache.clear();
#endif // LBANN_GRU_LAYER_GPU_SUPPORTED
  return *this;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gru_layer<TensorDataType,Layout,Device>*
gru_layer<TensorDataType,Layout,Device>
::copy() const
{
  return new gru_layer(*this);
}

// ---------------------------------------------
// Query functions
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
gru_layer<TensorDataType,Layout,Device>
::get_type() const
{
  return "GRU";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
gru_layer<TensorDataType,Layout,Device>
::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
gru_layer<TensorDataType,Layout,Device>
::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
gru_layer<TensorDataType,Layout,Device>
::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Hidden size", m_hidden_size);
  desc.add("Num layers", m_num_layers);
  return desc;
}

// ---------------------------------------------
// Setup
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Check parameters
  if (m_hidden_size <= 0) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "has an invalid hidden state size (",m_hidden_size,")");
  }
  if (m_num_layers <= 0) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "has an invalid number of layers (",m_num_layers,")");
  }

  // Check input dims
  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);
  auto dims_to_str = [] (const std::vector<int>& dims) -> std::string {
    std::ostringstream ss;
    for (size_t i=0; i<dims.size(); ++i) {
      if (i > 0) {
        ss << " x ";
      }
      ss << dims[i];
    }
    return ss.str();
  };
  if (input0_dims.size() != 2) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "expected a 2D input tensor for the input sequence, "
      "but recieved a tensor with ",
      "dimensions of ",dims_to_str(input0_dims));
  }
  if (input1_dims.size() != 2
      || static_cast<size_t>(input1_dims[0]) != m_num_layers
      || static_cast<size_t>(input1_dims[1]) != m_hidden_size) {
    LBANN_ERROR(
      this->get_type()," layer \"",this->get_name(),"\" ",
      "expected a ",m_num_layers," x ",m_hidden_size," input tensor ",
      "for the initial hidden state, ",
      "but recieved a tensor with ",
      "dimensions of ",dims_to_str(input1_dims));
  }

  // Set output dims
  const std::vector<int> output_dims = {input0_dims[0], static_cast<int>(m_hidden_size)};
  this->set_output_dims(output_dims);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>
::setup_data(size_t max_mini_batch_size) {
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  const size_t sequence_length = this->get_input_dims()[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // Construct default weights if needed
  if (!this->has_weights()) {
    const std::vector<std::string> weight_names
      = {"ih_matrix", "hh_matrix", "ih_bias", "hh_bias"};
    this->set_num_weights(4*m_num_layers);
    const auto scale = El::To<TensorDataType>(1./std::sqrt(m_hidden_size));
    for (size_t i=0; i<m_num_layers; ++i) {
      for (size_t j=0; j<4; ++j) {
        auto w = make_unique<data_type_weights<TensorDataType>>(this->get_comm());
        auto init = make_unique<uniform_initializer<TensorDataType>>(-scale, scale);
        auto opt = this->m_model->template create_optimizer<TensorDataType>();
        w->set_name(lbann::build_string(this->get_name(),"_",weight_names[j],"_l",i));
        w->set_initializer(std::move(init));
        w->set_optimizer(std::move(opt));
        this->set_weights(4*i+j, w.get());
        this->m_model->add_weights(std::move(w));
      }
    }
  }
  if (this->num_weights() != 4*m_num_layers) {
    LBANN_ERROR(
      "attempted to setup ",
      this->get_type()," layer \"",this->get_name(),"\" ",
      "with an invalid number of weights ",
      "(expected ",4*m_num_layers,", found ",this->num_weights(),")");
  }

  // Setup weight dimensions and distribution
  for (size_t i=0; i<m_num_layers; ++i) {
    auto& ih_matrix = this->get_weights(4*i);
    auto& hh_matrix = this->get_weights(4*i+1);
    auto& ih_bias = this->get_weights(4*i+2);
    auto& hh_bias = this->get_weights(4*i+3);

    ih_matrix.set_dims(
      {static_cast<int>(3*m_hidden_size)},
      {static_cast<int>(i == 0 ? input_size : m_hidden_size)});
    hh_matrix.set_dims(
      {static_cast<int>(3*m_hidden_size)},
      {static_cast<int>(m_hidden_size)});
    ih_bias.set_dims({static_cast<int>(3*m_hidden_size)});
    hh_bias.set_dims({static_cast<int>(3*m_hidden_size)});
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    ih_matrix.set_matrix_distribution(dist);
    hh_matrix.set_matrix_distribution(dist);
    ih_bias.set_matrix_distribution(dist);
    hh_bias.set_matrix_distribution(dist);
  }

}

#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED
template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::setup_gpu() {

  // Dimensions
  const size_t sequence_length = this->get_input_dims(0)[0];
  const size_t input_size = this->get_input_size(0) / sequence_length;

  // RNN descriptor
  static cudnn::DropoutDescriptor dropout_desc;
  dropout_desc.set(0, nullptr, 0, 0);
  m_rnn_cudnn_desc.set(
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_GRU,
    CUDNN_RNN_DOUBLE_BIAS,
    CUDNN_UNIDIRECTIONAL,
    CUDNN_LINEAR_INPUT,
    cudnn::get_data_type<TensorDataType>(),
    cudnn::get_data_type<TensorDataType>(),
    cudnn::get_default_convolution_math_type(),
    input_size,
    m_hidden_size,
    m_hidden_size,  // proj_size
    m_num_layers,
    dropout_desc,
    CUDNN_RNN_PADDED_IO_ENABLED);

}
#endif // LBANN_GRU_LAYER_GPU_SUPPORTED

// ---------------------------------------------
// Forward prop
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::fp_compute() {
  fp_compute_impl(*this);
}

#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED

namespace {
template <typename TensorDataType>
void pack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const cudnn::RNNDescriptor& rnn_desc,
  const El::SyncInfo<El::Device::GPU>& sync_info,
  size_t input_size,
  size_t hidden_size,
  size_t num_layers,
  void* packed_weights_buffer,
  size_t packed_weights_size,
  const std::vector<El::Matrix<TensorDataType,El::Device::GPU>>& weights_list) {

  // Construct objects
  static cudnn::TensorDescriptor matrix_desc, bias_desc;
  El::Matrix<TensorDataType,El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Function to get pointers in packed weights buffer
  using PtrPair = std::pair<TensorDataType*,TensorDataType*>;
  auto get_ptrs = [&] (size_t i, size_t id) -> PtrPair {
    PtrPair ptrs;
    matrix_desc.create();
    bias_desc.create();
    CHECK_CUDNN(
      cudnnGetRNNWeightParams(
        handle,
        rnn_desc,
        i,
        packed_weights_size,
        packed_weights_buffer,
        id,
        matrix_desc,
        reinterpret_cast<void**>(&ptrs.first),
        bias_desc,
        reinterpret_cast<void**>(&ptrs.second)));
    return ptrs;
  };

  for (size_t i=0; i<num_layers; ++i) {

    // Copy from ih_matrix
    const auto& ih_matrix = weights_list[4*i];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(
        i == 0 ? input_size : hidden_size,
        hidden_size,
        get_ptrs(i, id).first,
        i == 0 ? input_size : hidden_size);
      El::Transpose(
        ih_matrix(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL),
        packed_weights_view,
        false);
    }

    // Copy from hh_matrix
    const auto& hh_matrix = weights_list[4*i+1];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(
        hidden_size,
        hidden_size,
        get_ptrs(i, 3+id).first,
        hidden_size);
      El::Transpose(
        hh_matrix(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL),
        packed_weights_view,
        false);
    }

    // Copy from ih_bias
    const auto& ih_bias = weights_list[4*i+2];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(
        hidden_size,
        1,
        get_ptrs(i, id).second,
        hidden_size);
      El::Copy(
        ih_bias(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL),
        packed_weights_view);
    }

    // Copy from hh_bias
    const auto& hh_bias = weights_list[4*i+3];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.Attach(
        hidden_size,
        1,
        get_ptrs(i, 3+id).second,
        hidden_size);
      El::Copy(
        hh_bias(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL),
        packed_weights_view);
    }

  }

}

} // namespace <anon>

#define MIN_WORKSPACE_MINI_BATCH_SIZE 128
size_t active_min_workspace_mini_batch_size = 0;

template <typename TensorDataType>
void fp_compute_impl(
  gru_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& input_sequence
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& init_hidden
    = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& output_sequence
    = dynamic_cast<LocalMat&>(l.get_local_activations());

  // Dimensions
  const size_t sequence_length = l.get_input_dims(0)[0];
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;
  const size_t num_layers = l.m_num_layers;

  // Configure workspace mini-batch size
  // Note: Return immediately if there is no local data.
  // Note (tym 10/3/20): We experience an error in cuDNN with certain
  // mini-batch sizes. Hack around it by padding to a minimum batch
  // size.
  /// @todo Figure out cuDNN bug
  const size_t mini_batch_size = input_sequence.Width();
  if (mini_batch_size <= 0) {
    return;
  }
  // Note (BVE 10/3/20): Note that the bug seems to be triggered by
  // switching mini-batch sizes when the last mini-batch is too small.
  // This means that we need a lower bound, which is emperically
  // tested to be 128 for WAE.  However, if the initial mini-batch
  // size is less than 128 and it isn't changed, things seem to be
  // okay.  So set the threshold to be the smaller of the initial
  // mini-batch size or 128.
  if(active_min_workspace_mini_batch_size == 0) {
    // Set the minumum to the smaller of the initial mini-batch size
    // or a predefined minumim
    active_min_workspace_mini_batch_size = El::Min(mini_batch_size, MIN_WORKSPACE_MINI_BATCH_SIZE);
  }
  const size_t workspace_mini_batch_size = El::Max(mini_batch_size, active_min_workspace_mini_batch_size);

  // GPU objects
  auto&& sync_info = input_sequence.GetSyncInfo();
  auto&& stream = sync_info.Stream();
  auto&& handle = cudnn::get_handle();
  auto&& rnn_desc = l.m_rnn_cudnn_desc;
  const auto data_type = cudnn::get_data_type<TensorDataType>();

  // Configure input and output tensor descriptors
  std::vector<int> sequence_lengths(workspace_mini_batch_size, sequence_length);
  l.m_input_cudnn_desc.set(
    data_type,
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
    sequence_length,
    workspace_mini_batch_size,
    input_size,
    sequence_lengths.data(),
    nullptr);
  l.m_output_cudnn_desc.set(
    data_type,
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
    sequence_length,
    workspace_mini_batch_size,
    hidden_size,
    sequence_lengths.data(),
    nullptr);
  l.m_hidden_cudnn_desc.set(data_type, num_layers, workspace_mini_batch_size, hidden_size);

  // Allocate cuDNN workspace buffers
  size_t cudnn_workspace_size, cudnn_reserve_space_size;
  CHECK_CUDNN(
    cudnnGetRNNTempSpaceSizes(
      handle,
      rnn_desc,
      CUDNN_FWD_MODE_TRAINING,
      l.m_input_cudnn_desc,
      &cudnn_workspace_size,
      &cudnn_reserve_space_size));
  if (l.m_cudnn_workspace.size() < cudnn_workspace_size) {
    /// @todo Handle synchronization
    l.m_cudnn_workspace.allocate(cudnn_workspace_size);
  }
  if (l.m_cudnn_reserve_space.size() < cudnn_reserve_space_size) {
    /// @todo Handle synchronization
    l.m_cudnn_reserve_space.allocate(cudnn_reserve_space_size);
  }
  if (l.m_gpu_sequence_lengths.size() < workspace_mini_batch_size) {
    /// @todo Handle synchronization
    l.m_gpu_sequence_lengths.allocate(workspace_mini_batch_size);
    std::vector<int32_t> cpu_sequence_lengths(workspace_mini_batch_size, sequence_length);
    CHECK_CUDA(
      cudaMemcpyAsync(
        l.m_gpu_sequence_lengths.data(),
        cpu_sequence_lengths.data(),
        cpu_sequence_lengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Make sure tensors are in correct format
  l.m_input_sequence_workspace.SetSyncInfo(sync_info);
  l.m_init_hidden_workspace.SetSyncInfo(sync_info);
  l.m_output_sequence_workspace.SetSyncInfo(sync_info);
  constexpr size_t one{1};
  l.m_input_sequence_workspace.Resize(
    sequence_length*input_size, workspace_mini_batch_size);
  l.m_init_hidden_workspace.Resize(
    workspace_mini_batch_size*hidden_size, num_layers);
  l.m_output_sequence_workspace.Resize(
    sequence_length*hidden_size, workspace_mini_batch_size);
  El::Zero(l.m_input_sequence_workspace);
  El::Zero(l.m_init_hidden_workspace);
  auto input_sequence_workspace_
    = l.m_input_sequence_workspace(El::ALL, El::IR(0, mini_batch_size));
  El::Copy(input_sequence, input_sequence_workspace_);
  cuda::copy_tensor(
    stream,
    {mini_batch_size, num_layers, hidden_size},
    init_hidden.LockedBuffer(),
    {static_cast<size_t>(init_hidden.LDim()), hidden_size, one},
    l.m_init_hidden_workspace.Buffer(),
    {hidden_size, workspace_mini_batch_size*hidden_size, one});

  // Pack weights into workspace buffer
  /// @todo Handle synchronization
  size_t weights_size;
  CHECK_CUDNN(
    cudnnGetRNNWeightSpaceSize(
      handle,
      l.m_rnn_cudnn_desc,
      &weights_size));
  l.m_weights_cudnn_workspace.allocate(weights_size);
  std::vector<LocalMat> weights_list;
  for (size_t i=0; i<4*num_layers; ++i) {
    const auto& w
      = dynamic_cast<const LocalMat&>(l.weights_values(i).LockedMatrix());
    weights_list.emplace_back(El::LockedView(w));
  }
  pack_cudnn_rnn_weights<TensorDataType>(
    handle,
    rnn_desc,
    sync_info,
    input_size,
    hidden_size,
    num_layers,
    l.m_weights_cudnn_workspace.data(),
    l.m_weights_cudnn_workspace.size(),
    weights_list);

  // Compute hash with cuDNN function arguments
  size_t hash{0};
  hash = hash_combine(hash, l.m_gpu_sequence_lengths.data());
  hash = hash_combine(hash, l.m_input_sequence_workspace.LockedBuffer());
  hash = hash_combine(hash, l.m_init_hidden_workspace.LockedBuffer());
  hash = hash_combine(hash, l.m_output_sequence_workspace.Buffer());
  hash = hash_combine(hash, l.m_weights_cudnn_workspace.data());
  hash = hash_combine(hash, l.m_cudnn_workspace.data());
  hash = hash_combine(hash, l.m_cudnn_reserve_space.data());

  // Update graph cache if cuDNN function arguments don't match
  if (l.m_cuda_graph_forward_prop_cache.count(workspace_mini_batch_size) < 1
      || l.m_cuda_graph_forward_prop_cache[workspace_mini_batch_size].first != hash) {

    // Capture graph
    cuda::Graph::begin_capture(stream);
    CHECK_CUDNN(
      cudnnRNNForward(
        handle,
        rnn_desc,
        CUDNN_FWD_MODE_TRAINING,
        l.m_gpu_sequence_lengths.data(),
        l.m_input_cudnn_desc,
        l.m_input_sequence_workspace.LockedBuffer(),
        l.m_output_cudnn_desc,
        l.m_output_sequence_workspace.Buffer(),
        l.m_hidden_cudnn_desc,
        l.m_init_hidden_workspace.LockedBuffer(),
        nullptr,                // hy
        l.m_hidden_cudnn_desc,  // cDesc
        nullptr,                // cx
        nullptr,                // cy
        l.m_weights_cudnn_workspace.size(),
        l.m_weights_cudnn_workspace.data(),
        l.m_cudnn_workspace.size(),
        l.m_cudnn_workspace.data(),
        l.m_cudnn_reserve_space.size(),
        l.m_cudnn_reserve_space.data()));
    auto graph = cuda::Graph::end_capture(stream);

    // Update cache
    auto& cache_pair = l.m_cuda_graph_forward_prop_cache[workspace_mini_batch_size];
    cache_pair.first = hash;
    cache_pair.second.update(graph);

  }

  // Launch CUDA graph with cuDNN kernels
  l.m_cuda_graph_forward_prop_cache[workspace_mini_batch_size].second.launch(stream);

  // Output tensor
  El::LockedView(
    output_sequence,
    l.m_output_sequence_workspace,
    El::ALL,
    El::IR(0, mini_batch_size));

}

#endif // LBANN_GRU_LAYER_GPU_SUPPORTED

// ---------------------------------------------
// Back prop
// ---------------------------------------------

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gru_layer<TensorDataType, Layout, Device>::bp_compute() {
  bp_compute_impl(*this);
}

#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED

namespace {
template <typename TensorDataType>
void unpack_cudnn_rnn_weights(
  const cudnnHandle_t& handle,
  const cudnn::RNNDescriptor& rnn_desc,
  const El::SyncInfo<El::Device::GPU>& sync_info,
  size_t input_size,
  size_t hidden_size,
  size_t num_layers,
  const void* packed_weights_buffer,
  size_t packed_weights_size,
  const std::vector<El::Matrix<TensorDataType,El::Device::GPU>>& weights_list) {

  // Construct objects
  static cudnn::TensorDescriptor matrix_desc, bias_desc;
  El::Matrix<TensorDataType,El::Device::GPU> packed_weights_view;
  packed_weights_view.SetSyncInfo(sync_info);

  // Function to get pointers in packed weights buffer
  using PtrPair = std::pair<TensorDataType*,TensorDataType*>;
  auto get_ptrs = [&] (size_t i, size_t id) -> PtrPair {
    PtrPair ptrs;
    matrix_desc.create();
    bias_desc.create();
    CHECK_CUDNN(
      cudnnGetRNNWeightParams(
        handle,
        rnn_desc,
        i,
        packed_weights_size,
        packed_weights_buffer,
        id,
        matrix_desc,
        reinterpret_cast<void**>(&ptrs.first),
        bias_desc,
        reinterpret_cast<void**>(&ptrs.second)));
    return ptrs;
  };

  for (size_t i=0; i<num_layers; ++i) {

    // Copy to ih_matrix
    auto& ih_matrix = weights_list[4*i];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(
        i == 0 ? input_size : hidden_size,
        hidden_size,
        get_ptrs(i, id).first,
        i == 0 ? input_size : hidden_size);
      auto ih_matrix_view = ih_matrix(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL);
      El::Transpose(packed_weights_view, ih_matrix_view, false);
    }

    // Copy to hh_matrix
    auto& hh_matrix = weights_list[4*i+1];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(
        hidden_size,
        hidden_size,
        get_ptrs(i, 3+id).first,
        hidden_size);
      auto hh_matrix_view = hh_matrix(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL);
      El::Transpose(packed_weights_view, hh_matrix_view, false);
    }

    // Copy to ih_bias
    auto& ih_bias = weights_list[4*i+2];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(
        hidden_size,
        1,
        get_ptrs(i, id).second,
        hidden_size);
      auto ih_bias_view = ih_bias(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL);
      El::Copy(packed_weights_view, ih_bias_view);
    }

    // Copy to hh_bias
    auto& hh_bias = weights_list[4*i+3];
    for (auto id : {0, 1, 2}) {
      packed_weights_view.LockedAttach(
        hidden_size,
        1,
        get_ptrs(i, 3+id).second,
        hidden_size);
      auto hh_bias_view = hh_bias(El::IR(id*hidden_size, (id+1)*hidden_size), El::ALL);
      El::Copy(packed_weights_view, hh_bias_view);
    }

  }

}
} // namespace <anon>

template <typename TensorDataType>
void bp_compute_impl(
  gru_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

  // Matrices
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& output_sequence_grad
    = dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& input_sequence_grad
    = dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& init_hidden_grad
    = dynamic_cast<LocalMat&>(l.get_local_error_signals(1));

  // Dimensions
  const size_t sequence_length = l.get_input_dims(0)[0];
  const size_t input_size = l.get_input_size(0) / sequence_length;
  const size_t hidden_size = l.m_hidden_size;
  const size_t num_layers = l.m_num_layers;

  // Configure workspace mini-batch size
  // Note (tym 10/3/20): We experience an error in cuDNN with certain
  // mini-batch sizes. Hack around it by padding to a minimum batch
  // size.
  /// @todo Figure out cuDNN bug
  const size_t mini_batch_size = output_sequence_grad.Width();
  if (mini_batch_size <= 0) {
    return;
  }
  const size_t workspace_mini_batch_size = El::Max(mini_batch_size, active_min_workspace_mini_batch_size);

  // GPU objects
  auto&& sync_info = output_sequence_grad.GetSyncInfo();
  auto&& stream = sync_info.Stream();
  auto&& handle = cudnn::get_handle();
  auto&& rnn_desc = l.m_rnn_cudnn_desc;

  // Define closure to send weight gradients to optimizers
  std::vector<LocalMat> weights_grad_list(4*num_layers);
  for (auto& dw : weights_grad_list) {
    dw.SetSyncInfo(sync_info);
  }
  for (size_t i=0; i<num_layers; ++i) {
    weights_grad_list[4*i].Resize(
      3*hidden_size,
      i == 0 ? input_size : hidden_size);
    weights_grad_list[4*i+1].Resize(3*hidden_size, hidden_size);
    weights_grad_list[4*i+2].Resize(3*hidden_size, 1);
    weights_grad_list[4*i+3].Resize(3*hidden_size, 1);
  }
  auto send_weight_grads_to_optimizers = [&] () {
    TensorDataType buf_scale, in_scale;
    for (size_t i=0; i<4*num_layers; ++i) {
      auto&& opt = l.get_weights(i).get_optimizer();
      if (opt != nullptr) {
        auto& buf = opt->get_gradient_buffer(buf_scale, in_scale, true);
        El::Scale(buf_scale, buf);
        El::Axpy(in_scale, weights_grad_list[i], buf.Matrix());
      }
    }
  };

  // Return immediately if there is no local data
  if (mini_batch_size <= 0) {
    for (auto& dw : weights_grad_list) {
      El::Zero(dw);
    }
    send_weight_grads_to_optimizers();
  }

  // Make sure tensors are in correct format
  // Note: m_input_sequence_workspace and m_init_hidden_workspace have
  // already been setup in forward prop
  l.m_output_sequence_grad_workspace.SetSyncInfo(sync_info);
  l.m_input_sequence_grad_workspace.SetSyncInfo(sync_info);
  l.m_init_hidden_grad_workspace.SetSyncInfo(sync_info);
  l.m_output_sequence_grad_workspace.Resize(
    sequence_length*hidden_size, workspace_mini_batch_size);
  l.m_input_sequence_grad_workspace.Resize(
    sequence_length*input_size, workspace_mini_batch_size);
  l.m_init_hidden_grad_workspace.Resize(
    workspace_mini_batch_size*hidden_size, num_layers);
  El::Zero(l.m_output_sequence_grad_workspace);
  auto output_sequence_grad_workspace_
    = l.m_output_sequence_grad_workspace(El::ALL, El::IR(0, mini_batch_size));
  El::Copy(output_sequence_grad, output_sequence_grad_workspace_);

  // Initialize workspace for weight gradients
  // Note: Weights have already been packed in forward prop
  l.m_weights_grad_cudnn_workspace.allocate(l.m_weights_cudnn_workspace.size());
  CHECK_CUDA(
    cudaMemsetAsync(
      l.m_weights_grad_cudnn_workspace.data(),
      0,
      l.m_weights_grad_cudnn_workspace.size(),
      stream));

  // Compute hash with cuDNN function arguments
  size_t hash{0};
  hash = hash_combine(hash, l.m_gpu_sequence_lengths.data());
  hash = hash_combine(hash, l.m_input_sequence_workspace.LockedBuffer());
  hash = hash_combine(hash, l.m_input_sequence_grad_workspace.Buffer());
  hash = hash_combine(hash, l.m_init_hidden_workspace.LockedBuffer());
  hash = hash_combine(hash, l.m_init_hidden_grad_workspace.Buffer());
  hash = hash_combine(hash, l.m_output_sequence_workspace.LockedBuffer());
  hash = hash_combine(hash, l.m_output_sequence_grad_workspace.LockedBuffer());
  hash = hash_combine(hash, l.m_weights_cudnn_workspace.data());
  hash = hash_combine(hash, l.m_weights_grad_cudnn_workspace.data());
  hash = hash_combine(hash, l.m_cudnn_workspace.data());
  hash = hash_combine(hash, l.m_cudnn_reserve_space.data());

  // Update graph cache if cuDNN function arguments don't match
  if (l.m_cuda_graph_backward_prop_cache.count(workspace_mini_batch_size) < 1
      || l.m_cuda_graph_backward_prop_cache[workspace_mini_batch_size].first != hash) {

    // Capture graph
    cuda::Graph::begin_capture(stream);
    CHECK_CUDNN(
      cudnnRNNBackwardData_v8(
        handle,
        rnn_desc,
        l.m_gpu_sequence_lengths.data(),
        l.m_output_cudnn_desc,
        l.m_output_sequence_workspace.LockedBuffer(),
        l.m_output_sequence_grad_workspace.LockedBuffer(),
        l.m_input_cudnn_desc,
        l.m_input_sequence_grad_workspace.Buffer(),
        l.m_hidden_cudnn_desc,
        l.m_init_hidden_workspace.LockedBuffer(),
        nullptr,                // dhy
        l.m_init_hidden_grad_workspace.Buffer(),
        l.m_hidden_cudnn_desc,  // cDesc
        nullptr,                // cx
        nullptr,                // dcy
        nullptr,                // dcx
        l.m_weights_cudnn_workspace.size(),
        l.m_weights_cudnn_workspace.data(),
        l.m_cudnn_workspace.size(),
        l.m_cudnn_workspace.data(),
        l.m_cudnn_reserve_space.size(),
        l.m_cudnn_reserve_space.data()));
    CHECK_CUDNN(
      cudnnRNNBackwardWeights_v8(
        handle,
        rnn_desc,
        CUDNN_WGRAD_MODE_ADD,
        l.m_gpu_sequence_lengths.data(),
        l.m_input_cudnn_desc,
        l.m_input_sequence_workspace.LockedBuffer(),
        l.m_hidden_cudnn_desc,
        l.m_init_hidden_workspace.LockedBuffer(),
        l.m_output_cudnn_desc,
        l.m_output_sequence_workspace.LockedBuffer(),
        l.m_weights_grad_cudnn_workspace.size(),
        l.m_weights_grad_cudnn_workspace.data(),
        l.m_cudnn_workspace.size(),
        l.m_cudnn_workspace.data(),
        l.m_cudnn_reserve_space.size(),
        l.m_cudnn_reserve_space.data()));
    auto graph = cuda::Graph::end_capture(stream);

    // Update cache
    auto& cache_pair = l.m_cuda_graph_backward_prop_cache[workspace_mini_batch_size];
    cache_pair.first = hash;
    cache_pair.second.update(graph);

  }

  // Launch CUDA graph with cuDNN kernels
  l.m_cuda_graph_backward_prop_cache[workspace_mini_batch_size].second.launch(stream);

  // Send gradients to optimizers
  unpack_cudnn_rnn_weights<TensorDataType>(
    handle,
    rnn_desc,
    sync_info,
    input_size,
    hidden_size,
    num_layers,
    l.m_weights_grad_cudnn_workspace.data(),
    l.m_weights_grad_cudnn_workspace.size(),
    weights_grad_list);
  send_weight_grads_to_optimizers();

  // Gradients w.r.t. input tensors
  // Note: We can't output directly to layer's input grad tensors
  // since they are allocated every step from the memory pool,
  // preventing us from reusing a CUDA graph.
  constexpr size_t one{1};
  El::LockedView(
    input_sequence_grad,
    l.m_input_sequence_grad_workspace,
    El::ALL,
    El::IR(0, mini_batch_size));
  cuda::copy_tensor(
    stream,
    {mini_batch_size, num_layers, hidden_size},
    l.m_init_hidden_grad_workspace.LockedBuffer(),
    {hidden_size, workspace_mini_batch_size*hidden_size, one},
    init_hidden_grad.Buffer(),
    {static_cast<size_t>(init_hidden_grad.LDim()), hidden_size, one});

}

#endif // LBANN_GRU_LAYER_GPU_SUPPORTED

// ---------------------------------------------
// Builder
// ---------------------------------------------

namespace
{

template <typename TensorDataType, data_layout Layout, El::Device Device>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR(
      "Attempted to construct gru_layer with invalid parameters ",
      "(TensorDataType=",TypeName<TensorDataType>(),", ",
      "Layout=",to_string(Layout),", ",
      "Device=",to_string(Device),")");
    return nullptr;
  }
};

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
struct Builder<TensorDataType,data_layout::DATA_PARALLEL,El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    constexpr auto Layout = data_layout::DATA_PARALLEL;
    constexpr auto Device = El::Device::GPU;
#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED
    using LayerType = gru_layer<TensorDataType,Layout,Device>;
    return make_unique<LayerType>(std::forward<Args>(args)...);
#else
    LBANN_ERROR(
      "GPU gru_layer requires at least CUDA 11.0 and cuDNN 8.0.4 "
      "(TensorDataType=",TypeName<TensorDataType>(),", ",
      "Layout=",to_string(Layout),", ",
      "Device=",to_string(Device),")");
    return nullptr;
#endif // LBANN_GRU_LAYER_GPU_SUPPORTED
  }
};
#endif // LBANN_HAS_GPU

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_gru_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& proto_layer)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, gru);
  const auto& params = proto_layer.gru();
  const size_t num_layers = (params.has_num_layers()
                             ? params.num_layers().value()
                             : 1);
  return BuilderType::Build(comm, params.hidden_size(), num_layers);
}

// ---------------------------------------------
// Explicit template instantiation
// ---------------------------------------------

/// @todo CPU implementation
#ifdef LBANN_GRU_LAYER_GPU_SUPPORTED
#define PROTO(T)                                                        \
  template class gru_layer<                                             \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_GRU_LAYER_GPU_SUPPORTED

#define PROTO_DEVICE(T, Device)                 \
  LBANN_LAYER_BUILDER_ETI(gru, T, Device)
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

} // namespace lbann
