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

#ifndef LBANN_LAYER_CONCATENATION_HPP_INCLUDED
#define LBANN_LAYER_CONCATENATION_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

/** @brief Concatenate tensors along specified dimension. */
template <data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class concatenation_layer : public transform_layer {
public:

  concatenation_layer(lbann_comm *comm, El::Int concat_dim)
    : transform_layer(comm), m_concat_dim(concat_dim) {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  concatenation_layer(const concatenation_layer& other)
    : transform_layer(other),
      m_concat_dim(other.m_concat_dim),
      m_concat_points(other.m_concat_points) {
    m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
    m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
  }

  concatenation_layer& operator=(const concatenation_layer& other) {
    transform_layer::operator=(other);
    m_concat_dim = other.m_concat_dim;
    m_concat_points = other.m_concat_points;
    m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
    m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
    return *this;
  }

  concatenation_layer* copy() const override { return new concatenation_layer(*this); }
  std::string get_type() const override { return "concatenation"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    desc.add("Concatenation dimension", m_concat_dim);
    return desc;
  }

protected:

  void setup_pointers() override {
    transform_layer::setup_pointers();
    if (get_num_parents() < 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has no parents";
      LBANN_ERROR(err.str());
    }
  }

  void setup_matrices(const El::Grid& grid) override {
    transform_layer::setup_matrices(grid);
    const auto& input = get_prev_activations();
    m_input_v.reset(input.Construct(input.Grid(), input.Root()));
    m_output_v.reset(input.Construct(input.Grid(), input.Root()));
  }

  void setup_dims() override {
    transform_layer::setup_dims();

    // Get concatenation points for first parent layer
    auto output_dims = get_input_dims(0);
    if (m_concat_dim < 0
        || m_concat_dim >= (El::Int) output_dims.size()) {
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has " << output_dims.size() << " dimensions, "
          << "but attempted to concatenate along "
          << "dimension " << m_concat_dim;
      LBANN_ERROR(err.str());
    }
    m_concat_points.clear();
    m_concat_points.push_back(0);
    m_concat_points.push_back(output_dims[m_concat_dim]);

    // Get concatenation points for remaining parent layers
    for (int i = 1; i < get_num_parents(); ++i) {
      const auto& input_dims = get_input_dims(i);
      if (input_dims.size() != output_dims.size()
          || !std::equal(input_dims.begin(),
                         input_dims.begin() + m_concat_dim,
                         output_dims.begin())
          || !std::equal(input_dims.begin() + m_concat_dim + 1,
                         input_dims.end(),
                         output_dims.begin() + m_concat_dim + 1)) {
        std::stringstream err;
        err << get_type() << " layer \"" << get_name() << "\" "
            << "expects input tensors with dimensions ";
        for (size_t j = 0; j < output_dims.size(); ++j) {
          err << (j > 0 ? " x " : "");
          if ((int) j == m_concat_dim) {
            err << "X";
          } else {
            err << output_dims[j];
          }
        }
        err << ", but parent layer "
            << "\"" << m_parent_layers[i]->get_name() << "\" "
            << "outputs with dimensions ";
        for (size_t j = 0; j < input_dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << input_dims[j];
        }
        LBANN_ERROR(err.str());
      }
      output_dims[m_concat_dim] += input_dims[m_concat_dim];
      m_concat_points.push_back(output_dims[m_concat_dim]);
    }

    // Update output dimensions
    set_output_dims(output_dims);

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    if (distconv_enabled() && !keep_original_output(0)) {
      return;
    }

    const auto& num_inputs = get_num_parents();
    const auto& output_dims = get_output_dims();

    // Initialize output tensor
    auto& output = get_activations();
    output.Empty(false);
    if (num_inputs > 1) {
      output.AlignWith(get_prev_activations());
      output.Resize(get_output_size(), mini_batch_size);
    } else {
      El::LockedView(output, get_prev_activations());
      return;
    }

    // Divide output tensor into unit slices along concat dimension
    // Note: Each unit slice is divided into contiguous "unit blocks"
    const auto& output_num_unit_slices = output_dims[m_concat_dim];
    const auto& blocks_per_slice
      = (m_concat_dim > 0 ?
         std::accumulate(&output_dims[0], &output_dims[m_concat_dim],
                         1, std::multiplies<int>()) :
         1);
    const auto& unit_block_size
      = std::accumulate(output_dims.begin() + m_concat_dim + 1,
                        output_dims.end(),
                        1, std::multiplies<int>());
    const auto& output_block_stride = (output_num_unit_slices
                                       * unit_block_size);

    // Populate slices of output tensor with input tensors
    for (int i = 0; i < num_inputs; ++i) {
      const auto& input_dims = get_input_dims(i);
      auto& input = get_prev_activations(i);

      // Divide input tensor into unit slices
      const auto& input_num_unit_slices = input_dims[m_concat_dim];

      // Merge unit slices
      const auto& block_size = input_num_unit_slices * unit_block_size;
      const auto& output_block_offset = m_concat_points[i] * unit_block_size;

      // Populate output tensor one block at a time
      for (int block = 0; block < blocks_per_slice; ++block) {
        const auto& input_offset = block * block_size;
        const auto& output_offset = (output_block_offset
                                     + block * output_block_stride);
        El::LockedView(*m_input_v, input,
                       El::IR(input_offset, input_offset + block_size),
                       El::ALL);
        El::View(*m_output_v, output,
                 El::IR(output_offset, output_offset + block_size),
                 El::ALL);
        El::Copy(*m_input_v, *m_output_v);
      }

    }

  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
#ifdef LBANN_HAS_DISTCONV
  if (skip_first_layer_bp()) {
    return;
  }
#endif
    const auto& num_inputs = get_num_parents();
    const auto& output_dims = get_output_dims();

    // Divide output tensor into unit slices along concat dimension
    // Note: Each unit slice is divided into contiguous "unit blocks"
    const auto& output_num_unit_slices = output_dims[m_concat_dim];
    const auto& blocks_per_slice
      = (m_concat_dim > 0 ?
         std::accumulate(&output_dims[0], &output_dims[m_concat_dim],
                         1, std::multiplies<int>()) :
         1);
    const auto& unit_block_size
      = std::accumulate(output_dims.begin() + m_concat_dim + 1,
                        output_dims.end(),
                        1, std::multiplies<int>());
    const auto& output_block_stride = (output_num_unit_slices
                                       * unit_block_size);

    // Populate gradient w.r.t. input tensors
    const auto& gradient_wrt_output = get_prev_error_signals();
    for (int i = 0; i < num_inputs; ++i) {
#ifdef LBANN_HAS_DISTCONV
      if (distconv_enabled() && !keep_original_input(i)) continue;
#endif
      const auto& input_dims = get_input_dims(i);
      const auto& input_size = get_input_size(i);
      auto& gradient_wrt_input = get_error_signals(i);

      // Divide input tensor into unit slices
      const auto& input_num_unit_slices = input_dims[m_concat_dim];

      // Merge unit slices and get first contiguous output block
      const auto& block_size = input_num_unit_slices * unit_block_size;
      const auto& output_block_offset = m_concat_points[i] * unit_block_size;
      El::LockedView(*m_output_v, gradient_wrt_output,
                     El::IR(output_block_offset,
                            output_block_offset + block_size),
                     El::ALL);

      // Populate gradient w.r.t. input tensor one block at a time
      // Note: If there is only one block, the tensor can be a view
      if (blocks_per_slice > 1) {
        gradient_wrt_input.AlignWith(*m_output_v);
        gradient_wrt_input.Resize(input_size, mini_batch_size);
        for (int block = 0; block < blocks_per_slice; ++block) {
          const auto& input_offset = block * block_size;
          const auto& output_offset = (output_block_offset
                                       + block * output_block_stride);
          El::LockedView(*m_output_v, gradient_wrt_output,
                         El::IR(output_offset, output_offset + block_size),
                         El::ALL);
          El::View(*m_input_v, gradient_wrt_input,
                         El::IR(input_offset, input_offset + block_size),
                         El::ALL);
          El::Copy(*m_output_v, *m_input_v);
        }
      } else {
        El::LockedView(gradient_wrt_input, *m_output_v);
      }

    }

  }

  void fp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled()) {
      fp_compute_distconv();
      return;
    }
#endif
  }
  void bp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled()) {
      bp_compute_distconv();
      return;
    }
#endif
  }

private:

  /** Tensor dimension to concatenation. */
  El::Int m_concat_dim;
  /** Concatenation points for each child layer. */
  std::vector<El::Int> m_concat_points;

  /** View into input tensor. */
  std::unique_ptr<AbsDistMat> m_input_v;
  /** View into output tensor. */
  std::unique_ptr<AbsDistMat> m_output_v;

#ifdef LBANN_HAS_DISTCONV
 protected:
  std::vector<dc::TensorDev> m_prev_activations_siblings;
  std::vector<dc::TensorDev> m_error_signals_siblings;

  dc::Shape get_activations_tensor_local_shape() const override {
    auto shape = m_prev_activations_t.get_local_shape();
    shape[-2] = get_output_tensor_shape()[-2];
    return shape;
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;

    this->setup_prev_activations_tensor(dists);
    this->setup_activations_tensor(dists);
    this->setup_activations_copyout_tensor(dists);

    m_prev_activations_siblings.reserve(get_num_parents() - 1);
    for (int i = 0; i < get_num_parents() - 1; ++i) {
      if (m_parent_shuffle_required[i] ||
          m_parent_copy_in_required[i]) {
        LBANN_ERROR("Copyin non-first tensor not supported");
      }
      m_prev_activations_siblings.emplace_back(
          get_parent_layers()[i+1]->get_activations_t(*this));
    }
  }

  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    this->setup_prev_error_signals_tensor(dists);
    this->setup_error_signals_tensor(dists);
    this->setup_error_signals_copyout_tensor(dists);

    m_error_signals_siblings.reserve(get_num_parents() - 1);
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    for (int i = 0; i < get_num_parents() - 1; ++i) {
      const auto &global_shape = m_prev_activations_siblings[i].get_shape();
      const auto &local_shape = m_prev_activations_siblings[i].get_local_shape();
      m_error_signals_siblings.emplace_back(
          dc::TensorDev(global_shape, loc, dists[2], local_shape));
      assert0(m_error_signals_siblings.back().allocate());
      m_error_signals_siblings.back().zero(dc::get_stream());
    }
  }

  // TODO: Make the layer class have multiple parents and children
  const dc::TensorDev &get_error_signals_t(const Layer &parent) const {
    const auto parents = get_parent_layers();
    for (int i = 0; i < (int)parents.size(); ++i) {
      if (parents[i] == &parent) {
        if (i == 0) {
          return m_error_signals_t;
        } else {
          return m_error_signals_siblings[i-1];
        }
      }
    }
    LBANN_ERROR("No such parent found");
  }

  void fp_compute_distconv() {
    dc::MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__;
    assert_always(distconv_enabled());
    assert_always(get_num_parents() == 2);
    dc::tensor::Concatenate(m_activations_t, m_prev_activations_t,
                            m_prev_activations_siblings[0],
                            dc::get_stream());
    copy_out_activations();
  }

  void bp_compute_distconv() {
    dc::MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__;
    assert_always(distconv_enabled());
    dc::tensor::Slice(m_error_signals_t, m_error_signals_siblings[0],
                      m_prev_error_signals_t, dc::get_stream());
    copy_out_error_signals();
  }
#endif // LBANN_HAS_DISTCONV

};

#ifndef LBANN_CONCATENATION_LAYER_INSTANTIATE
extern template class concatenation_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class concatenation_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class concatenation_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class concatenation_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_CONCATENATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_CONCATENATION_HPP_INCLUDED
