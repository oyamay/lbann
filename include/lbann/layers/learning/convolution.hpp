////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

/// Convolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class convolution_layer : public base_convolution_layer<Dev> {
 private:

  friend class lbann_callback_imcomm;

 public:

  /// kernel tensor is output channels, input channels, conv dimension (w x h)
  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " convolution; conv_dims: ";
    // for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
    //   if (h == 0) { s << " channels (out x in) "; }
    //   if (h == 2) { s << " filters (w x h) "; }
    //   s << this->m_kernel_dims[h] << " ";
    // }
    s << get_topo_description();
    s << " pads: ";
    for (size_t h=0; h<this->m_pads.size(); h++) {
      s << this->m_pads[h] << " ";
    }
    s << " strides: ";
    for (size_t h=0; h<this->m_strides.size(); h++) {
      s << this->m_strides[h] << " ";
    }
    s << " num_output_channels: " << this->get_output_dims()[0]
      << " has_bias: " << this->m_bias_scaling_factor
      << " dataLayout: " << this->get_data_layout_string(get_data_layout())
      << " device alloc: " + this->get_device_allocation_string(get_device_allocation());
    return s.str();
  }

  std::string get_topo_description() const override {
    std::stringstream s;
    // Get the topo description from any parent class
    std::string str = base_convolution_layer<Dev>::get_topo_description();
    s << str << " - ";

    // Display the topology of the kernel
    for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
      if (h == 0) { s << "C="; }
      s << this->m_kernel_dims[h] ;
      if (h == 0) { s << "o,"; }
      if (h == 1) { s << "i F="; }
      if (this->m_kernel_dims.size() == 3) {
        if (h == 2) { s << "w "; }
      }else if (this->m_kernel_dims.size() == 4) {
        if (h == 2) { s << "w x "; }
        if (h == 3) { s << "h"; }
      }else {
        if (h > 1) {
          s << " ";
        }
      }
    }
    return s.str();;
  }

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int pad,
                    int stride,
                    bool has_bias = true)
    : convolution_layer(comm,
                        num_data_dims,
                        num_output_channels,
                        std::vector<int>(num_data_dims, conv_dim),
                        std::vector<int>(num_data_dims, pad),
                        std::vector<int>(num_data_dims, stride),
                        has_bias) {}

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    std::vector<int> conv_dims,
                    std::vector<int> pads,
                    std::vector<int> strides,
                    bool has_bias = true)
    : base_convolution_layer<Dev>(comm,
                                  num_data_dims,
                                  num_output_channels,
                                  conv_dims,
                                  pads,
                                  strides,
                                  has_bias) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");

  }

  convolution_layer* copy() const override { return new convolution_layer(*this); }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    base_convolution_layer<Dev>::setup_dims();

    // Get tensor dimensions
    auto& kernel_dims = this->m_kernel_dims;
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;

    // Initialize convolution kernel dimensions
    kernel_dims.insert(kernel_dims.begin() + 1, input_dims[0]);
    this->m_kernel_size = std::accumulate(kernel_dims.begin(),
                                          kernel_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Check if input tensor dimensions are valid
    if (input_dims.size() != kernel_dims.size() - 1) {
      std::stringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << "has an input tensor with "
          << input_dims.size() << " dimensions "
          << "and a convolution kernel with "
          << kernel_dims.size() << " dimensions";
      LBANN_ERROR(err.str());
    }

    // Initialize output tensor dimensions
    output_dims[0] = kernel_dims[0];
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const auto& stride = this->m_strides[i];
      const auto& pad = this->m_pads[i];
      const auto& effective_dim = (input_dims[i+1]
                                   + 2 * pad
                                   - kernel_dims[i+2] + 1);
      output_dims[i+1] = (effective_dim + stride - 1) / stride;
    }
    this->set_output_dims(output_dims);

  }

 protected:

  
  void fp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        apply_convolution_distconv();
        apply_bias_distconv();
        this->copy_out_activations();
        if (this->early_terminate_last_iteration()) {
          base_convolution_layer<Dev>::apply_convolution_cudnn(true);
          base_convolution_layer<Dev>::apply_bias_cudnn();
          this->dump_reference_activations();
        }
      } else {
        base_convolution_layer<Dev>::apply_convolution_cudnn(true);
        base_convolution_layer<Dev>::apply_bias_cudnn();
      }
#else
      base_convolution_layer<Dev>::apply_convolution_cudnn(true);
      base_convolution_layer<Dev>::apply_bias_cudnn();
#endif
    } else {
      base_convolution_layer<Dev>::apply_convolution_im2col(true);
      base_convolution_layer<Dev>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        compute_gradients_distconv();
        apply_transposed_convolution_distconv();
        if (this->early_terminate_last_iteration()) {
          base_convolution_layer<Dev>::compute_gradients_cudnn(false);
          base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(false);
          this->dump_reference_error_signals();
        }
      } else {
        base_convolution_layer<Dev>::compute_gradients_cudnn(false);
        base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(false);
      }
#else
      base_convolution_layer<Dev>::compute_gradients_cudnn(false);
      base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(false);
#endif
    } else {
      base_convolution_layer<Dev>::compute_gradients_im2col(false);
      base_convolution_layer<Dev>::apply_transposed_convolution_im2col(false);
    }
  }

  void apply_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    dc::MPIPrintStreamDebug() << this->get_name() << ": Forward convolution\n";

    assert0(dc::tensor::View(
        m_kernel_t, this->get_weights()[0]->get_values().LockedBuffer()));

    m_conv->forward(DataType(1.0), this->m_prev_activations_t, m_kernel_t,
                    DataType(0.0), this->m_activations_t);
#endif
  }

  void apply_bias_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    if (this->m_bias_scaling_factor == DataType(0)) return;

    dc::MPIPrintStreamDebug() << "Applying bias\n";

    assert0(dc::tensor::View(
        m_bias_t, this->get_weights()[1]->get_values().LockedBuffer()));
    m_conv->apply_bias(this->m_bias_scaling_factor, m_bias_t,
                       DataType(1), this->m_activations_t);
#endif
  }

  void apply_transposed_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    dc::MPIPrintStreamDebug() << this->get_name() << ": Backward convolution\n";

    // input: m_prev_error_signals_d[0]
    // kernel: m_weights[0]->get_values_gpu()
    // output: m_error_signals_d[0]

    assert0(dc::tensor::View(
        m_kernel_t, this->get_weights()[0]->get_values().LockedBuffer()));

    this->m_error_signals_t.zero();
    dc::MPIPrintStreamDebug() << "Calling backward_data\n";
    m_conv->backward_data(DataType(1.0), m_kernel_t, this->m_prev_error_signals_t,
                          DataType(1.0), this->m_error_signals_t);

    this->copy_out_error_signals();
#endif    
  }

  void compute_gradients_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    dc::MPIPrintStreamDebug() << this->get_name() << ": Compute gradients\n";

    const int effective_mini_batch_size =
        this->m_model->get_effective_mini_batch_size();    

    optimizer* bias_optimizer = this->get_weights()[1]->get_optimizer();
    if (bias_optimizer != nullptr && this->m_bias_scaling_factor != DataType(0)) {
      dc::MPIPrintStreamDebug() << "Compute bias gradients\n";
      assert0(dc::tensor::View(m_bias_gradient_t,
                               this->m_bias_gradient.Buffer()));
      m_conv->backward_bias(DataType(1.0), this->m_prev_error_signals_t,
                            DataType(0.0), m_bias_gradient_t, false);
      const DataType bias_scale = this->m_bias_scaling_factor / effective_mini_batch_size;
      if (!this->early_terminate_last_iteration()) {
        bias_optimizer->add_to_gradient_staging(this->m_bias_gradient,
                                                bias_scale);
      }
    }

    optimizer* kernel_optimizer = this->get_weights()[0]->get_optimizer();
    if (kernel_optimizer == nullptr) return;

    dc::MPIPrintStreamDebug() << "Compute kernel gradients\n";          

    assert0(dc::tensor::View(
        m_kernel_gradient_e, this->m_kernel_gradient.Buffer()));

    m_conv->backward_filter(DataType(1.0), this->m_prev_activations_t,
                            this->m_prev_error_signals_t, DataType(0),
                            m_kernel_gradient_e, false);

    // Add gradient contribution
    const DataType kernel_scale = DataType(1) / effective_mini_batch_size;
    if (!this->early_terminate_last_iteration()) {
      kernel_optimizer->add_to_gradient_staging(this->m_kernel_gradient,
                                                kernel_scale);
    }
#endif    
  }

#ifdef LBANN_HAS_DISTCONV
 public:
  
  dc::Array4 get_prev_activations_overlap() const override {
    if (this->distconv_enabled()) {
      int stencil_h = (this->m_kernel_dims[2] - 1) / 2;
      int stencil_w = (this->m_kernel_dims[3] - 1) / 2;
      return dc::Array4({stencil_w, stencil_h, 0, 0});
    } else {
      return dc::Array4(0);
    }
  }

  dc::Array4 get_activations_overlap() const override {
    return dc::Array4(0);
  }

  dc::Array4 get_prev_error_signals_overlap() const override {
    if (this->distconv_enabled()) {
      return get_prev_activations_overlap();
    } else {
      return dc::Array4(0);
    }
  }

  dc::Array4 get_error_signals_overlap() const override {
    return dc::Array4(0);
  }

  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<dc::Dist, 4>> &dists,      
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(
        dists, invariants, updated, fixed);
    if (this->distconv_enabled()) {
      int stencil_h = (this->m_kernel_dims[2] - 1) / 2;
      int stencil_w = (this->m_kernel_dims[3] - 1) / 2;
      dc::Array4 overlap(0);
      if (this->get_parallel_strategy().width_groups > 1) {
        overlap[0] = stencil_w;
      }
      if (this->get_parallel_strategy().height_groups > 1) {
        overlap[1] = stencil_h;
      }
      auto &prev_activations_dist = dists[this][0];
      prev_activations_dist.set_overlap(overlap);
      updated.insert(&prev_activations_dist);
      fixed.insert(&prev_activations_dist);
      auto &prev_error_signals_dist = dists[this][3];      
      prev_error_signals_dist.set_overlap(overlap);
      updated.insert(&prev_error_signals_dist);
      fixed.insert(&prev_error_signals_dist);
      // To deal with strides, error signals must have the same size
      // of overlap 
      auto &error_signals_dist = dists[this][2];
      error_signals_dist.set_overlap(overlap);
      updated.insert(&error_signals_dist);
      fixed.insert(&error_signals_dist);
    }
  }

  // Deprecated
  dc::Array4 get_strides() const override {
    return dc::Array4({this->m_strides[1], this->m_strides[0], 1, 1});
  }

  dc::Array4 get_activations_tensor_local_shape() const override {
    const int filter_dims[4] = {this->m_kernel_dims[3], this->m_kernel_dims[2],
                                this->m_kernel_dims[1], this->m_kernel_dims[0]};
    const int strides[2] = {this->m_strides[1], this->m_strides[0]};
    const dc::Array4 output_spatial_local_shape =
        ::distconv::get_convolution_output_local_tensor_shape(
            this->m_prev_activations_t,
            filter_dims, strides, true);
    return output_spatial_local_shape;
  }

  void setup_tensors_fwd(const std::array<dc::Dist, 4> &dists) override {
    using namespace dc;
    Layer::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;

    std::stringstream ss;
    util::print_vector(ss, this->m_kernel_dims.begin(), this->m_kernel_dims.end());
    MPIPrintStreamDebug()
        << "m_kernel_dims: " << ss.str() << "\n";
    
    this->setup_prev_activations_tensor(dists);
    this->setup_activations_tensor(dists);
    this->setup_activations_copyout_tensor(dists);    
    
    Array4 kernel_shape = {this->m_kernel_dims[3], this->m_kernel_dims[2],
                           this->m_kernel_dims[1], this->m_kernel_dims[0]};
    const LocaleMPI loc(this->m_comm->get_model_comm().comm, false);
    m_kernel_t = TensorDev(kernel_shape, loc, dc::Dist());
    assert0(tensor::View(
        m_kernel_t, this->get_weights()[0]->get_values().LockedBuffer()));
    m_kernel_gradient_e = TensorDev(kernel_shape, loc, dc::Dist());
    assert0(tensor::View(
        m_kernel_gradient_e, this->m_kernel_gradient.Buffer()));
    
    m_conv = new Convolution(dc::get_backend());

    // Bias tensor. Shared by all procs
    MPIPrintStreamDebug()
        << "Bias desc: "
        << dc::util::tostring(this->m_bias_cudnn_desc)
        << ", bias factor: " << this->m_bias_scaling_factor
        << "\n";
    if (this->m_bias_scaling_factor != DataType(0)) {
      Array4 bias_shape = {1, 1, this->m_neuron_dims[0], 1};
      m_bias_t = TensorDev(bias_shape, loc, dc::Dist());
      assert0(tensor::View(m_bias_t, this->get_weights()[1]->get_values().LockedBuffer()));
      MPIPrintStreamDebug() << "Bias tensor: " << m_bias_t << "\n";
      m_conv->setup_bias(m_bias_t);

      // Bias backprop
      optimizer* bias_optimizer = this->get_weights()[1]->get_optimizer();      
      if (bias_optimizer != nullptr) {
        m_bias_gradient_t = TensorDev(bias_shape, loc, dc::Dist());
        assert0(tensor::View(m_bias_gradient_t,
                             this->m_bias_gradient.Buffer()));
        m_conv->setup_bias_gradient(m_bias_gradient_t);
      }
    }
  }

  void setup_tensors_bwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;    

    this->setup_prev_error_signals_tensor(dists);
    this->setup_error_signals_tensor(dists);
    this->setup_error_signals_copyout_tensor(dists);

    if (getenv("DISTCONV_DETERMINISTIC")) {
      // Same algorithm as LBANN
      m_fwd_algo = "IMPLICIT_GEMM";
      // Deterministic algorithm
      m_bwd_data_algo = "ALGO1";
      m_bwd_filter_algo = "ALGO1";
    }
    
    m_conv->setup(this->m_prev_activations_t,
                  m_kernel_t, this->m_activations_t,
                  this->m_error_signals_t, m_kernel_gradient_e,
                  this->m_prev_error_signals_t,
                  this->m_pads[0], this->m_pads[1],
                  this->m_strides[0], this->m_strides[1],
                  m_fwd_algo, m_bwd_data_algo,
                  m_bwd_filter_algo);
  }
  
 protected:
  dc::Convolution *m_conv;
  dc::TensorDev m_kernel_t;
  dc::TensorDev m_kernel_gradient_e;
  // Bias
  dc::TensorDev m_bias_t;
  dc::TensorDev m_bias_gradient_t;
  // Algorithms
  std::string m_fwd_algo = "DEFAULT";
  std::string m_bwd_data_algo = "DEFAULT";
  std::string m_bwd_filter_algo = "DEFAULT";

  bool using_distconv() const override {
    if (!(this->m_kernel_dims[2] == this->m_kernel_dims[3] &&
          this->m_kernel_dims[2] == this->m_pads[0] * 2 + 1 &&
          this->m_kernel_dims[3] == this->m_pads[1] * 2 + 1)) {
      dc::MPIPrintStreamDebug() << "Unsupported as padding does not match the kernel size\n";
      return false;
    }
    char *env = getenv("DISTCONV_DISABLE");
    if (env) {
      std::string s(env);
      if (s.find(this->get_name()) != std::string::npos) {
        return false;
      }
    }
    return true;
  }
#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
