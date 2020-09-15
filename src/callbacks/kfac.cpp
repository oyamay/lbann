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
//
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <sstream>

#include "lbann/callbacks/kfac.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/utils/im2col.hpp"

#include "cblas.h"
#include "lapacke.h"

namespace lbann {
namespace callback {

void kfac::setup(model *m) {
  const auto v2s =
      [](const std::vector<double> v) {
        std::ostringstream oss;
        for(auto i = v.begin(); i != v.end(); i++) {
          if(i != v.begin())
            oss << ",";
          oss << *i;
        }
        return oss.str();
      };

  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    std::ostringstream oss;
    oss << "K-FAC callback setup:"
        << " damping_act=" << v2s(m_damping_act_params)
        << " damping_err=" << v2s(m_damping_err_params)
        << " damping_bn_act=" << v2s(m_damping_bn_act_params)
        << " damping_bn_err=" << v2s(m_damping_bn_err_params)
        << " damping_warmup_steps=" << m_damping_warmup_steps
        << " kronecker_decay=" << m_kronecker_decay
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac::on_backward_prop_end(model *m) {
  // Using a modified Tikhonov damping tequnique from
  // http://arxiv.org/abs/1811.12019.
  const auto get_next_damping =
      [](const double damping_prev,
         const std::vector<double> damping_params,
         const double damping_warmup_steps) {
        if(damping_params.size() == 1)
          return damping_params[0];
        const DataType alpha = 2.0 * log10(damping_params[0] / damping_params[1]) / damping_warmup_steps;
        return (1.0-alpha) * damping_prev + alpha * damping_params[1];
      };

  m_damping_act = get_next_damping(
      m_damping_act, m_damping_act_params, m_damping_warmup_steps);
  m_damping_err = get_next_damping(
      m_damping_err, m_damping_err_params, m_damping_warmup_steps);
  m_damping_bn_act = get_next_damping(
      m_damping_bn_act, m_damping_bn_act_params, m_damping_warmup_steps);
  m_damping_bn_err = get_next_damping(
      m_damping_bn_err, m_damping_bn_err_params, m_damping_warmup_steps);
}

void kfac::on_epoch_end(model *m) {
  const auto comm = m->get_comm();
  if(comm->am_trainer_master()) {
    const auto& c = static_cast<const sgd_execution_context&>(m->get_execution_context());
    const auto epoch = c.get_epoch();
    std::ostringstream oss;
    oss << "K-FAC callback: changing damping value to "
        << m_damping_act << " (act)"
        << ", " << m_damping_err << " (err)"
        << ", " << m_damping_bn_act << " (bn_act)"
        << ", " << m_damping_bn_err << " (bn_err)"
        << " at " << epoch << " epochs"
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac::on_backward_prop_end(model *m, Layer *l) {
  const auto comm = m->get_comm();
  const auto *l_fc = dynamic_cast<fully_connected_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
  const auto *l_conv = dynamic_cast<convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
  const auto *l_bn = dynamic_cast<batch_normalization_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(l);
  const bool is_fc = (l_fc != nullptr);
  const bool is_conv = (l_conv != nullptr);
  const bool is_bn = (l_bn != nullptr);

  if(is_fc || is_conv || is_bn) {
    // Get the layer ID
    const auto layers = m->get_layers();
    const auto layer_it_in_list = std::find(layers.begin(), layers.end(), l);
    assert(layer_it_in_list != layers.end());
    const size_t layer_id = std::distance(layers.begin(), layer_it_in_list);

    // Get activations, errors, and gradients
    if(l->get_num_parents() != 1 || l->get_num_children() != 1) {
      std::stringstream err;
      err << "The K-FAC callback only supports layers who have exact one parent and child."
          << " layer: " << l->get_name()
          << ", #parent: " << l->get_num_parents()
          << ", #child: " << l->get_num_children();
      LBANN_ERROR(err.str());
    }
    const auto parent = l->get_parent_layers()[0];
    const auto child = l->get_child_layers()[0];
    const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
    const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
    const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
    const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
    const auto mini_batch_size = dtl_parent.get_activations().Width();
    assert(mini_batch_size == dtl_child.get_error_signals().Width());
    const auto local_batch_size = local_activations.Width();

    if(local_activations.GetDevice() != El::Device::GPU
       || local_errors.GetDevice() != El::Device::GPU) {
      std::stringstream err;
      err << "The K-FAC callback only supports GPU layers."
          << " layer: " << l->get_name();
      LBANN_ERROR(err.str());
    }

    if(is_fc || is_conv) {
      if(l->num_weights() != 1) {
        std::stringstream err;
        err << "The K-FAC callback does not currently support biases."
            << " layer: " << l->get_name()
            << ", #weights: " << l->num_weights();
        LBANN_ERROR(err.str());
      }

      auto& weights = l->get_weights(0);
      optimizer *w_optimizer = weights.get_optimizer();
      auto* w_dto = dynamic_cast<data_type_optimizer<DataType>*>(w_optimizer);
      El::Matrix<DataType, El::Device::GPU> w_gradients = w_dto->get_gradient().Matrix();

      // Compute Kronecker factors, assuming that local_errors are
      // already multiplied by 1/N in the loss layer.
      El::Matrix<DataType, El::Device::GPU> A, G;
      if(is_fc) {
        assert(local_activations.Height() == w_gradients.Width());
        assert(local_errors.Height() == w_gradients.Height());
        A = get_kronecker_factor_fc(local_activations, 1.0/mini_batch_size);
        G = get_kronecker_factor_fc(local_errors, mini_batch_size);
      } else {

        const auto input_dims = l->get_input_dims(); // CHW
        const auto output_dims = l->get_output_dims(); // KH'W'

        if(input_dims.size() != 3 && input_dims.size() != 4) {
          std::stringstream err;
          err << "The K-FAC callback only supports 2D or 3D tensors."
              << " layer: " << l->get_name()
              << ", input_dims: ";
          for(auto i = input_dims.begin(); i != input_dims.end(); i++)
            err << (std::distance(input_dims.begin(), i) > 0 ? "," : "") << *i;
          LBANN_ERROR(err.str());
        }

        const size_t num_input_channels = input_dims[0];
        const size_t num_output_channels = output_dims[0];
        size_t spatial_input_prod = 1, spatial_output_prod = 1;
        // std::accumulate might overflow for large 3D layers
        std::vector<int> input_spatial_dims, output_spatial_dims;
        for(auto i = input_dims.begin()+1; i != input_dims.end(); i++) {
          spatial_input_prod *= *i;
          input_spatial_dims.push_back(*i);
        }
        for(auto i = output_dims.begin()+1; i != output_dims.end(); i++) {
          spatial_output_prod *= *i;
          output_spatial_dims.push_back(*i);
        }
        assert((size_t) local_activations.Height() == num_input_channels*spatial_input_prod);
        assert((size_t) local_errors.Height() == num_output_channels*spatial_output_prod);

        A = get_kronecker_factor_conv(
            local_activations, 1.0/mini_batch_size,
            local_batch_size, num_input_channels, input_spatial_dims,
            l_conv, true);
        G = get_kronecker_factor_conv(
            local_errors, DataType(mini_batch_size)/spatial_output_prod,
            local_batch_size, num_output_channels, output_spatial_dims,
            l_conv, false);
      }

      // TODO: Communicate only the lower triangulars
      comm->allreduce((El::AbstractMatrix<DataType>&) A, comm->get_trainer_comm());
      comm->allreduce((El::AbstractMatrix<DataType>&) G, comm->get_trainer_comm());

      // Compute exponential moving average of the factors
      if(m_kronecker_average.find(layer_id) == m_kronecker_average.end())
        m_kronecker_average.emplace(layer_id, std::make_pair(A, G));
      auto& AGave = (*m_kronecker_average.find(layer_id)).second;
      auto& Aave = AGave.first;
      auto& Gave = AGave.second;
      kfac_update_kronecker_average(
          Aave.Buffer(), A.Buffer(), A.Height()*A.Width(), m_kronecker_decay);
      kfac_update_kronecker_average(
          Gave.Buffer(), G.Buffer(), G.Height()*G.Width(), m_kronecker_decay);

      // Compute the pi constant
      const DataType pi = m_use_pi ? compute_pi(Aave, Gave) : 1.0;

      // Compute the inverse of the factors
      const bool print_time = comm->am_trainer_master() && m_print_time;
      // Since setting different damping constants for A and G is an
      // alternative heuristics to pi, they should be the same if pi is used.
      if(m_use_pi && m_damping_act != m_damping_err) {
        std::stringstream err;
        err << "Damping values for activations and errors are different while the pi constant is used."
            << " layer: " << l->get_name()
            << ", m_damping_act: " << m_damping_act
            << ", m_damping_err: " << m_damping_err;
        LBANN_WARNING(err.str());
      }
      const auto Ainv = get_matrix_inverse(Aave, print_time, DataType(m_damping_act*pi));
      const auto Ginv = get_matrix_inverse(Gave, print_time, DataType(m_damping_err/pi));

      if(is_conv) {
        const auto num_output_channels = l->get_output_dims()[0];
        assert(w_gradients.Width() == 1);
        assert((w_gradients.Height()%num_output_channels) == 0);
        const auto height_reshaped = w_gradients.Height()/num_output_channels;
        w_gradients.Attach(height_reshaped,
                           num_output_channels,
                           w_gradients.Buffer(),
                           height_reshaped);
      }

      // Compute preconditioned gradients
      El::Matrix<DataType, El::Device::GPU> Gg(G.Height(), is_conv ? w_gradients.Height() : w_gradients.Width());
      El::Gemm(
          El::NORMAL, is_conv ? El::TRANSPOSE : El::NORMAL,
          El::TypeTraits<DataType>::One(), Ginv, w_gradients,
          El::TypeTraits<DataType>::Zero(), Gg);
      El::Matrix<DataType, El::Device::GPU> Fgrad(G.Height(), A.Width());
      El::Gemm(
          El::NORMAL, El::NORMAL,
          El::TypeTraits<DataType>::One(), Gg, Ainv,
          El::TypeTraits<DataType>::Zero(), Fgrad);

      if(is_conv) {
        Fgrad.Attach(Fgrad.Width()*Fgrad.Height(), 1,
                     Fgrad.Buffer(),
                     Fgrad.Width()*Fgrad.Height());
      } else {
        assert(Fgrad.Height() == w_gradients.Height());
        assert(Fgrad.Width() == w_gradients.Width());
      }

      // Apply preconditioned grads
      DataType dst_scale = El::TypeTraits<DataType>::Zero(),
          gradient_scale = El::TypeTraits<DataType>::One();
      auto& grad_buffer = w_optimizer->get_gradient_buffer(
          dst_scale, gradient_scale, false);
      El::Copy(Fgrad, grad_buffer.Matrix());

      // Damp matrices for debugging
      if(comm->am_trainer_master() && m_print_matrix) {
        if(comm->am_trainer_master()) {
          std::cout << std::endl;
          El::Print(A, "A");
          std::cout << std::endl;
          El::Print(G, "G");
          std::cout << std::endl;
          El::Print(Aave, "Aave");
          std::cout << std::endl;
          El::Print(Gave, "Gave");
          std::cout << std::endl;
          El::Print(Ainv, "Ainv");
          std::cout << std::endl;
          El::Print(Ginv, "Ginv");
          std::cout << std::endl;
          El::Print(w_gradients, "w_grad");
          std::cout << std::endl;
          El::Print(Fgrad, "Fgrad");
          std::cout << std::endl;
        }
      }

      // dump L2 norm of matrices
      if(comm->am_trainer_master() && m_print_matrix_summary) {
        const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
        const auto &w_values = dtw->get_values();
        std::ostringstream oss;
        oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
            << get_matrix_stat(w_values.LockedMatrix(), "W")
            << ", " << get_matrix_stat(local_activations, "acts")
            << ", " << get_matrix_stat(local_errors, "errs")
            << ", " << get_matrix_stat(A, "A")
            << ", " << get_matrix_stat(G, "G")
            << ", " << get_matrix_stat(Aave, "Aave")
            << ", " << get_matrix_stat(Gave, "Gave")
            << ", " << get_matrix_stat(Ainv, "Ainv")
            << ", " << get_matrix_stat(Ginv, "Ginv")
            << ", " << get_matrix_stat(w_gradients, "grad")
            << ", " << get_matrix_stat(Fgrad, "Finvgrad")
            << ", pi=" << pi
            << std::endl;
        std::cout << oss.str();
      }
    } else {
      assert(is_bn);
      const bool is_bn_after_fc =
          (dynamic_cast<const fully_connected_layer<DataType,
           data_layout::DATA_PARALLEL, El::Device::GPU>*>(parent) != nullptr);
      const bool is_bn_after_conv =
          (dynamic_cast<const convolution_layer<DataType,
           data_layout::DATA_PARALLEL, El::Device::GPU>*>(parent) != nullptr);
      if(!is_bn_after_fc && !is_bn_after_conv) {
        std::stringstream err;
        err << "The K-FAC callback only supports batch-normalization layers after "
            << "fully-connected layers or convolutional layers."
            << " layer: " << l->get_name()
            << " parent type: " << parent->get_type();
        LBANN_ERROR(err.str());
      }

      assert(l->num_weights() == 4); // scale, bias, r_mean, r_var
      auto& scales = l->get_weights(0);
      auto& biases = l->get_weights(1);
      optimizer *s_optimizer = scales.get_optimizer();
      optimizer *b_optimizer = biases.get_optimizer();
      auto* s_dto = dynamic_cast<data_type_optimizer<DataType>*>(s_optimizer);
      auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
      El::Matrix<DataType, El::Device::GPU> s_gradients = s_dto->get_gradient().Matrix();
      El::Matrix<DataType, El::Device::GPU> b_gradients = b_dto->get_gradient().Matrix();
      const auto &s_dtw = dynamic_cast<data_type_weights<DataType>*>(&scales);
      const auto &b_dtw = dynamic_cast<data_type_weights<DataType>*>(&biases);
      const auto &scale_values = s_dtw->get_values();
      const auto &bias_values = b_dtw->get_values();

      size_t num_channels;
      size_t spatial_prod;
      if(is_bn_after_fc) {
        num_channels = local_activations.Height();
        spatial_prod = 1;
        assert(num_channels == (size_t) local_errors.Height());
      } else {
        const auto input_dims = l->get_input_dims(); // CHW
        num_channels = input_dims[0];
        spatial_prod = 1;
        // std::accumulate might overflow for large 3D layers
        for(auto i = input_dims.begin()+1; i != input_dims.end(); i++)
          spatial_prod *= *i;
      }

      assert(num_channels == (size_t) scale_values.Height());
      assert(num_channels == (size_t) scale_values.LocalHeight());
      assert(num_channels == (size_t) bias_values.Height());
      assert(num_channels == (size_t) bias_values.LocalHeight());

      El::Matrix<DataType, El::Device::GPU> factor(num_channels*2, local_batch_size);
      kfac_compute_bn_factor(
          local_activations.LockedBuffer(),
          local_errors.LockedBuffer(),
          scale_values.LockedMatrix().LockedBuffer(),
          bias_values.LockedMatrix().LockedBuffer(),
          factor.Buffer(),
          local_batch_size,
          num_channels,
          spatial_prod);

      El::Matrix<DataType, El::Device::GPU> fisher_block(num_channels*2, num_channels*2);
      const DataType alpha = mini_batch_size;
      El::Gemm(
          El::NORMAL, El::TRANSPOSE,
          alpha, factor, factor,
          El::TypeTraits<DataType>::Zero(), fisher_block);
      comm->allreduce((El::AbstractMatrix<DataType>&) fisher_block, comm->get_trainer_comm());

      El::Matrix<DataType, El::Device::GPU> stacked_grads(num_channels*2, 1);
      // TODO: Better way to copy?
      CHECK_CUDA(cudaMemcpy(
          stacked_grads.Buffer(), s_gradients.LockedBuffer(),
          num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice));
      CHECK_CUDA(cudaMemcpy(
          stacked_grads.Buffer()+num_channels, b_gradients.LockedBuffer(),
          num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice));

      const bool print_time = comm->am_trainer_master() && m_print_time;
      const auto Finv = get_matrix_inverse(
          fisher_block, print_time,
          DataType(m_damping_bn_act),
          DataType(m_damping_bn_err),
          true);

      El::Matrix<DataType, El::Device::GPU> Fgrad(num_channels*2, 1);
      El::Gemm(
          El::NORMAL, El::NORMAL,
          El::TypeTraits<DataType>::One(), Finv, stacked_grads,
          El::TypeTraits<DataType>::Zero(), Fgrad);

      DataType dst_scale = El::TypeTraits<DataType>::Zero(),
          gradient_scale = El::TypeTraits<DataType>::One();
      auto& s_grad_buffer = s_optimizer->get_gradient_buffer(
          dst_scale, gradient_scale, false);
      auto& b_grad_buffer = b_optimizer->get_gradient_buffer(
          dst_scale, gradient_scale, false);
      // TODO: Better way to copy?
      CHECK_CUDA(cudaMemcpy(
          s_grad_buffer.Matrix().Buffer(), Fgrad.LockedBuffer(),
          num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice));
      CHECK_CUDA(cudaMemcpy(
          b_grad_buffer.Matrix().Buffer(), Fgrad.LockedBuffer()+num_channels,
          num_channels*sizeof(DataType), cudaMemcpyDeviceToDevice));

      // dump L2 norm of matrices
      if(comm->am_trainer_master() && m_print_matrix_summary) {
        std::ostringstream oss;
        oss << "K-FAC callback: L2 norm @ "<< l->get_name() << ": "
            << get_matrix_stat(scale_values.LockedMatrix(), "scale")
            << ", " << get_matrix_stat(bias_values.LockedMatrix(), "bias")
            << ", " << get_matrix_stat(local_activations, "acts")
            << ", " << get_matrix_stat(local_errors, "errs")
            << ", " << get_matrix_stat(s_gradients, "scale_grad")
            << ", " << get_matrix_stat(b_gradients, "bias_grad")
            << ", " << get_matrix_stat(Fgrad, "Fgrad")
            << std::endl;
        std::cout << oss.str();
      }
    }
  }
}

El::Matrix<DataType, El::Device::GPU> kfac::get_kronecker_factor_fc(
    const El::AbstractMatrix<DataType>& A,
    const DataType alpha) {
  assert(A.GetDevice() == El::Device::GPU);
  El::Matrix<DataType, El::Device::GPU> factor(A.Height(), A.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, A, A,
      El::TypeTraits<DataType>::Zero(), factor);
  return factor;
}

El::Matrix<DataType, El::Device::GPU> kfac::get_kronecker_factor_conv(
    const El::Matrix<DataType, El::Device::GPU>& A,
    const DataType alpha,
    const size_t local_batch_size, const size_t num_channels,
    const std::vector<int> spatial_dims,
    const convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU> *l_conv,
    const bool use_im2col) {
  assert(A.GetDevice() == El::Device::GPU);

  const auto dilations = l_conv->get_dilations();
  for(auto i = dilations.begin(); i != dilations.end(); i++)
    if(*i != 1) {
      std::stringstream err;
      err << "The K-FAC callback onky supports dilation width of 1."
          << " layer: " << l_conv->get_name();
      LBANN_ERROR(err.str());
    }

  // The matrix size will be overwritten later.
  El::Matrix<DataType, El::Device::GPU> Acol(1, 1);
  if(use_im2col) {
    im2col(A, Acol,
           num_channels, spatial_dims.size(),
           &(spatial_dims[0]),
           &(l_conv->get_pads()[0]),
           &(l_conv->get_conv_dims()[0]),
           &(l_conv->get_strides()[0]));
  } else {
    size_t spatial_prod = 1;
    for(auto i = spatial_dims.begin(); i != spatial_dims.end(); i++)
      spatial_prod *= *i;
    Acol.Resize(num_channels, local_batch_size*spatial_prod);
    kfac_conv_transpose(
        A.LockedBuffer(), Acol.Buffer(),
        local_batch_size, num_channels, spatial_prod);
  }

  El::Matrix<DataType, El::Device::GPU> factor(Acol.Height(), Acol.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, Acol, Acol,
      El::TypeTraits<DataType>::Zero(), factor);
  return factor;
}

El::Matrix<DataType, El::Device::GPU> kfac::get_matrix_inverse(
    const El::Matrix<DataType, El::Device::GPU>& A,
    const bool report_time,
    const DataType damping,
    const DataType damping_bn_err,
    const bool is_bn) {
  assert(A.Width() == A.Height());
  El::Matrix<DataType, El::Device::GPU> Ainv(A);

  const double t_start = get_time();

  if(damping > 0 || damping_bn_err > 0)
    kfac_add_to_diagonal(
        Ainv.Buffer(), Ainv.Height(),
        damping, damping_bn_err,
        is_bn);

  const double t_damping = get_time();

  const auto uplo = El::UpperOrLowerNS::LOWER;
  El::Cholesky(
      uplo,
      (El::AbstractMatrix<DataType> &) Ainv);

  const double t_spotrf = get_time();

  // TODO: El::Identity on GPU?
  El::Matrix<DataType, El::Device::GPU> Linv(Ainv.Height(), Ainv.Width());
  El::Zeros(Linv, Linv.Height(), Linv.Width());
  kfac_add_to_diagonal(Linv.Buffer(), Linv.Height(), DataType(1.0));

  El::Trsm(
      El::LeftOrRightNS::LEFT,
      uplo,
      El::OrientationNS::NORMAL,
      El::UnitOrNonUnitNS::NON_UNIT,
      El::TypeTraits<DataType>::One(),
      (const El::AbstractMatrix<DataType> &) Ainv,
      (El::AbstractMatrix<DataType> &) Linv,
      true);

  El::Gemm(
      El::TRANSPOSE, El::NORMAL,
      El::TypeTraits<DataType>::One(), Linv, Linv,
      El::TypeTraits<DataType>::Zero(), Ainv);

  const double t_spotri = get_time();

  // TRSM+GEMM is equivalent to POTRI+fill_upper_tri.
  // kfac_fill_upper_tri(Ainv.Buffer(), Ainv.Height());

  const double t_fill = get_time();

  if(report_time) {
    std::cout << "K-FAC callback: get_matrix_inverse of"
              << " " << A.Height() << "x" << A.Width()
              << " using Hydrogen"
              << " (damping=" << damping << "): "
              << " t_damping=" << (t_damping-t_start)
              << ", t_spotrf=" << (t_spotrf-t_damping)
              << ", t_spotri=" << (t_spotri-t_spotrf)
              << ", t_fill=" << (t_fill-t_spotri)
              << std::endl;
  }

  return Ainv;
}

double kfac::compute_pi(const El::Matrix<DataType, El::Device::GPU>& A,
                        const El::Matrix<DataType, El::Device::GPU>& G) {
  // TODO: El::Trace is defined but not implemented yet.
  const auto get_trace =
      [](const El::Matrix<DataType, El::Device::GPU> X) {
        const El::Matrix<DataType> XCPU(X);
        DataType s = 0.0;
        for(int i = 0; i < XCPU.Height(); i++)
          s += XCPU(i, i);
        return (double) s;
      };
  return sqrt((get_trace(A)/A.Height())/(get_trace(G)/G.Height()));
}

std::string kfac::get_matrix_stat(const El::Matrix<DataType, El::Device::GPU>& X,
                                  const char *name) {
  El::Matrix<DataType> XCPU(X);
  const auto nrm2 = El::Nrm2(El::Reshape(XCPU.Height()*XCPU.Width(), 1, XCPU));
  std::ostringstream oss;
  oss << name
      << "("
      << X.Height()
      << "x"
      << X.Width()
      << ")="
      << std::setprecision(2)
      << std::scientific
      << nrm2;
  return oss.str();
}

std::unique_ptr<callback_base>
build_kfac_callback_from_pbuf(
    const google::protobuf::Message& proto_msg,
    const std::shared_ptr<lbann_summary>&) {
  using MsgType = lbann_data::Callback::CallbackKFACTest;
  using CallbackType = kfac;
  const auto& params = dynamic_cast<const MsgType&>(proto_msg);

  const auto parse_damping_params =
      [](const std::string str) {
        if(str == "")
          return std::vector<double>({kfac::damping_0_default});
        else
          return parse_list<double>(str);
      };

  const std::vector<double> damping_act_params = parse_damping_params(params.damping_act());
  const std::vector<double> damping_err_params = parse_damping_params(params.damping_err());
  const std::vector<double> damping_bn_act_params = parse_damping_params(params.damping_bn_act());
  const std::vector<double> damping_bn_err_params = parse_damping_params(params.damping_bn_err());
  double damping_warmup_steps = params.damping_warmup_steps();
  if(damping_warmup_steps == 0.0) damping_warmup_steps = kfac::damping_warmup_steps_default;
  double kronecker_decay = params.kronecker_decay();
  if(kronecker_decay == 0.0)
    kronecker_decay = kfac::kronecker_decay_default;
  const bool print_time = params.print_time();
  const bool print_matrix = params.print_matrix();
  const bool print_matrix_summary = params.print_matrix_summary();
  const bool use_pi = params.use_pi();
  return make_unique<CallbackType>(
      damping_act_params,
      damping_err_params,
      damping_bn_act_params,
      damping_bn_err_params,
      damping_warmup_steps,
      kronecker_decay,
      print_time, print_matrix, print_matrix_summary,
      use_pi);
}

} // namespace callback
} // namespace lbann