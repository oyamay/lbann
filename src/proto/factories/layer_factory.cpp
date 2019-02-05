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

#include "lbann/proto/factories.hpp"
#include "lbann/utils/peek_map.hpp"

namespace lbann {
namespace proto {

#define LAYOUT_ERR(layer_name, layer_type) \
  { \
    std::stringstream s;  \
    s << "\nlayer type: " << layer_type << " layer name: " << layer_name << " -- is only supported for data_layout::DATA_PARALLEL";                  \
    LBANN_ERROR(s.str()); \
  }

#define DEVICE_ERR(layer_name, layer_type, layout, Dev) \
  { \
    if (layout != data_layout::DATA_PARALLEL) { \
      LAYOUT_ERR(layer_name, layer_type)  \
    } else if (Dev != El::Device::CPU) { \
      std::stringstream s;  \
      s << "\nlayer type: " << layer_type " layer name: " << layer_name << " -- is only supported for El::Device::CPU; it looks like you're attempting to run with a cuda build. You should be able to run by adding --disable_cuda to your command line (in which case you won't be using GPUs, which may not be what you want)";\
      LBANN_ERROR(s.str()); \
    } else {     \
      std::stringstream s;  \
      s << "\nsomething is weird with data_layout and/or El::Device but we can't determine what."; \
      LBANN_ERROR(s.str()); \
    } \
  }

std::vector<El::Int> get_slice_points_from_reader(const generic_data_reader* dr,
                                                  const std::string& var_category,
                                                  bool& is_supported);

template <data_layout layout, El::Device Dev>
Layer* construct_layer(lbann_comm* comm,
                       const std::map<execution_mode, generic_data_reader*>& data_readers,
                       int num_parallel_readers,
                       const lbann_data::Layer& proto_layer) {
  std::stringstream err;

  // Convenience macro to construct layers with no parameters
#define CONSTRUCT_LAYER(name)                           \
  do {                                                  \
    if (proto_layer.has_##name()) {                     \
      return new name##_layer<layout, Dev>(comm);       \
    }                                                   \
  } while (false)

  // Input layers
  if (proto_layer.has_input()) {
    const auto& params = proto_layer.input();
    const auto& io_buffer = params.io_buffer();
    const auto& mode_str = params.target_mode();
    data_reader_target_mode target_mode = data_reader_target_mode::CLASSIFICATION;
    if (mode_str.empty() || mode_str == "classification") { target_mode = data_reader_target_mode::CLASSIFICATION; }
    if (mode_str == "regression")                         { target_mode = data_reader_target_mode::REGRESSION; }
    if (mode_str == "reconstruction")                     { target_mode = data_reader_target_mode::RECONSTRUCTION; }
    if (mode_str == "na" || mode_str == "NA" || mode_str == "N/A") { target_mode = data_reader_target_mode::NA; }
    if (io_buffer == "partitioned") {
      return new input_layer<partitioned_io_buffer, layout, Dev>(comm,
                                                                 num_parallel_readers,
                                                                 data_readers,
                                                                 !params.data_set_per_model(),
                                                                 target_mode);
    }
  }

  // Fully connected layer
  if (proto_layer.has_fully_connected()) {
    const auto& params = proto_layer.fully_connected();
    int num_neurons = 0;
    std::string num_neurons_method_name;

    if (params.get_input_dimension_from_reader()
        || params.get_image_dimension_from_reader()
        || params.get_scalar_dimension_from_reader()
        || params.get_image_and_scalar_dimension_from_reader()) {
      num_neurons_method_name = "get_*_dimension_from_reader";
    #if defined(LBANN_HAS_CONDUIT)
      const auto dr_generic  = lbann::peek_map(data_readers, execution_mode::training);
      const auto dr = dynamic_cast<lbann::data_reader_jag_conduit_hdf5*>(dr_generic);
      if (dr != nullptr) {
        size_t input_dim = dr->get_linearized_input_size();
        size_t scalar_dim = dr->get_linearized_scalar_size();
        size_t image_dim = dr->get_linearized_channel_size() * dr->get_num_channels();
        size_t num_images = dr->get_num_img_srcs();

        if (params.get_input_dimension_from_reader()) {
          num_neurons += input_dim;
        }
        if (params.get_image_dimension_from_reader()) {
          num_neurons += (num_images * image_dim);
        }
        if (params.get_scalar_dimension_from_reader()) {
          num_neurons += scalar_dim;
        }
        if (params.get_image_and_scalar_dimension_from_reader()) {
          num_neurons += (num_images * image_dim + scalar_dim);
        }
      }
    #endif // defined(LBANN_HAS_CONDUIT)
    } else if (params.get_num_neurons_of_slice_from_reader_size() > 0) {
      num_neurons_method_name = "get_num_neurons_of_slice_from_reader";
    #if defined(LBANN_HAS_CONDUIT)
      const auto dr_generic  = lbann::peek_map(data_readers, execution_mode::training);
      const int num_slice_indices = params.get_num_neurons_of_slice_from_reader_size();
      if (dynamic_cast<lbann::data_reader_jag_conduit*>(dr_generic) != nullptr) {
        const std::string& var = params.get_slice_points_from_reader();
        bool is_supported = false; /// @todo Remove unneeded function parameter
        const auto slice_points = get_slice_points_from_reader(dr_generic, var, is_supported);
        for (int i = 0; i < num_slice_indices; ++i) {
          const size_t idx = static_cast<size_t>(params.get_num_neurons_of_slice_from_reader(i));
          if ((idx == 0u) || (idx >= slice_points.size())) {
            err << "invalid slice index from get_num_neurons_of_slice_from_reader";
            LBANN_ERROR(err.str());
          }
          const int diff = static_cast<int>(slice_points[idx] - slice_points[idx-1]);
          num_neurons += diff;
        }
      }
    #endif // defined(LBANN_HAS_CONDUIT)
    } else {
      num_neurons_method_name = "num_neurons";
      num_neurons = params.num_neurons();
      if (proto_layer.num_neurons_from_data_reader()) {
        const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
        if (!dr) {
          LBANN_ERROR("training data reader does not exist!");
        }
        num_neurons = dr->get_linearized_data_size();
      }
    }
    return new fully_connected_layer<layout, Dev>(comm,
                                                  num_neurons,
                                                  params.transpose(),
                                                  nullptr,
                                                  params.has_bias());
  }

  // Convolution and deconvolution layer
  if (proto_layer.has_convolution()) {
    const auto& params = proto_layer.convolution();
    const auto& num_output_channels = params.num_output_channels();
    const auto& bias = params.has_bias();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, dilations, num_groups, bias
                   );
      }
      LAYOUT_ERR(proto_layer.name(), "convolution");
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
      if (layout == data_layout::DATA_PARALLEL) {
        return new convolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, dilation, num_groups, bias
                   );
      }
      LAYOUT_ERR(proto_layer.name(), "convolution");
    }
  }
  if (proto_layer.has_deconvolution()) {
    const auto& params = proto_layer.deconvolution();
    const auto& bias = params.has_bias();
    int num_output_channels = params.num_output_channels();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
      if (!dr) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      num_output_channels = dr->get_linearized_data_size();
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, dims.size(), num_output_channels,
                     dims, pads, strides, dilations, num_groups, bias
                   );
      }
      LAYOUT_ERR(proto_layer.name(), "deconvolution");
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
      if (layout == data_layout::DATA_PARALLEL) {
        return new deconvolution_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, num_dims, num_output_channels,
                     dim, pad, stride, dilation, num_groups, bias
                   );
      }
      LAYOUT_ERR(proto_layer.name(), "deconvolution");
    }
  }

  // Transform layers
  if (proto_layer.has_reshape()) {
    const auto& params = proto_layer.reshape();
    std::vector<int> dims = parse_list<int>(params.dims());
    if (params.num_dims() != 0) {
      LBANN_WARNING("found unused and deprecated prototext field (Reshape.num_dims)");
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      dims.clear();
      const auto dr  = lbann::peek_map(data_readers, execution_mode::training);
      if (!dr) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      dims.push_back(dr->get_linearized_data_size());
    }
    return new reshape_layer<layout, Dev>(comm, dims);
  }
  if (proto_layer.has_sum()) {
    return new sum_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_weighted_sum()) {
    const auto& params = proto_layer.weighted_sum();
    const auto& scaling_factors = parse_list<DataType>(params.scaling_factors());
    return new weighted_sum_layer<layout, Dev>(comm, scaling_factors);
  }
  if (proto_layer.has_split()) {
    return new split_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_concatenation()) {
    const auto& axis = proto_layer.concatenation().concatenation_axis();
    return new concatenation_layer<layout, Dev>(comm, axis);
  }
  if (proto_layer.has_slice()) {
    const auto& params = proto_layer.slice();
    std::vector<El::Int> slice_points;
    bool is_supported = false;
    std::string slice_point_method_name;

    if (params.get_slice_points_from_reader_bool()) {
      slice_point_method_name = "'get_slice_points_from_reader_bool'";
    #if defined(LBANN_HAS_CONDUIT)
      size_t total = 0;
      slice_points.push_back(total);
      const auto dr_generic  = lbann::peek_map(data_readers, execution_mode::training);
      if (dynamic_cast<lbann::data_reader_jag_conduit_hdf5*>(dr_generic) != nullptr) {
        is_supported = true;
        const auto dr1  = lbann::peek_map(data_readers, execution_mode::training);
        lbann::data_reader_jag_conduit_hdf5 *dr = dynamic_cast<lbann::data_reader_jag_conduit_hdf5*>(dr1);
        total += dr->get_num_img_srcs() * dr->get_linearized_channel_size() * dr->get_num_channels()
              + dr->get_linearized_scalar_size();
        slice_points.push_back(total);
        total += dr->get_linearized_input_size();
        slice_points.push_back(total);
      }
    #endif // defined(LBANN_HAS_CONDUIT)
    } else if (params.get_slice_points_from_reader() != "") {
      slice_point_method_name = "'get_slice_points_from_reader'";
    #if defined(LBANN_HAS_CONDUIT)
      const auto dr_generic  = lbann::peek_map(data_readers, execution_mode::training);
      const std::string& var = params.get_slice_points_from_reader();
      slice_points = get_slice_points_from_reader(dr_generic, var, is_supported);
    #endif // defined(LBANN_HAS_CONDUIT)
    } else {
      slice_point_method_name = "'slice_points'";
      slice_points = parse_list<El::Int>(params.slice_points());
      is_supported = true;
    }
    if (slice_points.size() < 2u) {
      if (is_supported) {
        err << "Failed to get slice points via " << slice_point_method_name << '.';
      } else {
        err << slice_point_method_name << " is not supported by the reader.";
      }
      LBANN_ERROR(err.str());
      return nullptr;
    }
    return  new slice_layer<layout, Dev>(comm,
                                         params.slice_axis(),
                                         slice_points);
  }
  if (proto_layer.has_hadamard()) {
    return new hadamard_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_constant()) {
    const auto& params = proto_layer.constant();
    const auto& dims = parse_list<int>(params.num_neurons());
    return new constant_layer<layout, Dev>(comm, params.value(), dims);
  }
  if (proto_layer.has_gaussian()) {
    const auto& params = proto_layer.gaussian();
    const auto& dims = parse_list<int>(params.neuron_dims());
    if (params.mean() == 0 && params.stdev() == 0) {
      return new gaussian_layer<layout, Dev>(comm, dims);
    } else {
      return new gaussian_layer<layout, Dev>(comm,
                                             dims,
                                             params.mean(),
                                             params.stdev());
    }
  }
  if (proto_layer.has_bernoulli()) {
    const auto& params = proto_layer.bernoulli();
    const auto& dims = parse_list<int>(params.neuron_dims());
    return new bernoulli_layer<layout, Dev>(comm,
                                            dims,
                                            params.prob());
  }
  if (proto_layer.has_uniform()) {
    const auto& params = proto_layer.uniform();
    const auto& dims = parse_list<int>(params.neuron_dims());
    if (params.min() == 0 && params.max() == 0) {
      return new uniform_layer<layout, Dev>(comm, dims);
    } else {
      return new uniform_layer<layout, Dev>(comm, dims, params.min(), params.max());
    }
  }
  if (proto_layer.has_zero()) {
    const auto& params = proto_layer.zero();
    return new zero_layer<layout>(comm, params.first_half(), params.second_half());
  }
  if (proto_layer.has_pooling()) {
    const auto& params = proto_layer.pooling();
    const auto& mode_str = params.pool_mode();
    pool_mode mode = pool_mode::invalid;
    if (mode_str == "max" )            { mode = pool_mode::max; }
    if (mode_str == "average" )        { mode = pool_mode::average; }
    if (mode_str == "average_no_pad" ) { mode = pool_mode::average_no_pad; }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.pool_dims());
      const auto& pads = parse_list<int>(params.pool_pads());
      const auto& strides = parse_list<int>(params.pool_strides());
      if (layout == data_layout::DATA_PARALLEL) {
        return new pooling_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, dims.size(), dims, pads, strides, mode
                   );
      }
      LAYOUT_ERR(proto_layer.name(), "pooling");
    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.pool_dims_i();
      const auto& pad = params.pool_pads_i();
      const auto& stride = params.pool_strides_i();
      if (layout == data_layout::DATA_PARALLEL) {
        return new pooling_layer<data_layout::DATA_PARALLEL, Dev>(
                     comm, num_dims, dim, pad, stride, mode
                   );
      }
      LAYOUT_ERR(proto_layer.name(), "pooling");
    }
  }
  if (proto_layer.has_unpooling()) {
    if (layout == data_layout::DATA_PARALLEL && Dev == El::Device::CPU) {
      return new unpooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(comm);
    }
    DEVICE_ERR(proto_layer.name(), "unpooling", layout, Dev);
  }
  if (proto_layer.has_reduction()) {
    const auto& params = proto_layer.reduction();
    const auto& mode_str = params.mode();
    reduction_mode mode = reduction_mode::INVALID;
    if (mode_str == "sum" || mode_str.empty()) { mode = reduction_mode::SUM; }
    if (mode_str == "average") { mode = reduction_mode::AVERAGE; }
    if (layout == data_layout::DATA_PARALLEL) {
      return new reduction_layer<data_layout::DATA_PARALLEL, Dev>(comm, mode);
    }
    LAYOUT_ERR(proto_layer.name(), "reduction");
  }
  if (proto_layer.has_evaluation()) {
    return new evaluation_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_crop()) {
    const auto& params = proto_layer.crop();
    const auto& dims = parse_list<int>(params.dims());
    if (layout == data_layout::DATA_PARALLEL) {
      return new crop_layer<data_layout::DATA_PARALLEL, Dev>(comm, dims);
    }
    LAYOUT_ERR(proto_layer.name(), "crop");
  }
  if (proto_layer.has_categorical_random()) {
    if (layout == data_layout::DATA_PARALLEL
        && Dev == El::Device::CPU) {
      return new categorical_random_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(comm);
    }
    DEVICE_ERR(proto_layer.name(), "categorical_random", layout, Dev);
  }
  if (proto_layer.has_discrete_random()) {
    const auto& params = proto_layer.discrete_random();
    const auto& values = parse_list<DataType>(params.values());
    const auto& dims = parse_list<int>(params.dims());
    if (layout == data_layout::DATA_PARALLEL
        && Dev == El::Device::CPU) {
      return new discrete_random_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                   comm, values, dims);
    }
    DEVICE_ERR(proto_layer.name(), "discrete_random", layout, Dev);
  }
  if (proto_layer.has_dummy()) {
    return new dummy_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_stop_gradient()) {
    return new stop_gradient_layer<layout, Dev>(comm);
  }
  if (proto_layer.has_in_top_k()) {
    const auto& params = proto_layer.in_top_k();
    return new in_top_k_layer<layout, Dev>(comm, params.k());
  }
  if (proto_layer.has_sort()) {
    const auto& params = proto_layer.sort();
    if (layout == data_layout::DATA_PARALLEL) {
      return new sort_layer<data_layout::DATA_PARALLEL, Dev>(comm, params.descending());
    }
    LAYOUT_ERR(proto_layer.name(), "sort");
  }
  if (proto_layer.has_weights_layer()) {
    const auto& params = proto_layer.weights_layer();
    const auto& dims = parse_list<El::Int>(params.dims());
    return new weights_layer<layout, Dev>(comm, dims);
  }
  if (proto_layer.has_tessellate()) {
    const auto& params = proto_layer.tessellate();
    const auto& dims = parse_list<int>(params.dims());
    return new tessellate_layer<layout, Dev>(comm, dims);
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& params = proto_layer.batch_normalization();
    if (layout == data_layout::DATA_PARALLEL) {
      const auto& aggr_str = params.stats_aggregation();
      batch_normalization_stats_aggregation aggr =
        batch_normalization_stats_aggregation::local;
      if (aggr_str == "local" || aggr_str.empty()) {
        aggr = batch_normalization_stats_aggregation::local;
      } else if (aggr_str == "node_local") {
        aggr = batch_normalization_stats_aggregation::node_local;
      } else if (aggr_str == "global") {
        aggr = batch_normalization_stats_aggregation::global;
      } else {
        err << "Invalid batch normalization stats aggregation " << aggr_str;
        LBANN_ERROR(err.str());
        return nullptr;
      }
      // Set defaults if not given.
      auto decay = params.decay();
      if (decay == 0.0) {
        decay = 0.9;
      }
      auto epsilon = params.epsilon();
      if (epsilon == 0.0) {
        epsilon = 1e-5;
      }
      return new batch_normalization_layer<data_layout::DATA_PARALLEL, Dev>(
        comm,
        decay,
        epsilon,
        aggr);
    }
    LAYOUT_ERR(proto_layer.name(), "batch_normalization");
  }
  if (proto_layer.has_dropout()) {
    const auto& params = proto_layer.dropout();
    return new dropout<layout, Dev>(comm, params.keep_prob());
  }
  if (proto_layer.has_local_response_normalization()) {
 const auto& params = proto_layer.local_response_normalization();
    if (layout == data_layout::DATA_PARALLEL) {
      return new local_response_normalization_layer<data_layout::DATA_PARALLEL, Dev>(
             comm,
             params.window_width(),
             params.lrn_alpha(),
             params.lrn_beta(),
             params.lrn_k());
    }
    LAYOUT_ERR(proto_layer.name(), "local_response_normalization");
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& params = proto_layer.selu_dropout();
    const auto& keep_prob = params.keep_prob();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return new selu_dropout<layout, Dev>(comm, keep_prob, alpha, scale);
    } else {
      return new selu_dropout<layout, Dev>(comm, keep_prob);
    }
  }

  // Math layers
  CONSTRUCT_LAYER(logical_not);
  CONSTRUCT_LAYER(abs);
  CONSTRUCT_LAYER(negative);
  CONSTRUCT_LAYER(sign);
  CONSTRUCT_LAYER(round);
  CONSTRUCT_LAYER(ceil);
  CONSTRUCT_LAYER(floor);
  CONSTRUCT_LAYER(reciprocal);
  CONSTRUCT_LAYER(square);
  CONSTRUCT_LAYER(sqrt);
  CONSTRUCT_LAYER(rsqrt);
  CONSTRUCT_LAYER(safe_reciprocal);
  CONSTRUCT_LAYER(exp);
  CONSTRUCT_LAYER(expm1);
  CONSTRUCT_LAYER(log);
  CONSTRUCT_LAYER(log1p);
  CONSTRUCT_LAYER(cos);
  CONSTRUCT_LAYER(sin);
  CONSTRUCT_LAYER(tan);
  CONSTRUCT_LAYER(acos);
  CONSTRUCT_LAYER(asin);
  CONSTRUCT_LAYER(atan);
  CONSTRUCT_LAYER(cosh);
  CONSTRUCT_LAYER(sinh);
  CONSTRUCT_LAYER(tanh);
  CONSTRUCT_LAYER(acosh);
  CONSTRUCT_LAYER(asinh);
  CONSTRUCT_LAYER(atanh);
  CONSTRUCT_LAYER(add);
  CONSTRUCT_LAYER(subtract);
  CONSTRUCT_LAYER(multiply);
  CONSTRUCT_LAYER(divide);
  CONSTRUCT_LAYER(mod);
  CONSTRUCT_LAYER(pow);
  CONSTRUCT_LAYER(safe_divide);
  CONSTRUCT_LAYER(squared_difference);
  CONSTRUCT_LAYER(max);
  CONSTRUCT_LAYER(min);
  CONSTRUCT_LAYER(equal);
  CONSTRUCT_LAYER(not_equal);
  CONSTRUCT_LAYER(less);
  CONSTRUCT_LAYER(less_equal);
  CONSTRUCT_LAYER(greater);
  CONSTRUCT_LAYER(greater_equal);
  CONSTRUCT_LAYER(logical_and);
  CONSTRUCT_LAYER(logical_or);
  CONSTRUCT_LAYER(logical_xor);
  if (proto_layer.has_clamp()) {
    const auto& params = proto_layer.clamp();
    return new clamp_layer<layout, Dev>(comm, params.min(), params.max());
  }

  // Activation layers
  if (proto_layer.has_elu()) {
    const auto& params = proto_layer.elu();
    const auto& alpha = params.alpha();
    if (alpha != 0) {
      return new elu_layer<layout, Dev>(comm, alpha);
    } else {
      return new elu_layer<layout, Dev>(comm);
    }
  }
  CONSTRUCT_LAYER(identity);
  if (proto_layer.has_leaky_relu()) {
    const auto& params = proto_layer.leaky_relu();
    const auto& negative_slope = params.negative_slope();
    if (negative_slope != 0) {
      return new leaky_relu_layer<layout, Dev>(comm, negative_slope);
    } else {
      return new leaky_relu_layer<layout, Dev>(comm);
    }
  }
  CONSTRUCT_LAYER(log_sigmoid);
  CONSTRUCT_LAYER(log_softmax);
  CONSTRUCT_LAYER(relu);
  CONSTRUCT_LAYER(selu);
  CONSTRUCT_LAYER(sigmoid);
  CONSTRUCT_LAYER(softmax);
  CONSTRUCT_LAYER(softplus);
  CONSTRUCT_LAYER(softsign);

  // Loss layers
  CONSTRUCT_LAYER(categorical_accuracy);
  CONSTRUCT_LAYER(cross_entropy);
  CONSTRUCT_LAYER(mean_squared_error);
  CONSTRUCT_LAYER(mean_absolute_error);
  if (proto_layer.has_top_k_categorical_accuracy()) {
    const auto& params = proto_layer.top_k_categorical_accuracy();
    return new top_k_categorical_accuracy_layer<layout, Dev>(comm, params.k());
  }
  CONSTRUCT_LAYER(l2_norm2);
  CONSTRUCT_LAYER(l1_norm);
  CONSTRUCT_LAYER(binary_cross_entropy);
  CONSTRUCT_LAYER(sigmoid_binary_cross_entropy);
  CONSTRUCT_LAYER(boolean_accuracy);
  CONSTRUCT_LAYER(boolean_false_negative);
  CONSTRUCT_LAYER(boolean_false_positive);

  // Image layers
  if (proto_layer.has_bilinear_resize()) {
    const auto& params = proto_layer.bilinear_resize();
    if (layout == data_layout::DATA_PARALLEL) {
      return new bilinear_resize_layer<data_layout::DATA_PARALLEL, Dev>(
                         comm,
                         params.height(),
                         params.width());
    }
    LAYOUT_ERR(proto_layer.name(), "bilinear_resize");
  }

  // Miscellaneous layers
  if (proto_layer.has_covariance()) {
    const auto& params = proto_layer.covariance();
    return new covariance_layer<layout, Dev>(comm, params.biased());
  }
  if (proto_layer.has_variance()) {
    const auto& params = proto_layer.variance();
    return new variance_layer<layout, Dev>(comm, params.biased());
  }
  if (proto_layer.has_channelwise_mean()) {
    if (layout == data_layout::DATA_PARALLEL) {
      return new channelwise_mean_layer<data_layout::DATA_PARALLEL, Dev>(comm);
    }
    LAYOUT_ERR(proto_layer.name(), "channelwise_mean");
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(err.str());
  return nullptr;

}

// Template instantiation
template Layer* construct_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template Layer* construct_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#ifdef LBANN_HAS_GPU
template Layer* construct_layer<data_layout::DATA_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
template Layer* construct_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>(
  lbann_comm* comm,
  const std::map<execution_mode, generic_data_reader*>& data_readers,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer
);
#endif // LBANN_HAS_GPU

/// Obtain the slice points from the data reader
std::vector<El::Int> get_slice_points_from_reader(const generic_data_reader* dr_generic,
                                                  const std::string& var_category,
                                                  bool& is_supported) {
  std::vector<El::Int> slice_points;
  is_supported = false;
#if defined(LBANN_HAS_CONDUIT)
  // TODO: remove the dynamic cast when this feature gets merged into the base class
  const auto dr = dynamic_cast<const data_reader_jag_conduit*>(dr_generic);

  if (dr != nullptr) {
    is_supported = true;
    if (var_category == "independent") {
      slice_points = dr->get_slice_points_independent();
    } else if (var_category == "dependent") {
      slice_points = dr->get_slice_points_independent();
    } else {
      LBANN_ERROR("Unknown variable category \"" + var_category \
                  + "\". Must be either \"independent\" or \"dependent\".");
    }
  }
#endif
  return slice_points;
}

} // namespace proto
} // namespace lbann
