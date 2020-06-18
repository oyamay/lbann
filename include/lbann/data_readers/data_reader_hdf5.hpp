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
#ifndef LBANN_DATA_READER_HDF5_HPP
#define LBANN_DATA_READER_HDF5_HPP
#include "data_reader_image.hpp"
#include "hdf5.h"
#include "conduit/conduit.hpp"

namespace lbann {
/**
 * Data reader for data stored in hdf5 files will need to assume the file contains x
 */
template <typename TensorDataType>
class hdf5_reader : public generic_data_reader {
 public:
  hdf5_reader(const bool shuffle,
              const std::string key_data,
              const std::string key_label,
              const std::string key_responses,
              const bool hyperslab_labels);
  hdf5_reader(const hdf5_reader&);
  hdf5_reader& operator=(const hdf5_reader&);
  ~hdf5_reader() override {}

  hdf5_reader* copy() const override { return new hdf5_reader(*this); }

  void copy_members(const hdf5_reader& rhs);

  std::string get_type() const override {
    return "data_reader_hdf5_images";
  }
  //void set_input_params(int width, int height, int depth, int num_ch, int num_labels);
  void load() override;
  void set_hdf5_paths(const std::vector<std::string> hdf5_paths) {m_file_paths = hdf5_paths;}

  void set_has_labels(bool b) { m_has_labels = b; }
  void set_has_responses(bool b) { m_has_responses = b; }

  int get_num_labels() const override {
    if(!m_has_labels) {
      return generic_data_reader::get_num_labels();
    }
    return m_num_features; // TODO: Exclude this LiTS-specific hack.
  }
  int get_num_responses() const override {
    if(!m_has_responses) {
      return generic_data_reader::get_num_responses();
    }
    return get_linearized_response_size();
  }
  int get_linearized_data_size() const override {
    return m_num_features;
  }
  int get_linearized_label_size() const override {
    if(!m_has_labels) {
      return generic_data_reader::get_linearized_label_size();
    }
    return m_num_features; // TODO: Exclude this LiTS-specific hack.
  }
  int get_linearized_response_size() const override {
    if(!m_has_responses) {
      return generic_data_reader::get_linearized_response_size();
    }
    return m_num_response_features;
  }
  const std::vector<int> get_data_dims() const override {
    return m_data_dims;
  }
 protected:
  void read_hdf5_hyperslab(hsize_t h_data, hsize_t filespace, int rank,
                           TensorDataType *sample);
  void read_hdf5_sample(int data_id, TensorDataType *sample, TensorDataType *labels);
  //void set_defaults() override;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  void fetch_datum_conduit(Mat& X, int data_id);
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;
  void gather_responses(float *responses);
  hid_t get_hdf5_data_type() const;
  conduit::DataType get_conduit_data_type(conduit::index_t num_elements) const;

  /// Whether to fetch a label from the last column.
  bool m_has_labels = false;
  /// Whether to fetch a response from the last column.
  bool m_has_responses = false;
  int m_image_depth=0;
  size_t m_num_features;
  static constexpr int m_num_response_features = 4; // TODO: Exclude this CosmoFlow-specific parameter.
  float m_all_responses[m_num_response_features];
  std::vector<std::string> m_file_paths;
  MPI_Comm m_comm;
  std::vector<int> m_data_dims;
  std::vector<hsize_t> m_hyperslab_dims;
  hid_t m_fapl;
  hid_t m_dxpl;
  MPI_Comm m_response_gather_comm;
  bool m_use_data_store;
  std::string m_key_data, m_key_labels, m_key_responses;
  bool m_hyperslab_labels;

 private:
  static const std::string HDF5_KEY_DATA, HDF5_KEY_LABELS, HDF5_KEY_RESPONSES;
};
}
#endif // LBANN_DATA_READER_HDF5_HPP
