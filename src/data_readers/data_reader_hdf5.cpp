////////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//// Produced at the Lawrence Livermore National Laboratory.
//// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
//// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
////
//// LLNL-CODE-697807.
//// All rights reserved.
////
//// This file is part of LBANN: Livermore Big Artificial Neural Network
//// Toolkit. For details, see http://software.llnl.gov/LBANN or
//// https://github.com/LLNL/LBANN.
////
//// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
//// may not use this file except in compliance with the License.  You may
//// obtain a copy of the License at:
////
//// http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//// implied. See the License for the specific language governing
//// permissions and limitations under the license.
////
///////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_hdf5.hpp"
#include "lbann/utils/profiling.hpp"
#include <cstdio>
#include <string>
#include <fstream>
#include <unordered_set>
#include <iostream>
#include <cstring>
#include "lbann/utils/distconv.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"

namespace {
inline hid_t check_hdf5(hid_t hid, const char *file, int line) {
  if (hid < 0) {
    std::cerr << "HDF5 error" << std::endl;
    std::cerr << "Error at " << file << ":" << line << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return hid;
}
}

#define CHECK_HDF5(call) check_hdf5(call, __FILE__, __LINE__)

namespace lbann {
  const std::string hdf5_reader::HDF5_KEY_DATA = "full";
  const std::string hdf5_reader::HDF5_KEY_RESPONSES = "unitPar";

  hdf5_reader::hdf5_reader(const bool shuffle)
    : generic_data_reader(shuffle) {}

hdf5_reader::hdf5_reader(const hdf5_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

hdf5_reader& hdf5_reader::operator=(const hdf5_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void hdf5_reader::copy_members(const hdf5_reader &rhs) {
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);

  m_has_labels = rhs.m_has_labels;
  m_has_responses = rhs.m_has_responses;
  m_num_features = rhs.m_num_features;
  m_num_response_features = rhs.m_num_response_features;
  m_data_dims = rhs.m_data_dims;
  m_comm = rhs.m_comm;
  m_data_dims = rhs.m_data_dims;
  m_scaling_factor_int16 = rhs.m_scaling_factor_int16;
  m_file_paths = rhs.m_file_paths;

  for(size_t i = 0; i < 4 /*rhs.m_all_responses.size()*/; i++) {
    m_all_responses[i] = rhs.m_all_responses[i];
  }
}

  void hdf5_reader::read_hdf5_hyperslab(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims, conduit::Node& sample) {
    // this is the splits, right now it is hard coded to split along the z axis
    int num_io_parts = dc::get_number_of_io_partitions();
    int ylines = 1;
    int xlines = 1;
    int zlines = num_io_parts;
    int channellines = 1;

    hsize_t xPerNode = dims[3]/xlines;
    hsize_t yPerNode = dims[2]/ylines;
    hsize_t zPerNode = dims[1]/zlines;
    hsize_t cPerNode = dims[0]/channellines;
    // how many times the pattern should repeat in the hyperslab
    hsize_t count[4] = {1,1,1,1};
    // local dimensions aka the dimensions of the slab we will read in
    hsize_t dims_local[4] = {cPerNode, zPerNode, yPerNode, xPerNode};

    // necessary for the hdf5 lib
    hid_t memspace = H5Screate_simple(4, dims_local, NULL);
    int spatial_offset = rank%num_io_parts;

    hsize_t offset[4] = {0, zPerNode*spatial_offset, 0, 0};

    // from an explanation of the hdf5 select_hyperslab:
    // start -> a starting location for the hyperslab
    // stride -> the number of elements to separate each element or block to be selected
    // count -> the number of elemenets or blocks to select along each dimension
    // block -> the size of the block selected from the dataspace
    //hsize_t status;

    //todo add error checking
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, dims_local);
    sample.set(conduit::DataType::uint16(xPerNode * yPerNode * zPerNode * cPerNode));

    unsigned short* buf = sample.value();
    H5Dread(h_data, H5T_NATIVE_SHORT, memspace, filespace, H5P_DEFAULT, buf);
  }

  void hdf5_reader::read_hdf5_sample(int data_id, conduit::Node& sample) {
    int world_rank = get_rank_in_world();
    //int world_rank = dc::get_input_rank(*get_comm()); // Should probably be trainer rank
    auto file = m_file_paths[data_id];
    hid_t h_file = H5Fopen(file.c_str(), H5F_ACC_RDONLY, m_fapl);

#if 0
    dc::MPIPrintStreamInfo() << "HDF5 file opened: "
                             << file;
#endif

    if (h_file < 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                            " hdf5_reader::load() - can't open file : " + file);
    }

    // load in dataset
    hid_t h_data = CHECK_HDF5(
                              H5Dopen(h_file, HDF5_KEY_DATA.c_str(), H5P_DEFAULT));
    hid_t filespace = CHECK_HDF5(H5Dget_space(h_data));
    //get the number of dimesnionse from the dataset
    int rank1 = H5Sget_simple_extent_ndims(filespace);
    hsize_t dims[rank1];
    // read in what the dimensions are
    CHECK_HDF5(H5Sget_simple_extent_dims(filespace, dims, NULL));

    if (h_data < 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                            " hdf5_reader::load() - can't find hdf5 key : " + HDF5_KEY_DATA);
    }

    const std::string conduit_obj = LBANN_DATA_ID_STR(data_id);
    conduit::Node slab;
    read_hdf5_hyperslab(h_data, filespace, world_rank, HDF5_KEY_DATA, dims, slab);
    sample[conduit_obj+"/slab"] = slab;
    //close data set
    H5Dclose(h_data);

    if (m_has_responses) {
      h_data = H5Dopen(h_file, HDF5_KEY_RESPONSES.c_str(), H5P_DEFAULT);
      H5Dread(h_data, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_all_responses);
      // conduit::Node work;
      // conduit::relay::io::hdf5_read(h_data, HDF5_KEY_RESPONSES.c_str(), work);
      //      node[conduit_obj+"/responses"].set(conduit::DataType::c_float(4));
      sample[conduit_obj+"/responses"].set(m_all_responses, 4);
      H5Dclose(h_data);
    }
    H5Fclose(h_file);
    return;
  }

  void hdf5_reader::load() {
    lbann_comm* l_comm = get_comm();
    if(dc::get_rank_stride() != 1) {
      LBANN_ERROR("HDF5 MPI-IO data reader requires DistConv rank stride = 1");
    }
    const El::mpi::Comm & w_comm = l_comm->get_world_comm();
    MPI_Comm mpi_comm = w_comm.GetMPIComm();
    int world_rank = get_rank_in_world();
    // MPI_Comm mpi_comm = dc::get_input_comm(*l_comm);
    // int world_rank = dc::get_input_rank(*l_comm); // Should probably be trainer rank
    int color = world_rank/dc::get_number_of_io_partitions();
    MPI_Comm_split(mpi_comm, color, world_rank, &m_comm);
    m_shuffled_indices.clear();
    m_shuffled_indices.resize(m_file_paths.size());
    std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if ((nprocs%dc::get_number_of_io_partitions()) !=0) {
      std::cerr<<"nprocs should be divisible by num of io partitions otherwise this wont work \n";
    }
    m_data_dims = {4, 512, 512, 512};
    m_num_features = std::accumulate(m_data_dims.begin(),
                                     m_data_dims.end(),
                                     (size_t) 1,
                                     std::multiplies<size_t>());

#ifdef DATA_READER_HDF5_USE_MPI_IO
    m_fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK_HDF5(H5Pset_fapl_mpio(m_fapl, m_comm, MPI_INFO_NULL));
    m_dxpl = H5Pcreate(H5P_DATASET_XFER);
    CHECK_HDF5(H5Pset_dxpl_mpio(m_dxpl, H5FD_MPIO_COLLECTIVE));
#else
    m_fapl = H5P_DEFAULT;
    m_dxpl = H5P_DEFAULT;
#endif
    std::vector<int> local_list_sizes;
    options *opts = options::get();
    if (opts->get_bool("preload_data_store")) {
      LBANN_ERROR("preload_data_store not supported on HDF5 data reader");
#if 0
      int np = l_comm->get_procs_per_trainer();
      int base_files_per_rank = m_shuffled_indices.size() / np;
      int extra = m_shuffled_indices.size() - (base_files_per_rank*np);
      if (extra > np) {
        LBANN_ERROR("extra > np");
      }
      local_list_sizes.resize(np, 0);
      for (int j=0; j<np; j++) {
        local_list_sizes[j] = base_files_per_rank;
        if (j < extra) {
          local_list_sizes[j] += 1;
        }
      }
#endif
    }
    instantiate_data_store(local_list_sizes);

    select_subset_of_data();
  }
  bool hdf5_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
    return true;
  }
  bool hdf5_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
    prof_region_begin("fetch_datum", prof_colors[0], false);

    // In the Cosmoflow case, each minibatch should have only one
    // sample per rank.
    assert_eq(X.Width(), 1);
    // Assuming 512^3 samples
    assert_eq(X.Height(),
               512 * 512 * 512 * 4 / dc::get_number_of_io_partitions()
              / (sizeof(DataType) / sizeof(short)));

    // Create a node to hold all of the data
    conduit::Node node;
    if (data_store_active()) {
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
    }else {
      read_hdf5_sample(data_id, node);
      if (priming_data_store()) {
        // Once the node has been populated save it in the data store
        m_data_store->set_conduit_node(data_id, node);
      }
    }
    const std::string conduit_obj = LBANN_DATA_ID_STR(data_id);
    conduit::Node slab = node[conduit_obj+"/slab"];
    unsigned short *data = slab.value();
    std::memcpy(X.Buffer(), data, slab.dtype().number_of_elements()*slab.dtype().element_bytes());

    prof_region_end("fetch_datum", false);
    return true;
  }
  //get from a cached response
  bool hdf5_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
    prof_region_begin("fetch_response", prof_colors[0], false);
    assert_eq(Y.Height(), 4);
    float *buf;
    // Create a node to hold all of the data
    conduit::Node node;
    if (data_store_active()) {
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
      const std::string conduit_obj = LBANN_DATA_ID_STR(data_id);
      buf = node[conduit_obj+"/responses"].value();
    }else {
      buf = m_all_responses;
    }
    std::memcpy(Y.Buffer(), buf,
                m_num_response_features*sizeof(DataType));

    prof_region_end("fetch_response", false);
    return true;
  }

};
