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
//
// prototext .hpp .cpp
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/protobuf_utils.hpp"
#include "lbann/proto/proto_common.hpp"

#include <google/protobuf/util/field_mask_util.h>
using google::protobuf::FieldMask;
using google::protobuf::util::FieldMaskUtil;

/**
 * all methods in protobuf_utils are static
 */

namespace lbann {

void protobuf_utils::parse_prototext_filenames_from_command_line(
               bool master,
               int argc,
               char **argv,
               std::vector<prototext_fn_triple> &names) {
  std::vector<std::string> models;
  std::vector<std::string> optimizers;
  std::vector<std::string> readers;
  std::vector<std::string> data_set_metadata;
  bool single_file_load = false;
  for (int k=1; k<argc; k++) {
    std::string s(argv[k]);
    if (s[0] != '-' or s[1] != '-') {
      std::cerr << "badly formed cmd line param; must begin with '--': " << s << std::endl;
      exit(1);
    }
    if (s.find(',') != std::string::npos) {
      std::stringstream err;
      err << __FILE__ << __LINE__ << " :: "
          << " badly formed param; contains ','; " << s << "\n"
          << "possibly you left out '{' or '}' or both ??\n";
      throw lbann_exception(err.str());
    }

    size_t equal_sign = s.find("=");
    if (equal_sign != std::string::npos) {
      std::string which = s.substr(2, equal_sign-2);
      std::string fn = s.substr(equal_sign+1);
      if (which == "prototext") {
        models.push_back(fn);
        single_file_load = true;
      }
      if (which == "model") {
        models.push_back(fn);
      }
      if (which == "reader") {
        readers.push_back(fn);
      }
      if (which == "metadata") {
        data_set_metadata.push_back(fn);
      }
      if (which == "optimizer") {
        optimizers.push_back(fn);
      }
    }
  }

  if(!single_file_load) {
    size_t n = models.size();
    if (! (optimizers.size() == 1 || optimizers.size() == n)) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " you specified " << n << " model filenames, and " << optimizers.size()
          << " optimizer filenames; you must specify either one or "<< n
          << " optimizer filenames";
      throw lbann_exception(err.str());
    }
    if (! (readers.size() == 1 || readers.size() == n)) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " you specified " << n << " model filenames, and " << readers.size()
          << " reader filenames; you must specify either one or "<< n
          << " reader filenames";
      throw lbann_exception(err.str());
    }

    if (! (data_set_metadata.size() == 0 || data_set_metadata.size() == 1 || data_set_metadata.size() == n)) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " you specified " << n << " model filenames, and " << data_set_metadata.size()
          << " data set metadata filenames; you must specify either zero, one, or "<< n
          << " data set metadata filenames";
      throw lbann_exception(err.str());
    }
  }

  names.clear();
  for (size_t i=0; i<models.size(); i++) {
    prototext_fn_triple t;
    t.model = models[i];
    if (readers.size() == 0) {
      t.reader = "none";
    }else if (readers.size() == 1) {
      t.reader = readers[0];
    } else {
      t.reader = readers[i];
    }
    if (data_set_metadata.size() == 0) {
      t.data_set_metadata = "none";
    }else if (data_set_metadata.size() == 1) {
      t.data_set_metadata = data_set_metadata[0];
    } else {
      t.data_set_metadata = data_set_metadata[i];
    }
    if (optimizers.size() == 0) {
      t.optimizer = "none";
    }else if (optimizers.size() == 1) {
      t.optimizer = optimizers[0];
    } else {
      t.optimizer = optimizers[i];
    }
    names.push_back(t);
  }
}


void protobuf_utils::read_in_prototext_files(
                bool master,
                std::vector<prototext_fn_triple> &names,
                std::vector<lbann_data::LbannPB*> &models_out) {
  models_out.clear();
  for (auto t : names) {
    lbann_data::LbannPB *pb = new lbann_data::LbannPB;
    const auto mergeOpts = FieldMaskUtil::MergeOptions();
    if (t.model != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.model.c_str(), p, master);
      FieldMask mask;
      mask.add_paths("model");
      FieldMaskUtil::MergeMessageTo(p, mask, mergeOpts, pb);
    }
    if (t.reader != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.reader.c_str(), p, master);
      FieldMask mask;
      mask.add_paths("reader");
      FieldMaskUtil::MergeMessageTo(p, mask, mergeOpts, pb);
    }
    if (t.data_set_metadata != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.data_set_metadata.c_str(), p, master);
      FieldMask mask;
      mask.add_paths("data_set_metadata");
      FieldMaskUtil::MergeMessageTo(p, mask, mergeOpts, pb);
    }
    if (t.optimizer != "none") {
      lbann_data::LbannPB p;
      read_prototext_file(t.optimizer.c_str(), p, master);
      FieldMask mask;
      mask.add_paths("optimizer");
      FieldMaskUtil::MergeMessageTo(p, mask, mergeOpts, pb);
    }
    models_out.push_back(pb);
  }
}

void protobuf_utils::load_prototext(
                const bool master,
                const int argc,
                char **argv,
                std::vector<lbann_data::LbannPB *> &models_out) {
    std::vector<prototext_fn_triple> names;
    parse_prototext_filenames_from_command_line(master, argc, argv, names);
    read_in_prototext_files(master, names, models_out);
    if (models_out.size() == 0) {
      if (master) {
        std::stringstream err;
        err << __FILE__ << __LINE__ << " :: "
            << " failed to load any prototext files";
        throw lbann_exception(err.str());
      }
    }
    verify_prototext(master, models_out);
}

void protobuf_utils::verify_prototext(bool master, const std::vector<lbann_data::LbannPB *> &models) {
  if (master) {
    std::cout << "protobuf_utils::verify_prototext; starting verify for " << models.size() << " models\n";
  }
  for (size_t j=0; j<models.size(); j++) {
    bool is_good = true;
    lbann_data::LbannPB *t = models[j];
    if (! t->has_data_reader()) {
      is_good = false;
      if (master) {
        std::cerr << "model #" << j << " is missing data_reader\n";
      }
    } else {
      if (t->data_reader().requires_data_set_metadata() && (! t->has_data_set_metadata())) {
        is_good = false;
        if (master) {
          std::cerr << "model #" << j << " is missing data_set_metadata\n";
        }
      }
      if (!t->data_reader().requires_data_set_metadata() && t->has_data_set_metadata()) {
        is_good = false;
        if (master) {
          std::stringstream err;
          err << "model #" << j << " is has data_set_metadata but does not require it\n"
              << " please check your command line\n";
          LBANN_ERROR(err.str());
        }
      }
    }
    if (! t->has_model()) {
      is_good = false;
      if (master) {
        std::cerr << "model #" << j << " is missing model\n";
      }
    }
    if (! t->has_optimizer()) {
      is_good = false;
      if (master) {
        std::cerr << "model #" << j << " is missing optimizer\n";
      }
    }

    if (! is_good) {
      if (master) {
        std::stringstream err;
        err << __FILE__ << __LINE__ << " :: "
            << " prototext is missing reader, metadata, optimizer, and/or model;\n"
            << " please check your command line\n";
        throw lbann_exception(err.str());
      }
    }
  }
}


}  // namespace lbann
