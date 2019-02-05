#include "lbann/proto/proto_common.hpp"

#include "lbann/lbann.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/proto/init_image_data_readers.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

namespace lbann {

bool has_motifs(lbann_comm *comm, const lbann_data::LbannPB& p) {
  bool master = comm->am_world_master();
  if (master) std::cout << "starting has_motifs\n";
  const lbann_data::Model& m = p.model();
  const int num_layers = m.layer_size();
  for (int j=0; j<num_layers; j++) {
    const lbann_data::Layer& layer = m.layer(j);
    if (layer.has_motif_layer()) {
      return true;
    }
  }
  return false;
}

void expand_motifs(lbann_comm *comm, lbann_data::LbannPB& pb) {
  bool master = comm->am_world_master();
  if (master) std::cout << "starting expand_motifs\n";
  const lbann_data::MotifDefinitions& m = pb.motif_definitions();
  const int num_motifs = m.motif_size();
  for (int j=0; j<num_motifs; j++) {
  }
}

int get_requested_num_parallel_readers(const lbann_comm *comm, const lbann_data::LbannPB& p);

void init_data_readers(lbann_comm *comm, const lbann_data::LbannPB& p, std::map<execution_mode, generic_data_reader *>& data_readers,
                       bool is_shareable_training_data_reader, bool is_shareable_testing_data_reader, bool is_shareable_validation_data_reader)
{
#ifdef LBANN_HAS_CONDUIT
  static std::unordered_map<std::string, data_reader_jag_conduit*> leading_reader_jag_conduit;
#endif
  bool master = comm->am_world_master();
  std::stringstream err;

  options *opts = options::get();
  bool create_tarball = opts->has_string("create_tarball") ? true : false;

  const lbann_data::DataReader & d_reader = p.data_reader();
  int size = d_reader.reader_size();

  const lbann_data::DataSetMetaData& pb_metadata = p.data_set_metadata();

  // A separate explicit validation set is created only if a reader with role "validate"
  // is found in the list of data readers. Otherwise, a validation set is created as a
  // percentage of data from the train set.
  bool separate_validation = false;
  for (int j=0; j<size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);
    if (readme.role() == "validate") {
        separate_validation = true;
        break;
    }
  }

  for (int j=0; j<size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);
    // This is a temporary measure until we individually setup data reader specific preprocessors
    bool set_up_generic_preprocessor = true;

    const std::string& name = readme.name();

    const bool shuffle = readme.shuffle();

    generic_data_reader *reader = nullptr;
    generic_data_reader *reader_validation = nullptr;

    if ((name == "mnist") || (name == "cifar10") || (name == "moving_mnist")) {
      init_org_image_data_reader(readme, master, reader);
      set_up_generic_preprocessor = false;
    } else if ((name == "imagenet") || (name == "imagenet_patches") ||
               (name == "triplet") || (name == "mnist_siamese") || (name == "multi_images")) {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_up_generic_preprocessor = false;
    } else if (name == "jag") {
      auto* reader_jag = new data_reader_jag(shuffle);

      const lbann_data::DataSetMetaData::Schema& pb_schema = pb_metadata.schema();

      using var_t = data_reader_jag::variable_t;

      // composite independent variable
      std::vector< std::vector<var_t> > independent_type(pb_schema.independent_size());

      for (int i=0; i < pb_schema.independent_size(); ++i) {
        const lbann_data::DataSetMetaData::Schema::JAGDataSlice& slice = pb_schema.independent(i);
        const int slice_size = slice.pieces_size();
        for (int k=0; k < slice_size; ++k) {
          const auto var_type = static_cast<var_t>(slice.pieces(k));
          independent_type[i].push_back(var_type);
        }
      }

      reader_jag->set_independent_variable_type(independent_type);

      // composite dependent variable
      std::vector< std::vector<var_t> > dependent_type(pb_schema.dependent_size());

      for (int i=0; i < pb_schema.dependent_size(); ++i) {
        const lbann_data::DataSetMetaData::Schema::JAGDataSlice& slice = pb_schema.dependent(i);
        const int slice_size = slice.pieces_size();
        for (int k=0; k < slice_size; ++k) {
          const auto var_type = static_cast<var_t>(slice.pieces(k));
          dependent_type[i].push_back(var_type);
        }
      }

      reader_jag->set_dependent_variable_type(dependent_type);

      const lbann_data::ImagePreprocessor& pb_preproc = readme.image_preprocessor();
      reader_jag->set_image_dims(pb_preproc.raw_width(), pb_preproc.raw_height());
      reader_jag->set_normalization_mode(pb_preproc.early_normalization());
      reader = reader_jag;
      set_up_generic_preprocessor = false;
#ifdef LBANN_HAS_CONDUIT
    } else if (name == "jag_conduit") {
      init_image_data_reader(readme, pb_metadata, master, reader);
      auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader);
      const lbann_data::Model& pb_model = p.model();
      reader->set_mini_batch_size(static_cast<int>(pb_model.mini_batch_size()));

      /// Allow the prototext to control if the data readers is
      /// shareable for each phase training, validation, or testing
      if((is_shareable_training_data_reader && readme.role() == "train")
         || (is_shareable_testing_data_reader && readme.role() == "test")
         || (is_shareable_validation_data_reader && readme.role() == "validation")) {
        if (!peek_map(leading_reader_jag_conduit, readme.role())) {
          leading_reader_jag_conduit[readme.role()] = reader_jag_conduit;
        } else {
          const auto leader = peek_map(leading_reader_jag_conduit, readme.role());
          *reader_jag_conduit = *leader;
          reader_jag_conduit->set_leading_reader(leader);
        }
      }

      for (int i=0; i < pb_model.layer_size(); ++i) {
        const auto& proto_layer = pb_model.layer(i);
        if (proto_layer.has_input()) {
          const auto& params = proto_layer.input();
          const auto& io_buffer = params.io_buffer();
          reader_jag_conduit->set_io_buffer_type(io_buffer);
          const auto num_readers = get_requested_num_parallel_readers(comm, p);
          reader_jag_conduit->set_num_parallel_readers(num_readers);
          reader_jag_conduit->set_local_id(readme.role());
          break;
        }
      }
      set_up_generic_preprocessor = false;
    } else if (name == "jag_conduit_hdf5") {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_up_generic_preprocessor = false;
#endif // LBANN_HAS_CONDUIT
    } else if (name == "nci") {
      reader = new data_reader_nci(shuffle);
    } else if (name == "csv") {
      auto* reader_csv = new csv_reader(shuffle);
      reader_csv->set_label_col(readme.label_col());
      reader_csv->set_response_col(readme.response_col());
      reader_csv->disable_labels(readme.disable_labels());
      reader_csv->enable_responses(readme.disable_responses());
      reader_csv->set_separator(readme.separator()[0]);
      reader_csv->set_skip_cols(readme.skip_cols());
      reader_csv->set_skip_rows(readme.skip_rows());
      reader_csv->set_has_header(readme.has_header());
      reader = reader_csv;
    } else if (name == "numpy") {
      auto* reader_numpy = new numpy_reader(shuffle);
      reader_numpy->set_has_labels(!readme.disable_labels());
      reader_numpy->set_has_responses(!readme.disable_responses());
      reader = reader_numpy;
    } else if (name == "numpy_npz") {
      auto* reader_numpy_npz = new numpy_npz_reader(shuffle);
      reader_numpy_npz->set_has_labels(!readme.disable_labels());
      reader_numpy_npz->set_has_responses(!readme.disable_responses());
      reader_numpy_npz->set_scaling_factor_int16(readme.scaling_factor_int16());
      reader = reader_numpy_npz;
    } else if (name == "pilot2_molecular_reader") {
      pilot2_molecular_reader* reader_pilot2_molecular = new pilot2_molecular_reader(readme.num_neighbors(), readme.max_neighborhood(), shuffle);
      reader = reader_pilot2_molecular;
    } else if (name == "merge_samples" || name == "merge_features" || name == "multi_conduit") {
      //TODO: verify how much of wildcard conflict with label file, label file should be loaded separately
      auto filedir = readme.data_filedir();
      if(!endsWith(filedir, "/")) {
        filedir = filedir + "/";
      }
      auto paths = glob(filedir + readme.data_file_pattern());
      if(readme.merge_skip_overlapped()) {
        assert((paths.size()%comm->get_num_trainers()) == 0);
      }
      std::vector<generic_data_reader*> npy_readers;
      for(auto i = paths.begin(); i != paths.end(); i++) {
        const auto path = *i;
        if(readme.merge_skip_overlapped()
           && (std::distance(paths.begin(), i)%comm->get_procs_per_trainer()) != comm->get_rank_in_trainer()) {
          continue;
        }
        if(master) { std::cout << "Loading file: " << path << std::endl; }
        if (readme.format() == "numpy") {
          auto *reader_numpy = new numpy_reader(false);
          reader_numpy->set_data_filename(path);
          reader_numpy->set_has_labels(!readme.disable_labels());
          reader_numpy->set_has_responses(!readme.disable_responses());
          npy_readers.push_back(reader_numpy);
        } else if (readme.format() == "numpy_npz") {
          auto* reader_numpy_npz = new numpy_npz_reader(false);
          reader_numpy_npz->set_data_filename(path);
          reader_numpy_npz->set_has_labels(!readme.disable_labels());
          reader_numpy_npz->set_has_responses(!readme.disable_responses());
          reader_numpy_npz->set_scaling_factor_int16(readme.scaling_factor_int16());
          npy_readers.push_back(reader_numpy_npz);
#ifdef LBANN_HAS_CONDUIT
        } else if (readme.format() == "jag_conduit") {
          init_image_data_reader(readme, pb_metadata, master, reader);
          set_up_generic_preprocessor = false;
          npy_readers.push_back(reader);
#endif
        } else if (readme.format() == "pilot2_molecular_reader") {
          pilot2_molecular_reader* reader_pilot2_molecular = new pilot2_molecular_reader(readme.num_neighbors(), readme.max_neighborhood(), shuffle);
          reader_pilot2_molecular->set_data_filename(path);
          npy_readers.push_back(reader_pilot2_molecular);
        } else if (readme.format() == "csv") {
          auto* reader_csv = new csv_reader(shuffle);
          if(master) { std::cout << "Set data filename: " << path << std::endl; }
          reader_csv->set_data_filename(path);
          reader_csv->set_label_col(readme.label_col());
          reader_csv->set_response_col(readme.response_col());
          reader_csv->disable_labels(readme.disable_labels());
          reader_csv->enable_responses(readme.disable_responses());
          reader_csv->set_separator(readme.separator()[0]);
          reader_csv->set_skip_cols(readme.skip_cols());
          reader_csv->set_skip_rows(readme.skip_rows());
          reader_csv->set_has_header(readme.has_header());
          reader_csv->set_absolute_sample_count( readme.absolute_sample_count() );
          reader_csv->set_use_percent( readme.percent_of_data_to_use() );
          reader_csv->set_first_n( readme.first_n() );
          npy_readers.push_back(reader_csv);
        } else {
          err << __FILE__ << " " << __LINE__ << " :: unknown format for merged data reader: "
              << name;
          throw lbann_exception(err.str());
        }
      }
      if(name == "merge_samples") {
        data_reader_merge_samples* merged_samples = new data_reader_merge_samples(npy_readers, shuffle);
        reader = merged_samples;
      } else if (name == "multi_conduit") {
        //note: this is not a mistake! We may have a separate multi_conduit
        //      reader in the future, but for now merge_samples does what we need.
        data_reader_merge_samples* multi_conduit = new data_reader_merge_samples(npy_readers, shuffle);
        reader = multi_conduit;
      } else {
        //create label file
        //we can use merge_features without label
        generic_data_reader* label_reader = nullptr;
        if(readme.label_filename() != "") {
          if(master) { std::cout << "Set label filename: " << readme.label_filename() << std::endl; }
          if (readme.format() == "numpy") {
             auto* label_numpy  = new numpy_reader(false);
             label_numpy->set_label_filename(readme.label_filename());
             label_numpy->set_data_filename(readme.label_filename());
             label_reader = label_numpy;
           } else if (readme.format() == "csv") { //if format is csv and label_filename is not empty
             auto* label_csv = new csv_reader(shuffle);
             if(master) { std::cout << "Set label filename: " << readme.label_filename() << std::endl; }
             label_csv->set_label_filename(readme.label_filename());
             label_csv->set_data_filename(readme.label_filename());
             label_csv->disable_labels(readme.disable_labels());
             label_csv->enable_responses(readme.disable_responses());
             label_csv->set_has_header(readme.has_header()); //use same as parent file
             label_csv->set_comm(comm);
             label_csv->set_label_col(0); //assume there is only one label file and the column and is label column
             label_csv->set_response_col(0);
             label_reader = label_csv;
           } else {
             err << __FILE__ << " " << __LINE__ << " :: unknown format for merged features label: "
                << readme.format();
             throw lbann_exception(err.str());
           }
         }
        //data_reader_merge_features* merged_features = new data_reader_merge_features(npy_readers,label_csv, shuffle);
        data_reader_merge_features* merged_features = new data_reader_merge_features(npy_readers,label_reader, shuffle);
        reader = merged_features;
      }

    } else if (name == "synthetic") {
      if (readme.num_labels() != 0) {
        reader = new data_reader_synthetic(
          readme.num_samples(),
          proto::parse_list<int>(readme.synth_dimensions()),
          readme.num_labels(),
          shuffle);
      } else {
        reader = new data_reader_synthetic(
          readme.num_samples(),
          proto::parse_list<int>(readme.synth_dimensions()),
          proto::parse_list<int>(readme.synth_response_dimensions()),
          shuffle);
      }
    } else if (name == "mesh") {
      reader = new mesh_reader(shuffle);
    } else if (name == "moving_mnist") {
      reader = new moving_mnist_reader(7, 40, 40, 2);
    } else {
      if (master) {
        err << __FILE__ << " " << __LINE__ << " :: unknown name for data reader: "
            << name;
        throw lbann_exception(err.str());
      }
    }
    reader->set_comm(comm);

    if (readme.data_filename() != "") {
      reader->set_data_filename( readme.data_filename() );
    }
    if (readme.label_filename() != "" && name != "merge_features") { //label_file set differently for merge_features
      reader->set_label_filename( readme.label_filename() );
    }
    if (readme.data_filedir() != "") {
      reader->set_file_dir( readme.data_filedir() );
    }
    reader->set_max_files_to_load( readme.max_files_to_load() );
    if (readme.data_local_filedir() != "") {
      reader->set_local_file_dir( readme.data_local_filedir() );
    }

    if (create_tarball) {
      if (opts->has_int("test_tarball")) {
        reader->set_absolute_sample_count( opts->get_int("test_tarball"));
        reader->set_use_percent( 0. );
        reader->set_first_n(0);
      } else {
        reader->set_absolute_sample_count( 0. );
        reader->set_use_percent( 1.0 );
        reader->set_first_n( 0 );
      }
    } else {
      reader->set_absolute_sample_count( readme.absolute_sample_count() );
      reader->set_use_percent( readme.percent_of_data_to_use() );
      reader->set_first_n( readme.first_n() );

      reader->set_gan_labelling(readme.gan_labelling());
      reader->set_gan_label_value(readme.gan_label_value());

      reader->set_partitioned(readme.is_partitioned(), readme.partition_overlap(), readme.partition_mode());

      if (set_up_generic_preprocessor) {
        init_generic_preprocessor(readme, master, reader);
      }
    }

    if (readme.role() == "train") {
      reader->set_role("train");
    } else if (readme.role() == "test") {
      reader->set_role("test");
    } else if (readme.role() == "validate") {
      reader->set_role("validate");
    } else {
      reader->set_role("error");
    }
    if (readme.role() == "train") {
      if (create_tarball || separate_validation) {
        reader->set_validation_percent( 0. );
      } else {
        reader->set_validation_percent( readme.validation_percent() );
      }
    }

    reader->set_master(master);

    reader->load();

    if (readme.role() == "train") {
      data_readers[execution_mode::training] = reader;
    } else if (readme.role() == "test") {
      // While the default validation_percent is 0.0, this line is added to be consistent with the case of "train"
      reader->set_validation_percent( 0. );
      data_readers[execution_mode::testing] = reader;
    } else if (readme.role() == "validate") {
      reader->set_validation_percent( 0. );
      data_readers[execution_mode::validation] = reader;
    }

    if (readme.role() == "train" && readme.validation_percent() > 0. && !create_tarball && !separate_validation) {
      if (name == "mnist") {
        reader_validation = new mnist_reader(shuffle);
        (*(mnist_reader *)reader_validation) = (*(mnist_reader *)reader);
      } else if (name == "imagenet") {
        reader_validation = new imagenet_reader(*dynamic_cast<const imagenet_reader*>(reader));
      } else if (name == "imagenet_patches") {
        reader_validation = new imagenet_reader_patches(*dynamic_cast<const imagenet_reader_patches*>(reader));
      } else if (name == "triplet") {
        reader_validation = new data_reader_triplet(*dynamic_cast<const data_reader_triplet*>(reader));
      } else if (name == "mnist_siamese") {
        reader_validation = new data_reader_mnist_siamese(*dynamic_cast<const data_reader_mnist_siamese*>(reader));
      } else if (name == "multi_images") {
        reader_validation = new data_reader_multi_images(*dynamic_cast<const data_reader_multi_images*>(reader));
      } else if (name == "jag") {
        reader_validation = new data_reader_jag(shuffle);
        *dynamic_cast<data_reader_jag*>(reader_validation) = *dynamic_cast<const data_reader_jag*>(reader);
#ifdef LBANN_HAS_CONDUIT
      } else if (name == "jag_conduit") {
        /// If the training data reader was shared and the validate reader is split from it, then the validation data reader
        /// is also shared
        if(is_shareable_training_data_reader) {
          const std::string role = "validate";
          if (!peek_map(leading_reader_jag_conduit, role)) {
            reader_validation = new data_reader_jag_conduit(*dynamic_cast<const data_reader_jag_conduit*>(reader));
            auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader_validation);
            reader_jag_conduit->set_leading_reader(reader_jag_conduit);
            reader_jag_conduit->set_role(role);
            leading_reader_jag_conduit[role] = reader_jag_conduit;
          } else {
            // Copy construct the leading validation reader into another validation reader.
            // We do not copy the train reader as the subset of data may already have been
            // assigned to validation reader when validation percent is set.
            // Thus, we need to avoid taking a subset of a subset.
            const auto leader = peek_map(leading_reader_jag_conduit, role);
            reader_validation = new data_reader_jag_conduit(*leader);
            auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader_validation);
            reader_jag_conduit->set_leading_reader(leader);
          }
        } else {
          reader_validation = new data_reader_jag_conduit(*dynamic_cast<const data_reader_jag_conduit*>(reader));
        }
#endif // LBANN_HAS_CONDUIT
      } else if (name == "nci") {
        reader_validation = new data_reader_nci(shuffle);
        (*(data_reader_nci *)reader_validation) = (*(data_reader_nci *)reader);
      } else if (name == "csv") {
        reader_validation = new csv_reader(shuffle);
        (*(csv_reader *)reader_validation) = (*(csv_reader *)reader);
      } else if (name == "numpy") {
        reader_validation = new numpy_reader(shuffle);
        (*(numpy_reader *)reader_validation) = (*(numpy_reader *)reader);
      } else if (name == "merge_samples") {
        reader_validation = new data_reader_merge_samples(*(data_reader_merge_samples *)reader);
      } else if (name == "merge_features") {
        reader_validation = new data_reader_merge_features(*(data_reader_merge_features *)reader);
      } else if (name == "cifar10") {
        reader_validation = new cifar10_reader(shuffle);
        (*(cifar10_reader *)reader_validation) = (*(cifar10_reader *)reader);
      } else if (name == "synthetic") {
        reader_validation = new data_reader_synthetic(*(data_reader_synthetic *)reader);
        (*(data_reader_synthetic *) reader_validation) = (*(data_reader_synthetic *)reader);
      } else if (name == "mesh") {
        reader_validation = new mesh_reader(shuffle);
        (*(mesh_reader *)reader_validation) = (*(mesh_reader *)reader);
      } else if (name == "moving_mnist") {
        reader_validation = new moving_mnist_reader(7, 40, 40, 2);
        (*(moving_mnist_reader *)reader_validation) = (*(moving_mnist_reader *)reader);
      }

      reader_validation->set_role("validate");
      reader_validation->use_unused_index_set();

      if (master) {
        size_t num_train = reader->get_num_data();
        size_t num_validate = reader_validation->get_num_data();
        double validate_percent = ((double) num_validate / (double) (num_train+num_validate))*100.0;
        double train_percent = ((double) num_train / (double) (num_train+num_validate))*100.0;
        std::cout << "Training using " << train_percent << "% of the training data set, which is " << reader->get_num_data() << " samples." << std::endl
                  << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->get_num_data() << " samples.";
#ifdef LBANN_HAS_CONDUIT
        if (name == "jag_conduit") {
          std::cout << " jag conduit leading reader " << dynamic_cast<data_reader_jag_conduit*>(reader)->get_leading_reader()
                    << " of " << (is_shareable_training_data_reader? "shared" : "unshared") << " reader " << reader << " for " << reader->get_role() << std::endl;
        }
#endif // LBANN_HAS_CONDUIT
        std::cout << std::endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }

  if (master) {
    if (separate_validation) {
      const generic_data_reader* r_train = peek_map(data_readers, execution_mode::training);
      const generic_data_reader* r_validate = peek_map(data_readers, execution_mode::validation);
      const size_t num_train = (r_train == nullptr)? 0u : r_train->get_num_data();
      const size_t num_validate = (r_validate == nullptr)? 0u : r_validate->get_num_data();
      std::cout << "Training using " << num_train << " samples." << std::endl
                << "Validating using " << num_validate << " samples." << std::endl;
    }
    const generic_data_reader* r_test = peek_map(data_readers, execution_mode::testing);
    const size_t num_test = (r_test == nullptr)? 0u : r_test->get_num_data();
    std::cout << "Testing using " << num_test << " samples." << std::endl;
  }
  // remove null data_reader pointers if there is any
  for (auto it = data_readers.cbegin(); it != data_readers.cend() ; ) {
    if (!it->second) {
      it = data_readers.erase(it);
    } else {
      ++it;
    }
  }
}

void read_prototext_file(std::string fn, lbann_data::LbannPB& pb, bool master)
{
  std::stringstream err;
  int fd = open(fn.c_str(), O_RDONLY);
  if (fd == -1) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
      throw lbann_exception(err.str());
    }
  }
  auto *input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, &pb);
  if (!success) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << std::endl;
      throw lbann_exception(err.str());
    }
  }
  input->Close();
  delete input;
}

bool write_prototext_file(const char *fn, lbann_data::LbannPB& pb)
{
  int fd = open(fn, O_APPEND | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    return false;
  }
  auto *output = new google::protobuf::io::FileOutputStream(fd);
  if (!google::protobuf::TextFormat::Print(pb, output)) {
    close(fd);
    delete output;
    return false;
  }
  delete output;
  close(fd);
  return true;
}

bool check_if_num_parallel_readers_set(const lbann_comm *comm, const lbann_data::Model& model)
{
  const bool master = comm->am_world_master();
  const int parallel_io = model.num_parallel_readers();

  if (parallel_io == 0) {
    if (master) {
      std::cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_trainer() <<
        " (Limited to # Processes)" << std::endl;
    }
    return false;
  }
  if (master) {
    std::cout << "\tMax Parallel I/O Fetch: " << parallel_io << std::endl;
  }
  return true;
}

void set_num_parallel_readers(const lbann_comm *comm, lbann_data::LbannPB& p)
{
  lbann_data::Model *model = p.mutable_model();
  const bool is_set = check_if_num_parallel_readers_set(comm, *model);

  if (!is_set) {
    const int parallel_io = comm->get_procs_per_trainer();
    model->set_num_parallel_readers(parallel_io); //adjust the prototext
  }
}

int get_requested_num_parallel_readers(const lbann_comm *comm, const lbann_data::LbannPB& p)
{
  const lbann_data::Model& model = p.model();
  const bool is_set = check_if_num_parallel_readers_set(comm, model);

  if (!is_set) {
    return comm->get_procs_per_trainer();
  }
  return model.num_parallel_readers();
}

void set_data_readers_filenames(std::string which, lbann_data::LbannPB& p)
{
  options *opts = options::get();
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    if (r->role() == which) {
      std::stringstream s;
      s << "data_filedir_" << which;
      if (opts->has_string(s.str())) {
        r->set_data_filedir(opts->get_string(s.str()));
      }else {
        s.clear();
        s.str("");
        s << "data_filedir";
        if (opts->has_string(s.str())) {
          r->set_data_filedir(opts->get_string(s.str()));
        }
      }
      s.clear();
      s.str("");
      s << "data_filename_" << which;
      if (opts->has_string(s.str())) {
        r->set_data_filename(opts->get_string(s.str()));
      }
      s.clear();
      s.str("");
      s << "label_filename_" << which;
      if (opts->has_string(s.str())) {
        r->set_label_filename(opts->get_string(s.str()));
      }
    }
  }
}

void set_data_readers_percent(lbann_data::LbannPB& p)
{
  options *opts = options::get();
  double percent = opts->get_float("data_reader_percent");
  if (percent <= 0 || percent > 1.0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " --data_reader_percent=<float> must be > 0 and <= 1.0";
      throw lbann_exception(err.str());
  }
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    r->set_percent_of_data_to_use( percent );
  }
}

void get_cmdline_overrides(lbann_comm *comm, lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();
  std::stringstream err;

  options *opts = options::get();
  lbann_data::Model *model = p.mutable_model();
  lbann_data::DataReader *d_reader = p.mutable_data_reader();
  int size = d_reader->reader_size();

  if (opts->has_int("absolute_sample_count")) {
    for (int j=0; j<size; j++) {
      int n = opts->get_int("absolute_sample_count");
      lbann_data::Reader *readme = d_reader->mutable_reader(j);
      readme->set_percent_of_data_to_use(0.0);
      readme->set_absolute_sample_count(n);
    }
  }

  if (opts->has_string("dag_model")) {
    std::string sanity = model->type();
    if (sanity != "dnn") {
      err << __FILE__ << " " << __LINE__ << " :: "
          << " the current network model is: " << model->type()
          << "; you can only change the model to 'dag_model' if the current model is 'dnn'";
      throw lbann_exception(err.str());
    }
    if (master) {
      std::cout << "\nchanging model from " << model->type() << " to: dag\n\n";
    }
    model->set_type("dag_model");
  }

  if (opts->has_string("data_filedir")
      or opts->has_string("data_filedir_train")
      or opts->has_string("data_filename_train")
      or opts->has_string("label_filename_train")) {
    set_data_readers_filenames("train", p);
  }
  if (opts->has_string("data_filedir")
      or opts->has_string("data_filedir_test")
      or opts->has_string("data_filename_test")
      or opts->has_string("label_filename_test")) {
    set_data_readers_filenames("test", p);
  }
  if (opts->has_string("data_reader_percent")) {
    set_data_readers_percent(p);
  }
  if (opts->has_bool("no_im_comm") and opts->get_bool("no_im_comm")) {
    int sz = model->callback_size();
    for (int j=0; j<sz; j++) {
      lbann_data::Callback *c = model->mutable_callback(j);
      if (c->has_imcomm()) {
        c->clear_imcomm();
      }
    }
  }
  if (opts->has_int("mini_batch_size")) {
    model->set_mini_batch_size(opts->get_int("mini_batch_size"));
  }
  if (opts->has_int("num_epochs")) {
    model->set_num_epochs(opts->get_int("num_epochs"));
  }
  if (opts->has_int("block_size")) {
    model->set_block_size(opts->get_int("block_size"));
  }
  if (opts->has_int("procs_per_trainer")) {
    model->set_procs_per_trainer(opts->get_int("procs_per_trainer"));
  }
  if (opts->has_int("num_parallel_readers")) {
    model->set_num_parallel_readers(opts->get_int("num_parallel_readers"));
  }
  if (opts->has_bool("disable_cuda")) {
    model->set_disable_cuda(opts->get_bool("disable_cuda"));
  }
  if (opts->has_int("random_seed")) {
    model->set_random_seed(opts->get_int("random_seed"));
  }


  if (opts->has_string("opt")) {
    //defaults
    double learn_rate = opts->has_float("learn_rate") ? opts->get_float("learn_rate") : 0.01;
    double eps = opts->has_float("eps") ? opts->get_float("eps") : 1e-8;
    double beta1 = opts->has_float("beta1") ? opts->get_float("beta1") : 0.9;
    double beta2 = opts->has_float("beta2") ? opts->get_float("beta2") : 0.99;
    double init_learning_rate = opts->has_float("init_learning_rate") ? opts->get_float("init_learning_rate") : 0.01;
    double hyper_learning_rate = opts->has_float("hyper_learning_rate") ? opts->get_float("hyper_learning_rate") : 1e-7;
    double momentum = opts->has_float("momentum") ? opts->get_float("momentum") : 0.9;
    double decay_rate = opts->has_float("decay_rate") ? opts->get_float("decay_rate") : 0.5;
    bool nesterov = opts->has_bool("nesterov") ? opts->get_float("nesterov") : false;

    auto *opt = new lbann_data::Optimizer;

    //construct the new optimizer
    std::string opt_string = opts->get_string("opt");
    if (opt_string == "adagrad") {
      auto *a = new lbann_data::Adagrad;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      opt->set_allocated_adagrad(a);
    } else if (opt_string == "adam") {
      auto *a = new lbann_data::Adam;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      opt->set_allocated_adam(a);
    } else if (opt_string == "hypergradient_adam") {
      auto *a = new lbann_data::HypergradientAdam;
      a->set_init_learning_rate(init_learning_rate);
      a->set_hyper_learning_rate(hyper_learning_rate);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      a->set_eps(eps);
      opt->set_allocated_hypergradient_adam(a);
    } else if (opt_string == "rmsprop") {
      auto *a = new lbann_data::Rmsprop;
      a->set_learn_rate(learn_rate);
      a->set_decay_rate(decay_rate);
      a->set_eps(eps);
      opt->set_allocated_rmsprop(a);
    } else if (opt_string == "sgd") {
      if (master) std::cerr << "\n\nsetting: sgd\n\n";
      auto *a = new lbann_data::Sgd;
      a->set_learn_rate(learn_rate);
      a->set_momentum(momentum);
      a->set_decay_rate(decay_rate);
      a->set_nesterov(nesterov);
      opt->set_allocated_sgd(a);
    } else {
      err << __FILE__ << " " << __LINE__
          << " :: unknown string for --optimizer: " << opt_string
          << " should be on of: adagrad, adam, hypergradient_adam, rmsprop, sgd";
      throw lbann_exception(err.str());
    }
    p.set_allocated_optimizer(opt);
  }
}

void print_parameters(lbann_comm *comm, lbann_data::LbannPB& p)
{
  if (!comm->am_world_master()) {
    return;
  }

  const lbann_data::Model &m = p.model();

  std::cout << std::endl
            << "Running with these parameters:\n"
            << " General:\n"
            << "  datatype size:           " << sizeof(DataType) << std::endl
            << "  mini_batch_size:         " << m.mini_batch_size() << std::endl
            << "  num_epochs:              " << m.num_epochs()  << std::endl
            << "  block_size:              " << m.block_size()  << std::endl
            << "  procs_per_trainer:       " << m.procs_per_trainer()  << std::endl
            << "  num_parallel_readers:    " << m.num_parallel_readers()  << std::endl
            << "  serialize_background_io: " << m.serialize_background_io()  << std::endl
            << "  disable_cuda:            " << m.disable_cuda()  << std::endl
            << "  random_seed:             " << m.random_seed() << std::endl
            << "  data_layout:             " << m.data_layout()  << std::endl
            << "     (only used for metrics)\n";
}

void print_help(lbann_comm *comm)
{
  if (!comm->am_world_master()) {
    return;
  }

  std::cerr <<
       "General usage: you need to specify three prototext files, e.g:\n"
       "  srun -n# proto --model=<string> --optimizer=<string> --reader=<string> --metadata=<string>\n"
       "\n"
       "  However, if you are re-running an experiment from a previously saved\n"
       "  file, you only need to specify --prototext=<string>\n"
       "  When proto is run, an output file containing the concatenated prototext\n"
       "  files, along with other data is written. The default name for this file\n"
       "  is 'data.prototext'  You can specify an alternative name via the option:\n"
       "  --saveme=<string>  You can suppress writing the file via the option:\n"
       "  --saveme=0\n"
       "\n"
       "  To reload from a previous checkpoint you specify --ckpt_dir=<string>\n"
       "\n"
       "Some prototext values can be over-riden on the command line;\n"
       "(notes: use '1' or '0' for bool; if no value is given for a flag,\n"
       "        e.g: --disable_cuda, then a value of '1' is assigned)\n"
       "\n"
       "General:\n"
       "  --dag_model\n"
       "  --mini_batch_size=<int>\n"
       "  --num_epochs=<int>\n"
       "  --block_size=<int>\n"
       "  --procs_per_trainer=<int>\n"
       "  --num_gpus=<int>\n"
       "  --num_parallel_readers=<int>\n"
       "  --num_io_threads=<int>\n"
       "  --disable_background_io_activity=<bool>\n"
       "  --disable_cuda=<bool>\n"
       "     has no effect unless lbann was compiled with: LBANN_HAS_CUDNN\n"
       "  --random_seed=<int>\n"
       "  --objective_function<string>\n"
       "      <string> must be: categorical_cross_entropy or mean_squared_error\n"
       "  --data_layout<string>\n"
       "      <string> must be: data_parallel or model_parallel\n"
       "      note: this will be applied to all layers, metrics (and others)\n"
       "            that take DATA_PARALLEL or MODEL_PARALLEL as a template parameter\n"
       "  --print_affinity\n"
       "      display information on how OpenMP threads are provisioned\n"
       "\n"
       "DataReaders:\n"
       "  --data_filedir=<string>\n"
       "      sets the file directory for train and test data\n"
       "  --data_filedir_train=<string>   --data_filedir_test=<string>\n"
       "  --data_filename_train=<string>  --data_filename_test=<string>\n"
       "  --label_filename_train=<string> --label_filename_test=<string>\n"
       "  --data_reader_percent=<float>\n"
       "  --share_testing_data_readers=<bool:[0|1]>\n"
       "\n"
       "Callbacks:\n"
       "  --image_dir=<string>\n"
       "      if the model has callback_save_images, this determines where the\n"
       "      images are saved\n"
       "  --no_im_comm=<bool>\n"
       "      removes ImComm callback, if present; this is intended for\n"
       "      running alexnet with a single model, but may be useful elsewhere\n"
       "\n"
       "Optimizers; all values except for nesterov are floats;\n"
       "            the values shown in <...> are the default values, that will be\n"
       "            used if the option is not specified on the cmd line.\n"
       "            If you specify an option that is not applicable to your choice\n"
       "            of optimizer, the option is ignored\n"
       "\n"
       "  --opt=<string>\n"
       "     <string> must be one of:\n"
       "         adagrad, adam, hypergradient_adam, rmsprop, sgd\n"
       "\n"
       "  --learn_rate=< 0.01 >          (all except hypergradient_adam)\n"
       "  --eps=< 1e-8 >                 (all except sgd)\n"
       "  --beta1=< 0.9 >                (adam, hypergradient_adam)\n"
       "  --beta2=< 0.99 >               (adam, hypergradient_adam)\n"
       "  --init_learning_rate=< 0.01 >  (hypergradient_adam)\n"
       "  --hyper_learning_rate=< 1e-7 > (hypergradient_adam)\n"
       "  --momentum=< 0.9 >             (sgd)\n"
       "  --decay_rate=< 0.5 >           (sgd, rmsprop)\n"
       "  --nesterov=< false >           (sgd)\n";
}

void copy_file(std::string fn, std::ofstream &out)
{
  std::ifstream in(fn.c_str());
  if (!in.is_open()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for reading: " << fn;
    throw std::runtime_error(err.str());
  }
  std::stringstream s;
  s << in.rdbuf();
  out << s.str();
}

void save_session(lbann_comm *comm, int argc, char **argv, lbann_data::LbannPB& p)
{
  if (!comm->am_world_master()) {
    return;
  }

  options *opts = options::get();

  //do not write output file for a repeated experiment;
  //may want to revisit this decision later ...
  if (opts->has_string("prototext")) {
    return;
  }

  //setup file name
  // Note: If the file name is not unique, append numbers until it is.
  std::string model_name = p.model().name();
  if (model_name.empty()) { model_name = "model"; };
  std::string file_name = model_name + ".prototext";
  El::Int file_name_index = 1;
  while (std::ifstream(file_name.c_str())) {
    file_name_index++;
    file_name = (model_name
                 + "_" + std::to_string(file_name_index)
                 + ".prototext");
  }

  //open output file
  std::ofstream out(file_name.c_str());
  if (!out.is_open()) {
    std::stringstream err;
    err << "failed to open file (" << file_name << ") for writing";
    LBANN_ERROR(err.str());
  }
  std::cout << std::endl << "writing options and prototext to file: " << file_name << "\n\n";

  //output all data
  out << "# cmd line for original experiment:\n#  $ ";
  for (int h=0; h<argc; h++) {
    out << argv[h] << " ";
  }
  std::string lbann_version("unknown: LBANN_VERSION is not defined");

#ifdef LBANN_VERSION
  lbann_version = LBANN_MAKE_STR(LBANN_VERSION);
#endif

  std::time_t r = std::time(nullptr);
  char *tm = std::ctime(&r);
  size_t fixme = strlen(tm);
  tm[fixme-1] = 0;
  out << "\n#\n# Experiment conducted at: "
      <<  tm
      << "\n#\n#\n# Experiment was run with lbann version: "
      << lbann_version << "\n#\n#\n# To rerun the experiment: \n"
      << "#  $ srun -n" << comm->get_procs_in_world() << " " << argv[0]
      << " --prototext=" << file_name << "\n#\n#\n";

  out << "# Selected SLURM Environment Variables:\n";
  std::vector<std::string> v = {"HOST", "SLURM_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE"};
  for (auto & i : v) {
    char *c = std::getenv(i.c_str());
    if (c != nullptr) {
      out << "# " << i << "=" << c << std::endl;
    }
  }
  out << "\n#\n#\n";

  std::string s;
  google::protobuf::TextFormat::PrintToString(p, &s);
  out << s;
  out.close();
}

} // namespace lbann
