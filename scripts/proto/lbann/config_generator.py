import sys
import argparse

import lbann.proto as lp

def generate_config(model, data_reader, optimizer):
    """
    A wrapper function for lp.save_prototext.
    This function save a prototext to a path given via --prototext-path.
    This argument might be implicitly set by job-submission scripts.

    Users can use custom arguments as long as the arguments are parsed
    with `parse_known_args`:
    ```
    import argparse
    import lbann.config_generator as lcg

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model-arg", action="store", type=int)
        args,_ = parser.parse_known_args()

        lcg.save_prototext(
            create_model(args.model_arg),
            create_data_reader(),
            create_optimizer())
    ```
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prototext-path", type=str,
        help="Path to output a prototext",
        required=True)

    args, _ = parser.parse_known_args()

    lp.save_prototext(
        args.prototext_path,
        model=model,
        data_reader=data_reader,
        optimizer=optimizer)
