import functools
import math
import operator
import os
import os.path
import sys
import numpy as np
import pytest

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

def make_random_array(shape, seed):
    """Hacked function to generate a random array.

    NumPy's RNG produces different values with different NumPy
    versions. This function is helpful when array values must be
    identical across all runs, e.g. when checking against precomputed
    metric values.

    Args:
        shape (Iterable of int): Array dimensions
        seed (int): Parameter for RNG. Must be non-zero.
    Returns:
        numpy.ndarray: Array of `np.float32`. Values will be in
            [-0.5,0.5).

    """
    size = functools.reduce(operator.mul, shape)
    eps = np.finfo(np.float32).eps
    x = (seed / np.linspace(math.sqrt(eps), 0.1, size)) % 1 - 0.5
    return x.reshape(shape).astype(np.float32)

# Data
_num_samples = 23
_sample_dims = [6,11,7]
_sample_size = functools.reduce(operator.mul, _sample_dims)
_output_size = 3
_samples = make_random_array([_num_samples] + _sample_dims, 7)

# Sample access functions
def get_sample(index):
    return _samples[index,:].reshape(-1)
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    gpus_per_node = tools.gpus_per_node(lbann)
    if gpus_per_node == 0:
        e = 'this test requires GPUs.'
        print('Skip - ' + e)
        pytest.skip(e)

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0),
                              name='input_weights')
    x = lbann.Sum(lbann.Reshape(lbann.Input(),
                                dims=tools.str_list(_sample_dims)),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=tools.str_list(_sample_dims)))

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # Convolution/FC settings
    kernel_dims = (5, _sample_dims[0], 3, 3)
    strides = (1, 1)
    pads = (1, 1)
    dilations = (1, 1)
    kernel = make_random_array(kernel_dims, 11)
    fc_input_size = kernel_dims[0] * np.prod(_sample_dims[1:])
    linearity = make_random_array((_output_size, fc_input_size), 13)

    # Weight values
    kernel_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(kernel))),
        name='kernel1'
    )
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(linearity, order='F'))
        )
    )

    def create_bn_weights(layer_name, num_channels):
        weights_ary = []
        for i in range(4):
            val = make_random_array((num_channels, ), 15+i)
            weights_ary.append(lbann.Weights(
                optimizer=lbann.SGD(),
                initializer=lbann.ValueInitializer(
                    values=tools.str_list(np.nditer(val))),
                name='{}_{}'.format(layer_name, i)
            ))

        return weights_ary

    y = lbann.Convolution(
        x,
        weights=(kernel_weights, ),
        num_dims=3,
        num_output_channels=kernel_dims[0],
        has_vectors=True,
        conv_dims=tools.str_list(kernel_dims[2:]),
        conv_strides=tools.str_list(strides),
        conv_pads=tools.str_list(pads),
        conv_dilations=tools.str_list(dilations),
        has_bias=False)
    y = lbann.BatchNormalization(
        y,
        weights=create_bn_weights("bn1", kernel_dims[0]))
    y = lbann.FullyConnected(
        y,
        weights=(linearity_weights, ),
        data_layout='data_parallel',
        num_neurons=_output_size,
        has_bias=False)
    y = lbann.BatchNormalization(
        y,
        weights=create_bn_weights("bn2", _output_size))
    z = lbann.L2Norm2(y)
    obj.append(z)

    callbacks.append(lbann.CallbackKFAC(
        damping_act="1e-2",
        damping_err="2e-2 2e-3",
        damping_bn_act="3e-2",
        damping_bn_err="4e-2 4e-3",
        damping_warmup_steps=500,
        kronecker_decay=0.6,
        print_time=True,
        print_matrix=False,
        print_matrix_summary=True,
        use_pi=True,
    ))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 1
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(x),
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'train'
        )
    ])
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'test'
        )
    ])
    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
environment = {"LBANN_KEEP_ERROR_SIGNALS": 1}
for test in tools.create_tests(setup_experiment, _test_name,
                               environment=environment):
    globals()[test.__name__] = test
