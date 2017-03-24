#! /usr/bin/env python3

import click
import logging
import time
import numpy as np
from collections import OrderedDict

from async_train import train_params

# *before* importing async-train
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s %(module)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler("mnist.log"), logging.StreamHandler()])

from async_train.datasets import mnist


np.random.seed(1234)


def init_hidden_weights(shape):
    return np.random.uniform(
            low=-np.sqrt(6. / np.sum(shape)),
            high=np.sqrt(6. / np.sum(shape)),
            size=shape
        )


def init_params_mlp(*dims, dtype="float32"):
    params = OrderedDict()
    for i, shape in enumerate(zip(dims, dims[1:])):
        params["W_{}".format(i)] = init_hidden_weights(shape).astype(dtype)
        params["b_{}".format(i)] = np.zeros(shape[1], dtype=dtype)
    return params


def build_model_mlp(theano_params, **kwargs):

    import theano.tensor as T

    x = T.fmatrix("x")
    y = T.ivector("y")

    num_layers = len(theano_params) // 2

    hidden = x
    for i in range(num_layers-1):
        # inner nonlinearities could be different, sigmoid just happened to work very well during tests
        hidden = T.nnet.sigmoid(T.dot(hidden, theano_params["W_{}".format(i)]) + theano_params["b_{}".format(i)])
    pred_probs = T.nnet.softmax(T.dot(hidden, theano_params["W_{}".format(num_layers-1)]) + theano_params["b_{}".format(num_layers-1)])
    cost = T.mean(T.nnet.categorical_crossentropy(pred_probs, y))

    # additional graphs (with same input variables)
    prediction = T.argmax(pred_probs, axis=1)
    accuracy = T.mean(T.eq(prediction, y))

    return (x, y), cost, (prediction, accuracy)


def init_conv_weights(fan_in, fan_out, shape):
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-W_bound, high=W_bound, size=shape)


def init_params_conv(dtype="float32"):
    params = OrderedDict()
    params["filters_conv_0"] = init_conv_weights(25, 125, (20, 1, 5, 5)).astype(dtype)
    params["b_conv_0"] = np.zeros(20, dtype=dtype)
    params["filters_conv_1"] = init_conv_weights(500, 312, (50, 20, 5, 5)).astype(dtype)
    params["b_conv_1"] = np.zeros(50, dtype=dtype)
    params["W_hidden"] = init_hidden_weights((800, 500)).astype(dtype)
    params["b_hidden"] = np.zeros(500, dtype=dtype)
    params["W_out"] = init_hidden_weights((500, 10)).astype(dtype)
    params["b_out"] = np.zeros(10, dtype=dtype)
    return params


def build_model_conv(theano_params, **kwargs):

    import theano.tensor as T
    from theano.tensor.nnet import conv2d
    from theano.tensor.signal.pool import pool_2d

    x = T.tensor3("x")  # batch_size x dim_1 x dim_2
    y = T.ivector("y")

    # add channel dimension
    conv0_inp = x.reshape((kwargs["batch_size"], 1, 28, 28))

    inp_shape0 = (kwargs["batch_size"], 1, 28, 28)
    flt_shape0 = (20, 1, 5, 5)
    conv0 = conv2d(conv0_inp, filters=theano_params["filters_conv_0"], filter_shape=flt_shape0, input_shape=inp_shape0)
    pool0 = pool_2d(conv0, ds=(2, 2), ignore_border=True)
    conv0_out = T.tanh(pool0 + theano_params["b_conv_0"].dimshuffle("x", 0, "x", "x"))

    inp_shape1 = (kwargs["batch_size"], 20, 12, 12)
    flt_shape1 = (50, 20, 5, 5)
    conv1 = conv2d(conv0_out, filters=theano_params["filters_conv_1"], filter_shape=flt_shape1, input_shape=inp_shape1)
    pool1 = pool_2d(conv1, ds=(2, 2), ignore_border=True)
    conv1_out = T.tanh(pool1 + theano_params["b_conv_1"].dimshuffle("x", 0, "x", "x"))

    conv_flat = conv1_out.flatten(2)
    hidden = T.tanh(T.dot(conv_flat, theano_params["W_hidden"]) + theano_params["b_hidden"])

    pred_probs = T.nnet.softmax(T.dot(hidden, theano_params["W_out"]) + theano_params["b_out"])
    cost = T.mean(T.nnet.categorical_crossentropy(pred_probs, y))

    # additional graphs (with same input variables)
    prediction = T.argmax(pred_probs, axis=1)
    accuracy = T.mean(T.eq(prediction, y))

    return (x, y), cost, (prediction, accuracy)

@click.command()
@click.option("--devices", default="cpu,cpu", help="Comma separated identifiers of devices to run on.")
@click.option("--update-scheme", default="hogwild", type=click.Choice(["hogwild", "async_da", "async_agrad"]),
              help="The asynchronous update scheme to apply")
@click.option("--num-epochs", default=4, help="Number of epochs.")
@click.option("--learning-rate", default=.01, help="Learning rate to apply.")
@click.option("--model-type", default="mlp", type=click.Choice(["mlp", "conv"]), help="type of model to use")
@click.option("--hidden-dims", default="1024,512", help="Comma separated sizes of hidden layers of MLP, "
                                                        "empty for simple logistic regression. "
                                                        "Has no effect for convolutional model.")
@click.option("--batch-size", default=32)
@click.option("--save-to", default="mnist.npz", help="file to save the model to")
@click.option("--save-freq", default=250, help="save after this many updates")
def run(devices, update_scheme, num_epochs, learning_rate, model_type, hidden_dims, batch_size, save_to, save_freq):
    """Runs a small example of async-train on the MNIST data set."""

    logging.info("loading data")
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(flatten=True if model_type == "mlp" else False)

    logging.info("preparing data")
    train_batches = [(X_train[i * batch_size:(i + 1) * batch_size],
                      Y_train[i * batch_size:(i + 1) * batch_size])
                     for i in range(len(X_train) // batch_size)]

    #valid_split = int(len(data) * .8)
    #valid = data[valid_split:]
    #train = data[:valid_split]

    if model_type == "mlp":
        h_dims = [28*28] + [int(d) for d in hidden_dims.split(",")] + [10] if hidden_dims else [28*28, 10]
        params = init_params_mlp(*h_dims)
        build_model = build_model_mlp
        train_kwargs = {}
    elif model_type == "conv":
        params = init_params_conv()
        build_model = build_model_conv
        train_kwargs = {"batch_size": batch_size}
    else:
        raise ValueError("unknown model type: {}".format(model_type))

    used_devices = devices.split(",")

    start_time = time.time()

    trained_params = train_params(initial_params=params, build_model=build_model, data=train_batches,
                                  devices=used_devices, update_scheme=update_scheme, num_epochs=num_epochs,
                                  l_rate=learning_rate, shuffle=True, save_to=save_to, save_freq=save_freq,
                                  **train_kwargs)

    train_time = time.time() - start_time
    logging.info("Training took {:.4f} seconds. "
                 .format(train_time))

    # for testing, we first need to compile the relevant function
    # make sure to *not* import theano before calling the train_params function

    import theano

    tparams = OrderedDict()
    for pname, p in params.items():
        tparams[pname] = theano.shared(p, name=pname)
    (x_inp, y_inp), cost, (prediction, accuracy) = build_model(tparams)
    f_acc = theano.function([x_inp, y_inp], accuracy)
    logging.info("train accuracy before training: {}".format(f_acc(X_train, Y_train)))
    logging.info("test accuracy before training: {}".format(f_acc(X_test, Y_test)))

    # load the trained parameters into the test model
    for pname, p in trained_params.items():
        tparams[pname].set_value(p)
    logging.info("train accuracy after training: {}".format(f_acc(X_train, Y_train)))
    logging.info("test accuracy after training: {}".format(f_acc(X_test, Y_test)))

if __name__ == '__main__':
    run()
