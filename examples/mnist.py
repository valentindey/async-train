#! /usr/bin/env python3

import click
import logging
import time
import numpy as np
from collections import OrderedDict

from async_train.datasets import mnist
from async_train import train_params


np.random.seed(1234)


def init_params():
    params = OrderedDict()
    params["weights"] = np.random.random((28*28, 10))
    params["bias"] = np.ones((10,))
    return params


def build_model(theano_params):

    import theano.tensor as T

    x = T.imatrix("x")
    y = T.ivector("y")

    pred_probs = T.nnet.softmax(T.dot(x, theano_params["weights"]) + theano_params["bias"])
    cost = -T.mean(T.log(pred_probs)[T.arange(y.shape[0]), y])

    # additional graphs
    prediction = T.argmax(pred_probs, axis=1)
    accuracy = T.mean(T.eq(prediction, y))

    return (x, y), cost, (prediction, accuracy)


@click.command()
@click.option("--devices", default="cpu", help="Comma separated identifiers of devices to run on.")
@click.option("--update-scheme", default="hogwild", type=click.Choice(["hogwild", "async_da", "async_agrad"]),
              help="The asynchronous update scheme to apply")
@click.option("--num-epochs", default=2, help="Number of epochs.")
@click.option("--learning-rate", default=.01, help="Learning rate to apply.")
@click.option("--save-to", default="mnist.npz", help="file to save the model to")
def run(devices, update_scheme, num_epochs, learning_rate, save_to):
    """Runs a small example of async-train on the MNIST data set."""

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(flatten=True)

    batch_size = 32
    data = [(X_train[i * batch_size:(i + 1) * batch_size],
             Y_train[i * batch_size:(i + 1) * batch_size])
            for i in range(len(X_train) // batch_size)]

    valid_split = int(len(data) * .8)
    valid = data[valid_split:]
    train = data[:valid_split]

    params = init_params()
    start_time = time.time()
    trained_params = train_params(params, build_model=build_model, data=train,
                                  devices=devices.split(","), update_scheme=update_scheme,
                                  num_epochs=num_epochs, l_rate=learning_rate,
                                  log_level=logging.INFO, log_file="mnist.log",
                                  valid_data=valid, valid_freq=100,
                                  save_to=save_to, save_freq=10000)
    train_time = time.time() - start_time
    print("Training took {:.4f} seconds.".format(train_time))
    print("(includes time used for compilation of theano functions)")

    # to test the trained parameters, we need to rebuild the model
    # note that we should not import theano earlier, in order
    # to not mess with the build_model function
    import theano

    def get_theano_params(params):
        theano_params = OrderedDict()
        for param_name, param in params.items():
                theano_params[param_name] = theano.shared(param, name=param_name)
        return theano_params

    (x, y), _, (_, accuracy) = build_model(get_theano_params(params))
    f_acc_untrained = theano.function([x, y], accuracy)
    print("accuracy with untrained parameters:", f_acc_untrained(X_test, Y_test))

    (x, y), _, (_, accuracy) = build_model(get_theano_params(trained_params))
    f_acc_trained = theano.function([x, y], accuracy)
    print("accuracy trained parameters:", f_acc_trained(X_test, Y_test))


if __name__ == '__main__':
    run()
