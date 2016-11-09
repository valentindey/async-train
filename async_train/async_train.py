# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np
from collections import OrderedDict

from .shared import SharedParams, SharedCounter, SharedFloat


def async_update(device, build_model):
    """
    function run on different processes/devices to update the shared parameters
    (currently hogwild! style, i.e. without locks)
    builds a local copy of the model for this process/device and waits for data
    to process

    :param device:          the device identifier of the device to run this on
                            see `theano.sandbox.cuda.run`
    :param build_model:     a function returning a theano graph for the cost and
                            the corresponding inputs as TensorTypes
                            requires the parameters of the model to build as dict
                            of theano.shared variables {parameter_name: theano_shared}
                            has an additional return value that is not used here
                            but facilitates reusing this method in other places
                            list of inputs first, then the graph (and the optional
                            return value)
                            this function must import theano inside!
    """

    # importing theano only inside this function and bind it to the given device
    import theano.tensor as T
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(device)

    process_name = multiprocessing.current_process().name

    # stores the parameters for the currently run process as theano.shared
    # initialized with the initial parameter values
    theano_params = OrderedDict()
    for param_name, param in shared_params.as_dict().items():
        theano_params[param_name] = theano.shared(param, name=param_name)

    # function to update theano_params
    def push_to_tparams(params):
        for param_name, param in params.items():
            theano_params[param_name].set_value(param)

    # build the model on this process/device
    inputs, cost, _ = build_model(theano_params)
    grads = T.grad(cost, wrt=list(theano_params.values()))
    f_grads = theano.function(inputs, grads)

    # everything is compiled, this process/device is ready to process data
    while True:
        # wait for the next batch of data
        # until a stop signal is received, that terminates the process
        cur_data = data_queue.get()
        if cur_data == "STOP":
            return

        # get current parameters from shared parameters to compute gradient with
        push_to_tparams(shared_params.as_dict())

        # calculating the gradients with current data
        cur_grads = f_grads(*cur_data)

        # we might need to cast to numpy arrays when working on GPU
        # the results could be wrapped in CudaNdArrays
        cur_grads = [np.asarray(g) for g in cur_grads]

        # this can not be achieved with the updates parameter of theano.function()
        # as we update the shared parameters which are not stored as theano.shared
        for param_name, grad in zip(shared_params.keys(), cur_grads):
            # apply standard SGD rule p <- p - learning_rate*gradient
            shared_params[param_name] -= learning_rate.get_value() * grad

        # send information about update to main process
        num_update = update_count.increment().get_value()
        update_notify_queue.put((process_name, num_update))


def train_params(initial_params, build_model, data, devices, num_epochs=10):
    """

    :param initial_params:      initial parameters as OrderedDict
                                {parameter_name: numpy_array}
    :param build_model:         see `async_update`
    :param data:                data points used for training as the compiled cost function expects it
                                can be an iterator, generator or list
                                requires tuples corresponding to the number of inputs to the cost graph
                                if mini batch training is desired, this must contain/return these batches already
    :param devices:             list of devices to run training on as expected by theano
                                see `theano.sandbox.cuda.run`
    :param num_epochs:          number of epochs, i.e. iterations over the training data
    :return:
    """

    global shared_params, data_queue, learning_rate, update_count, update_notify_queue
    shared_params = SharedParams(initial_params)
    learning_rate = SharedFloat(.01)
    update_count = SharedCounter()
    mgr = multiprocessing.Manager()
    data_queue = mgr.Queue()
    update_notify_queue = mgr.Queue()

    processes = [multiprocessing.Process(target=async_update, args=(device, build_model),
                                         name="process on {}".format(device)) for device in devices]

    for p in processes:
        p.start()

    for num_epoch in range(1, num_epochs+1):

        # fill up the batch queue for this epoch
        for d in data:
            data_queue.put(d)

        while True:
            # wait until a new update was made and get its number
            # this *may* be in a somewhat weird order
            # but the number returned should be almost exact
            process_name, num_updates = update_notify_queue.get()

            # TODO: same procedure for validation, model checkpointing, ...
            # TODO: variable update indexes, that trigger these things
            if num_updates % 1000 == 0:
                print("epoch {}, update {}".format(num_epoch, num_updates))

            if data_queue.empty():
                break

    for _ in processes:
        data_queue.put("STOP")
    for p in processes:
        p.join()

    return shared_params.as_dict()
