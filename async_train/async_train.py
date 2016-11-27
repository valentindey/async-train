# -*- coding: utf-8 -*-
import json
import multiprocessing
import os

import numpy as np
from collections import OrderedDict

import logging
import multiprocessing_logging

from .shared import SharedParams, SharedCounter, SharedFloat


LOGGER = logging.Logger(__name__)
LOG_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
stream_handler = multiprocessing_logging.MultiProcessingHandler("mp-handler-0", logging.StreamHandler())
stream_handler.setFormatter(LOG_FORMATTER)
LOGGER.addHandler(stream_handler)


def async_agrad_update(device, build_model):
    """
        function run on different processes/devices to update the shared parameters
        with asynchronous adaptive gradient (async AdaGrad)
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
                                the given function must import theano inside!
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

    LOGGER.info("({}) building model on {}".format(process_name, device))
    inputs, cost, _ = build_model(theano_params)
    grads = T.grad(cost, wrt=list(theano_params.values()))
    f_grads = theano.function(inputs, grads)

    LOGGER.info("({}) model compiled on {}, waiting for data".format(process_name, device))
    while True:

        cur_data = data_queue.get()

        if cur_data == "STOP":
            LOGGER.info("({}) terminating".format(process_name))
            break

        cur_eta = learning_rate.get_value()

        # in the first iteration, all zs are still all zero and x would become all zero as well
        # this solution, while trivial, is not desired here!
        # the workaround is to "skip" the first update
        # TODO: this workaround might not be the best solution to this problem
        if update_count.get_value() > 0:
            for pname, p in shared_params.as_dict().items():
                # get current z and s from shared vars to compute the argmin in closed form
                G = np.sqrt(shared_s[pname])
                G[G == 0] = .00001  # fudge factor to avoid NaNs through 0 in denominator
                shared_params[pname] = (-cur_eta * shared_z[pname]) / G
                # set the new values of TPARAMS
                theano_params[pname].set_value(shared_params[pname])
        # note: shared_params and update_count may not be in sync
        # i.e. the current params do not correspond to the params at timestep t=update_count

        grads = [np.asarray(g) for g in f_grads(*cur_data)]

        for param_name, grad in zip(shared_z.keys(), grads):
            shared_z[param_name] += grad
            shared_s[param_name] += grad*grad

        # send information about update to main process
        num_update = update_count.increment().get_value()
        update_notify_queue.put(num_update)


def async_da_update(device, build_model):
    """
        function run on different processes/devices to update the shared parameters
        with asynchronous dual averaging
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
                                the given function must import theano inside!
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

    LOGGER.info("({}) building model on {}".format(process_name, device))
    inputs, cost, _ = build_model(theano_params)
    grads = T.grad(cost, wrt=list(theano_params.values()))
    f_grads = theano.function(inputs, grads)

    LOGGER.info("({}) model compiled on {}, waiting for data".format(process_name, device))
    while True:

        cur_data = data_queue.get()

        if cur_data == "STOP":
            LOGGER.info("({}) terminating".format(process_name))
            break

        cur_eta = learning_rate.get_value()

        # in the first iteration, all zs are still all zero and x would become all zero as well
        # this solution, while trivial, is not desired here!
        # the workaround is to "skip" the first update
        # TODO: this workaround might not be the best solution to this problem
        if update_count.get_value() > 0:
            for param_name, param in shared_params.as_dict().items():
                shared_params[param_name] = -cur_eta * shared_z[param_name]
                theano_params[param_name] = shared_params[param_name]
        # note: shared_params and update_count may not be in sync
        # i.e. the current params do not correspond to the params at timestep t=update_count

        grads = [np.asarray(g) for g in f_grads(*cur_data)]

        for param_name, grad in zip(shared_z.keys(), grads):
            shared_z[param_name] += grad

        # send information about update to main process
        num_update = update_count.increment().get_value()
        update_notify_queue.put(num_update)


def hogwild_update(device, build_model):
    """
    function run on different processes/devices to update the shared parameters
    hogwild! style, i.e. parallelized SGD without any locks
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
                            return value. This *must* be present but can be `None`)
                            the given function must import theano inside!
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

    LOGGER.info("({}) building model on {}".format(process_name, device))
    inputs, cost, _ = build_model(theano_params)
    grads = T.grad(cost, wrt=list(theano_params.values()))
    f_grads = theano.function(inputs, grads)

    LOGGER.info("({}) model compiled on {}, waiting for data".format(process_name, device))
    while True:

        cur_data = data_queue.get()

        if cur_data == "STOP":
            LOGGER.info("({}) terminating".format(process_name))
            break

        # get current parameters from shared parameters to compute gradient with
        push_to_tparams(shared_params.as_dict())

        # calculating the gradients with current data
        # we might need to cast to numpy arrays when working on GPU
        # the results could be wrapped in CudaNdArrays
        cur_grads = [np.asarray(g) for g in f_grads(*cur_data)]

        # this can not be achieved with the updates parameter of theano.function()
        # as we update the shared parameters which are not stored as theano.shared
        for param_name, grad in zip(shared_params.keys(), cur_grads):
            # apply standard SGD rule p <- p - learning_rate*gradient
            shared_params[param_name] -= learning_rate.get_value() * grad

        # send information about update to main process
        num_update = update_count.increment().get_value()
        update_notify_queue.put(num_update)


def train_params(initial_params, build_model, data, devices, update_scheme="hogwild",
                 num_epochs=10, l_rate=.01, log_level=logging.WARNING, log_file=None,
                 valid_data=None, valid_freq=5000, patience=5,
                 save_to=None, save_freq=5000):
    """

    :param initial_params:      initial parameters as OrderedDict
                                {parameter_name: numpy_array}
                                note that ALL parameters will be updated according to the same update rule without
                                exceptions
    :param build_model:         see `async_update`
    :param data:                data points used for training as the compiled cost function expects it
                                can be an iterator, generator or list
                                requires tuples corresponding to the number of inputs to the cost graph
                                if mini batch training is desired, this must contain/return these batches already
    :param devices:             list of devices to run training on as expected by theano
                                see `theano.sandbox.cuda.run`
    :param update_scheme        the update scheme to apply, one of 'hogwild', 'async_da' or 'async_agrad'
    :param num_epochs:          number of epochs, i.e. iterations over the training data
    :param l_rate:              the learning rate to apply
    :param log_level:           level of log messages to show, see logging module
    :param valid_data:          optional validation data
                                also requires tuples corresponding to the number of inputs to the cost graph
    :param valid_freq:          validation will be performed after this many updates, has no effect if valid_data
                                is not present
                                processing on sub-processes continues during validation, this should not be too low
                                in order to avoid slowdowns because sub-processes need to wait for validation to finish
                                (note that validation is currently only performed on CPU)
    :param patience:            perform this many validations before triggering early stopping because validation error
                                did not decrease, has no effect if valid_data is not present
    :param save_to:          the file to save the model parameters in as numpy npz file
    :param save_freq:           saves the model after this many updates, has no effect if 'model_name' is not set
    :return:                    the trained parameters
    """

    if update_scheme not in ["hogwild", "async_da", "async_agrad"]:
        raise ValueError("unsupported update scheme:" + str(update_scheme))

    LOGGER.setLevel(log_level)
    if log_file:
        file_handler = multiprocessing_logging.MultiProcessingHandler("mp-handler-1", logging.FileHandler(log_file))
        file_handler.setFormatter(LOG_FORMATTER)
        LOGGER.addHandler(file_handler)

    LOGGER.info("training parameters with {} on {} with learning rate {} for {} epochs"
                .format(update_scheme, devices, l_rate, num_epochs))

    # global variables used in the same way by all update schemes
    global data_queue, learning_rate, update_count, update_notify_queue
    learning_rate = SharedFloat(l_rate)
    update_count = SharedCounter()
    mgr = multiprocessing.Manager()
    data_queue = mgr.Queue()
    update_notify_queue = mgr.Queue()

    LOGGER.info("setting up global variables specific to update scheme")
    global shared_params
    global shared_z
    global shared_s
    if update_scheme == "hogwild":
        shared_params = SharedParams(initial_params, locked=False)
        target_func = hogwild_update
    elif update_scheme == "async_da":
        # async DA needs an additional map storing the sum of all previous updates
        # these sums are initialized all zero
        shared_params = SharedParams(initial_params, locked=True)
        target_func = async_da_update
        shared_z_zero = OrderedDict()
        for param_name, param in initial_params.items():
            shared_z_zero[param_name] = np.zeros_like(param)
        shared_z = SharedParams(shared_z_zero, locked=True)
    elif update_scheme == "async_agrad":
        # async AdaGrad needs again an additional map storing the squares of sums of all previous updates
        shared_params = SharedParams(initial_params, locked=True)
        target_func = async_agrad_update
        shared_z_zero = OrderedDict()
        shared_s_zero = OrderedDict()
        for param_name, param in initial_params.items():
            shared_z_zero[param_name] = np.zeros_like(param)
            shared_s_zero[param_name] = np.zeros_like(param)
        shared_z = SharedParams(shared_z_zero, locked=True)
        shared_s = SharedParams(shared_z_zero, locked=True)

    LOGGER.info("starting processes on {} with {}".format(devices, update_scheme))
    processes = [multiprocessing.Process(target=target_func, args=(device, build_model),
                                         name="process on {}".format(device)) for device in devices]
    for p in processes:
        p.daemon = True
        p.start()

    best_params = None
    if valid_data:
        LOGGER.info("compiling model for main process for validation")
        import theano
        theano_params = OrderedDict()
        for param_name, param in initial_params.items():
            theano_params[param_name] = theano.shared(param, name=param_name)

        def push_to_tparams(params):
            for param_name, param in params.items():
                theano_params[param_name].set_value(param)
        inputs, cost, _ = build_model(theano_params)
        f_cost = theano.function(inputs, cost)
        best_params = initial_params
        best_valid_error = np.inf
        patience_left = patience

    if save_to:
        filename, ext = os.path.splitext(save_to)
        extended_filename = filename + "_epoch_0_update_0" + ext
        LOGGER.info("saving initial model parameters to {}".format(extended_filename))
        np.savez(extended_filename, **initial_params)
        training_parameter_file = filename + ".json"
        LOGGER.info("saving training parameters to {}".format(training_parameter_file))
        with open(training_parameter_file, "w") as f:
            json.dump({"devices": devices,
                       "update_scheme": update_scheme,
                       "num_epochs": num_epochs,
                       "l_rate": l_rate},
                      f, indent=4)

    early_stop = False

    for epoch_idx in range(1, num_epochs+1):
        LOGGER.info("epoch {}/{}".format(epoch_idx, num_epochs))

        # fill up the batch queue for this epoch
        for d in data:
            data_queue.put(d)

        while not early_stop:

            # wait until a new update was made and get its number
            # this *may* be in a somewhat weird order
            # but the number returned should be almost exact
            update_idx = update_notify_queue.get()

            if update_idx % 1000 == 0:
                LOGGER.info("update {}".format(update_idx))

            if valid_data and update_idx % valid_freq == 0:
                LOGGER.info("update {}, validating".format(update_idx))
                cur_params = shared_params.as_dict()
                push_to_tparams(cur_params)
                cur_valid_error = f_cost(*valid_data)  # TODO: allow valid_data in batches
                LOGGER.info("validation error: {}".format(cur_valid_error))
                if cur_valid_error < best_valid_error:
                    LOGGER.info("validation error did decrease compared to previous best value {}".format(best_valid_error))
                    patience_left = patience
                    best_params = cur_params
                    best_valid_error = cur_valid_error
                else:
                    patience_left -= 1
                    LOGGER.info("validation error did not decrease compared to current best value {}, patience left: {}"
                                .format(best_valid_error, patience_left))

                if patience_left == 0:
                    LOGGER.info("validation error did not decrease the last {} times, triggering early stopping".format(patience))
                    # early stopping triggered
                    early_stop = True

            if save_to and update_idx % save_freq == 0:
                cur_params = shared_params.as_dict()
                filename, ext = os.path.splitext(save_to)
                extended_filename = filename + "_epoch_" + str(epoch_idx) + "_update_" + str(update_idx) + ext
                LOGGER.info("update {}, saving current model parameters to {}".format(update_idx, extended_filename))
                np.savez(extended_filename, **cur_params)

            if data_queue.empty() or early_stop:
                break

        if early_stop:
            # break out of epoch loop and remove all left data points from the queue, no further processing necessary
            while not data_queue.empty():
                data_queue.get()
            break

    LOGGER.info("training finished")
    for _ in processes:
        data_queue.put("STOP")
    # daemon processes will get joined automatically

    if save_to:
        LOGGER.info("saving best model parameters to {}".format(save_to))
        np.savez(filename, **best_params)

    return best_params or shared_params.as_dict()
