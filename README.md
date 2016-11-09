# async-train

This project aims to offer a framework to train machine learning models
 defined as theano computational graphs with asynchronous algorithms.

Currently only the hogwild! algorithm [\[1\]](#ref1) is implemented.
(implementations of other algorithms can be found in [parallel-nmt](https://github.com/valentindey/parallel-nmt)
 but these lack a clean interface)

Also, currently only python3 is supported.

*async-train* supports training on multiple GPUs. While this is actually
 the main objective, it is not thoroughly tested, yet.

There are a lot of possible improvements on my todo-list, above all the 
 problem of slow data transfer to GPUs.

`mnist.py` provides an example of a logitic regression classifier
 trained for classifying the MNIST data set. While not being meaningful, 
 this exemplifies the usage of *async-train* in code form.

To see some options to run it, call

    ./mnist.py --help


<a name="ref1">[1]</a>  Recht, B., Re, C., Wright, S., & Niu, F. (2011).
 Hogwild: A lock-free approach to parallelizing stochastic gradient descent.
 In Advances in Neural Information Processing Systems (pp. 693-701).