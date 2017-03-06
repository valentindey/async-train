# async-train

This project aims to offer a framework to train machine learning models
 defined as [theano](https://github.com/Theano/theano/) computational graphs with asynchronous algorithms.
 Obviously, `theano` is a requirement.

Currently the _hogwild!_ algorithm [\[1\]](#ref1), _asynchronous dual
 averaging_ and _asynchronous AdaGrad_ [\[2\]](#ref2) are implemented.

Also, currently only python3 is supported, i.e. the code is only tested
 against this version of python.

*async-train* supports training on multiple GPUs. While this is actually
 the main objective, it is not thoroughly tested, yet.

There are a lot of possible improvements on my todo-list, above all the 
 problem of slow data transfer to GPUs.


### installation

Clone this repository `cd` into it and run one of the following

    python setup.py install
    pip install .

note that this project is under development and subject to ongoing
changes

### examples

`mnist.py` provides an example of a logitic regression classifier
 trained for classifying the MNIST data set. While not being meaningful, 
 this exemplifies the usage of *async-train* in code form.
 (_async\_da_ seems not work very well in this setting)

To see some options to run it, call

    ./mnist.py --help

There are more examples to come. They most probably all will require
 `click` for command line argument parsing.


### references

<a name="ref1">[1]</a> Recht, B., Re, C., Wright, S., & Niu, F. (2011). 
 Hogwild: A lock-free approach to parallelizing stochastic gradient descent. 
 In Advances in Neural Information Processing Systems (pp. 693-701).
 
<a name="ref2">[2]</a> Duchi, J., Jordan, M. I., & McMahan, B. (2013). 
 Estimation, optimization, and parallelism when data is sparse. 
 In Advances in Neural Information Processing Systems (pp. 2832-2840).