# async-train

This project aims to offer a framework to train machine learning models
 defined with [theano](https://github.com/Theano/theano/) with
 asynchronous optimization algorithms (data parallelism).
 Obviously, `theano` is a requirement.

Currently the _hogwild!_ algorithm [\[1\]](#ref1), _asynchronous dual
 averaging_ and _asynchronous AdaGrad_ [\[2\]](#ref2) are implemented.

Also, currently only python3 is supported, i.e. the code is only tested
 against this version of python.

*async-train* supports training on multiple GPUs and works with theano's
 new [`gpuarray` backend](http://deeplearning.net/software/theano/tutorial/using_gpu.html#gpuarray)

There are a lot of improvements on my todo-list and the code is subject
 to ongoing changes.

People interested in data parallelism for theano models might also want
 to check out [platoon](https://github.com/mila-udem/platoon/).

### installation

    pip install git+http://github.com/valentindey/async-train

### examples

`mnist.py` provides an example of a logitic regression classifier and
 CNN model trained for classifying the MNIST data set.
 While not being really meaningful due to its simplicity (and fast
 training),
 this exemplifies the usage of *async-train* in code form.
 (_async\_da_ seems not work very well...)

To see some options to run it, call

    ./mnist.py --help

Needs the module [click](http://click.pocoo.org/) installed.


### references

<a name="ref1">[1]</a> Recht, B., Re, C., Wright, S., & Niu, F. (2011). 
 Hogwild: A lock-free approach to parallelizing stochastic gradient descent. 
 In Advances in Neural Information Processing Systems (pp. 693-701).
 
<a name="ref2">[2]</a> Duchi, J., Jordan, M. I., & McMahan, B. (2013). 
 Estimation, optimization, and parallelism when data is sparse. 
 In Advances in Neural Information Processing Systems (pp. 2832-2840).