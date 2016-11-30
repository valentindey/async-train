from setuptools import setup, find_packages

setup(
    name="async_train",
    version="0.0.1",
    license="MIT",
    install_requires=["theano", "multiprocessing-logging"],
    packages=find_packages()
)
