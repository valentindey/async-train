import numpy as np
import ctypes
from collections import OrderedDict
from multiprocessing import Lock, Value, Array


class SharedParams(object):
    """the parameters of a model as numpy arrays arrays are stored in a dict
    {parameter_name : parameter_values}
    these key/value pairs are stored in order of insertion
    """

    def __init__(self, initial, locked=True, dtype="float32"):
        if not isinstance(initial, OrderedDict):
            raise ValueError("SharedParams requires OrderedDict as initial parameters!")
        self._locked = locked
        self._dtype = dtype
        self._ordered_keys = []
        self._arrays = {}
        self._shapes = {}
        for k, v in initial.items():
            self._ordered_keys.append(k)
            self._arrays[k] = Array(ctypes.c_float, v.flatten())
            self._shapes[k] = v.shape

    def __len__(self):
        return len(self._ordered_keys)

    def __contains__(self, key):
        return key in self._ordered_keys

    def __getitem__(self, key):
        if self._locked:
            with self._arrays[key].get_lock():
                ret = np.frombuffer(self._arrays[key].get_obj(), dtype=self._dtype) \
                    .reshape(self._shapes[key])
        else:
            ret = np.frombuffer(self._arrays[key].get_obj(), dtype=self._dtype) \
                .reshape(self._shapes[key])
        return ret

    def __setitem__(self, key, value):
        """sets the value of the given key to the given value
           currently does not allow adding values after creation
           or altering the dimension of the value
        """
        if key not in self._ordered_keys:
            raise ValueError("Adding additional data is not supported (yet)!")
        if value.shape != self._shapes[key]:
            raise ValueError("new value has shape {}, expected {}".format(value.shape,
                                                                          self._shapes[key]))

        if self._locked:
            with self._arrays[key].get_lock():
                np.frombuffer(self._arrays[key].get_obj(), dtype=self._dtype)[:] = \
                    value.astype(self._dtype).flatten()
        else:
            np.frombuffer(self._arrays[key].get_obj(), dtype=self._dtype)[:] = \
                value.astype(self._dtype).flatten()

    def __delitem__(self, key):
        self._ordered_keys.remove(key)
        self._arrays.__delitem__(key)
        self._shapes.__delitem__(key)

    def __iter__(self):
        return self.keys()

    def __repr__(self):
        return "SharedParams: {} parameters with shapes: {}"\
            .format(len(self), self._shapes)

    def keys(self):
        for k in self._ordered_keys:
            yield k

    def values(self):
        for k in self.keys():
            yield self[k]

    def items(self):
        for k in self.keys():
            yield (k, self[k])

    def as_dict(self):
        """returns an OrderedDict of parameters with the *current* values"""
        d = OrderedDict()
        for k in self:
            d[k] = self[k]
        return d


class SharedFloat(object):

    def __init__(self, val=0.):
        self.val = Value("f", val)
        self.lock = Lock()

    def set_value(self, val):
        with self.lock:
            self.val.value = val
        return self

    @property
    def value(self):
        with self.lock:
            return self.val.value


class SharedCounter(object):

    def __init__(self, val=0):
        self.val = Value("i", val)
        self.lock = Lock()

    def increment(self, val=1):
        with self.lock:
            self.val.value += val
            return self.val.value

    @property
    def value(self):
        with self.lock:
            return self.val.value
