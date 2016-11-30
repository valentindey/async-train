import os

import numpy as np


def save_params(param_dict, filename, epoch_update=None):
    if epoch_update:
        fn, ext = os.path.splitext(filename)
        filename = (fn + "_epoch_{}_update_{}" + ext).format(*epoch_update)
    np.savez(filename, **param_dict)
    return filename
