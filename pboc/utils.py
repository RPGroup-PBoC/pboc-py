import numpy as np
import pandas as pd

def is_number_array(arg):
    if type(arg) in [pd.core.series.Series, np.ndarray]:
        if arg.dtype not in [int, float]:
            raise TypeError("Argument {} is {}, but elements are not of type into or float.".format(arg, type(arg)))

    elif type(arg) == list:
        if any([type(a) not in [int, float] for a in arg]):
            raise TypeError("Argument {} is list, but not all elements are not of type into or float.".format(arg))

    else:
        raise TypeError("Argument {} is {}, has to be array-like.")