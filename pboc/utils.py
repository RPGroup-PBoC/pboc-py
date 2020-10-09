import numpy as np
import pandas as pd

def is_numbers_array(arg, name):
    print(locals().keys())
    if type(arg) in [pd.core.series.Series, np.ndarray]:
        if arg.dtype not in [int, float]:
            raise TypeError("Argument {} is of type {}, but elements are not of type into or float.".format(name, type(arg)))

    elif type(arg) == list:
        if any([type(a) not in [int, float] for a in arg]):
            raise TypeError("Argument {} is of type list, but not all elements are not of type into or float.".format(name))

    else:
        raise TypeError("Argument {} is of type {}, has to be array-like.".format(name, type(arg)))

def is_int(arg):
     if type(arg) not in [int, np.int_]: 
         raise TypeError("Argument {} is of type {}, but has to be integer valued.".format(name, type(arg)))
        