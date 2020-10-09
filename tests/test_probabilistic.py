import pytest
import numpy as np
import pandas as pd
import os

import sys
sys.path.insert(0, '../')
import pboc.probabilistic

def test_model_loading():
    # Clean up
    if os.path.isfile('tests/stan/bernoulli'):
        os.remove("tests/stan/bernoulli")
    if os.path.isfile('tests/stan/bernoulli.hpp'):
        os.remove("tests/stan/bernoulli.hpp")
    if os.path.isfile('tests/stan/bernoulli.pkl'):
        os.remove("tests/stan/bernoulli.pkl")
    

    # Create model
    model = pboc.probabilistic.StanModel("tests/stan/bernoulli.stan")
    # Test implicit import of precompiled model
    model = pboc.probabilistic.StanModel("tests/stan/bernoulli.stan")
    # Test explicit import of precompiled model
    model = pboc.probabilistic.StanModel("tests/stan/bernoulli.pkl")


def test_model_sampling():
    # Load model
    model = pboc.probabilistic.StanModel("tests/stan/bernoulli.pkl")
    model.data = {
        "N" : 10,
        "y" : [0,1,0,0,0,0,0,0,0,1]
        }
    model.sample()


def test_growth_rate_inference_inputs():
    # Error for missing daa
    with pytest.raises(RuntimeError):
        t = [0.1, 0.2]
        pboc.probabilistic.infer_growth_rate(time=t)

    # Error for non number input
    with pytest.raises(TypeError):
        t = ["a", "b"]
        OD = [0.1, 0.2]
        pboc.probabilistic.infer_growth_rate(time=t, OD=OD)

    with pytest.raises(TypeError):
        OD = ["a", "b"]
        t = [0.1, 0.2]
        pboc.probabilistic.infer_growth_rate(time=t, OD=OD)

    # Error for data of different lengths
    with pytest.raises(RuntimeError):
        OD = [1]
        t = [0.1, 0.2]
        pboc.probabilistic.infer_growth_rate(time=t, OD=OD)

    



    
    return

