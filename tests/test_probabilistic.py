import pytest
import numpy as np
import pandas as pd
import os

import sys
sys.path.insert(0, '../')
import pboc.probabilistic

def test_model_loading():
    # Clean up
    os.remove("tests/stan/bernoulli")
    os.remove("tests/stan/bernoulli.hpp")
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
    