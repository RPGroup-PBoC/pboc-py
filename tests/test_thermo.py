import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../')
import pboc.thermo




def test_MWC_inputs():
    # Test error for missing arguments
    with pytest.raises(TypeError):
        pboc.thermo.MWC()

    # Test error for string input
    with pytest.raises(TypeError):
        pboc.thermo.MWC(effector_conc="1", ka=2, ki=3, ep_ai=1)
    
    # Test error for list of strings
    with pytest.raises(TypeError):
        pboc.thermo.MWC(effector_conc=["1"], ka=2, ki=3, ep_ai=1)
    
    # Test error for numpy array of strings
    with pytest.raises(TypeError):
        pboc.thermo.MWC(effector_conc=np.array(["1"]), ka=2, ki=3, ep_ai=1)
    
    # Test error for pandas series of strings
    with pytest.raises(TypeError):
        pboc.thermo.MWC(effector_conc=pd.Series(["1"]), ka=2, ki=3, ep_ai=1)

    # Test correct input types
    assert pboc.thermo.MWC(effector_conc=1, ka=2, ki=3, ep_ai=1)
    assert pboc.thermo.MWC(effector_conc=np.array([1]), ka=2, ki=3, ep_ai=1)
    assert pboc.thermo.MWC(effector_conc=pd.Series([1]), ka=2, ki=3, ep_ai=1)
    assert pboc.thermo.MWC(effector_conc=[1], ka=2, ki=3, ep_ai=1)

    # Test error for 0 inputs
    with pytest.raises(ValueError):
        pboc.thermo.MWC(effector_conc=1, ka=0, ki=3, ep_ai=1)

    with pytest.raises(ValueError):
        pboc.thermo.MWC(effector_conc=1, ka=[0], ki=3, ep_ai=1)

    # Test error for negative inputs
    with pytest.raises(ValueError):
        pboc.thermo.MWC(effector_conc=-1, ka=1, ki=3, ep_ai=1)

    with pytest.raises(ValueError):
        pboc.thermo.MWC(effector_conc=np.array([-1]), ka=1, ki=3, ep_ai=1)


def test_Simple_repression_inputs():
    # Test error for missing arguments
    with pytest.raises(TypeError):
        pboc.thermo.SimpleRepression()

    # Test error for string input
    with pytest.raises(TypeError):
        pboc.thermo.SimpleRepression(R="1", ep_r=1)
    
    # Test error for list of strings
    with pytest.raises(TypeError):
        pboc.thermo.SimpleRepression(R=["1"], ep_r=2)
    
    # Test error for numpy array of strings
    with pytest.raises(TypeError):
        pboc.thermo.SimpleRepression(R=np.array(["1"]), ep_r=2)
    
    # Test error for pandas series of strings
    with pytest.raises(TypeError):
        pboc.thermo.SimpleRepression(R=pd.Series(["1"]), ep_r=2)

    # Test correct input types
    assert pboc.thermo.SimpleRepression(R=1, ep_r=2)
    assert pboc.thermo.SimpleRepression(R=np.array([1]), ep_r=2)
    assert pboc.thermo.SimpleRepression(R=pd.Series([1]), ep_r=2)
    assert pboc.thermo.SimpleRepression(R=[1], ep_r=2)

    # Test error for negative inputs
    with pytest.raises(ValueError):
        pboc.thermo.SimpleRepression(R=-1, ep_r=2)

    with pytest.raises(ValueError):
        pboc.thermo.SimpleRepression(R=np.array([-1]), ep_r=2)

    with pytest.raises(ValueError):
        pboc.thermo.SimpleRepression(R=1, ep_r=2, n_ns=-1)

    with pytest.raises(ValueError):
        pboc.thermo.SimpleRepression(R=1, ep_r=2, n_ns=[-1])
