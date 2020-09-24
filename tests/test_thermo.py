import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../')
import pboc.thermo




def test_MWC_inputs():
    with pytest.raises(RuntimeError):
        pboc.thermo.MWC()

    with pytest.raises(RuntimeError):
        pboc.thermo.MWC(effector_conc="1", ka=2, ki=3, ep_ai=1)

    with pytest.raises(RuntimeError):
        pboc.thermo.MWC(effector_conc=["1"], ka=2, ki=3, ep_ai=1)
    
    with pytest.raises(RuntimeError):
        pboc.thermo.MWC(effector_conc=np.array(["1"]), ka=2, ki=3, ep_ai=1)
    
    with pytest.raises(RuntimeError):
        pboc.thermo.MWC(effector_conc=pd.Series(["1"]), ka=2, ki=3, ep_ai=1)

    assert pboc.thermo.MWC(effector_conc=1, ka=2, ki=3, ep_ai=1)
    assert pboc.thermo.MWC(effector_conc=np.array([1]), ka=2, ki=3, ep_ai=1)
    assert pboc.thermo.MWC(effector_conc=pd.Series([1]), ka=2, ki=3, ep_ai=1)
    assert pboc.thermo.MWC(effector_conc=[1], ka=2, ki=3, ep_ai=1)
