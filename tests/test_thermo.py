import pytest
import pboc.thermo



def test_MWC_error():
    with pytest.raises(RuntimeError):
        pboc.thermo.MWC()