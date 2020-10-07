import numpy as np
import pandas as pd
import pickle
import os

import cmdstanpy

from .utils import is_number_array


class StanModel(object):
    r"""
    Custom StanModel class for crafting and sampling from Stan
    models using cmdstanpy.
    """

    def __init__(self, file, data_dict=None, samples=None, force_compile=False):
        """
        Parameters
        ----------
        model: str
            Relative path to saved Stan model code. To deter bad habits,
            this class does not accept a string as the model code. Save 
            your Stan models. 
        data_dict: dictionary
            Dictonary of all data block parameters for the model.
        force_compile: bool
            If True, model will be forced to compile. If False, 
            a precompiled file will be loaded if present. 
        """
        if ".pkl" in file:
            s = self._load(file)
            self.model = s[0]
            self.samples = s[1]
        else:
            self.model = self.loadStanModel(file, force=force_compile)
            self.data = data_dict
            self.samples = samples
            self.df = None


    def loadStanModel(self, fname, force=False):
        """Loads a precompiled Stan model. If no compiled model is found, one will be saved."""
        # Identify the model name and directory structure
        rel, sm_dir = fname.split("stan/")
        sm_name = sm_dir.split(".stan")[0]
        pkl_name = f"{rel}/stan/{sm_name}.pkl"
        # Check if the model is precompiled
        if (os.path.exists(pkl_name) == True) and (force != True):
            print("Found precompiled model. Loading...")
            model = pickle.load(open(pkl_name, "rb"))
            print("finished!")
        else:
            print("Precompiled model not found. Compiling model...")
            _path = rel + "/stan/"
            model = cmdstanpy.CmdStanModel(stan_file=fname)
            print("finished!")
            with open(pkl_name, "wb") as f:
                pickle.dump(model, f)
        return model

    def _load(self, fname):
        with open(fname, "rb") as _file:
            gurke = pickle.load(_file)
        print(type(gurke))
        if type(gurke) == cmdstanpy.model.CmdStanModel:
            model = gurke
            samples = None
        elif type(gurke) == dict:
            model = gurke[0]
            samples = gurke[1]
        return [model, samples]


    def sample(self, data_dict=None, iter=2000, chains=4, **kwargs):
        """
        Samples the assembled model given the supplied data dictionary
        and returns output as a dataframe.
        """
        if data_dict == None:
            data_dict = self.data
        self.chains = chains
        self.iter = iter
        self.samples = self.model.sample(
            data_dict, chains=chains, iter_sampling=iter, **kwargs
        )
        
        return self.samples



def infer_growth_rate(
    time, 
    OD, 
    t_ppc=None,
    N_ppc=100,
    rho_params=None,
    alpha_param=None,
    sigma_param=None,
    ):
    
    # Initiate model
    stan_model = StanModel("stan/gp_growth_rate/")

    # Set parameters if not given
    if t_ppc != None:
        is_number_array(t_ppc)
    else:
        t_ppc = np.linspacep(time[0], time[-1], N_ppc)


    stan_model.data = {
        "N" : len(time),            # number of time points
        "t": time,                  # time points where data was evaluated
        "y": OD,                    # data's optical density
        "N_predict": N_ppc,         # number of datum in PPC
        "t_predict": t_ppc,         # time points where PPC is evaluated
        "alpha_param": [0, 1],      # parameters for alpha prior
        "sigma_param": [0, 1],      # parameters for sigma prior
        "rho_param": [1000, 1000],  # parameters for rho prior
    }

    stan_model.sample()
    return stan_model

