import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import warnings

import cmdstanpy
import arviz as az

from .utils import is_numbers_array,is_int


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

def infer_growth_rate(
    time=None, 
    OD=None, 
    t_ppc=None,
    N_ppc=100,
    rho_param=[1000, 1000],
    alpha_param=[0, 1],
    sigma_param=[0, 1],
    draws=2000, 
    chains=4,
    to_df=False,
    **kwargs
    ):
    """
    Use Stan to infer maximum growth rates from time series data.

    Parameters
    ----------
    time : array-like
        time points of measurements, given as array
    OD : array-like
        observed values of OD (or any other measure of which the derivative is the growth rate)
    t_ppc : array-like, default None
        Time points at which random variables are drawn from gaussian process. If None, values are interpolated
        between the minimum and maximum observed time points.
    N_ppc : int, default 100
        Number of interpolated time points if no t_ppc is None.
    rho_params : 2-element array, default [1000, 1000]
        Mean and variance of normal prior for rho.
    alpha_param : 2-element array, default [0, 1]
        Mean and variance of normal prior for alpha.
    sigma_param : 2-element array, default [0, 1]
        Mean and variance of normal prior for rho. 
    draws : int, default 2000
        Number of iterations of MCMC sampling
    chains : int, default 4
        Number of chains for MCMC sampling
    to_df : Bool, default False
        If True, return a Dataframe containing growth rates. If false, returns arviz object.
    kwargs : Dict
        Dictionary of keyword arguments given to cmdstanpy.sample()


    """
    
    # Verify  inputs
    if (time is None) or (OD is None):
        raise RuntimeError("Time series and OD series have to be given.")

    is_numbers_array(time, "time")
    is_numbers_array(OD, "OD")

    if len(time) != len(OD):
        raise RuntimeError("Length of time and OD series have to be equal.")

    for arg, name in zip([rho_param, alpha_param, sigma_param], ["rho_param", "alpha_param", "sigma_param"]):
        if arg != None:
            is_numbers_array(arg, name)
            if len(arg) != 2:
                raise TypeError("Argument {name} has to be of length 2 (mean and variance).")

    is_int(chains, "chains")
    is_int(draws, "draws")
    is_int(N_ppc, "N_ppc")

    if type(to_df) != bool:
        raise TypeError("to_df has to be of type bool.")

    # Set parameters if not given
    if t_ppc != None:
        is_numbers_array(t_ppc, "t_ppc")
    else:
        t_ppc = np.linspace(time[0], time[-1], N_ppc)
    
    path = __file__.split("probabilistic.py")[0]

    # Initiate model
    stan_model = StanModel(path + "stan/gp_growth_rate/gp_growth_rate.stan")


    stan_model.data = {
        "N" : len(time),            # number of time points
        "t": time,                  # time points where data was evaluated
        "y": OD,                    # data's optical density
        "N_predict": N_ppc,         # number of datum in PPC
        "t_predict": t_ppc,         # time points where PPC is evaluated
        "alpha_param": alpha_param, # parameters for alpha prior
        "sigma_param": sigma_param, # parameters for sigma prior
        "rho_param": rho_param,     # parameters for rho prior
    }

    stan_model.sample(iter=draws, chains=chains, **kwargs)
    intermediate = az.from_cmdstanpy(
            stan_model.samples, 
            posterior_predictive=["y_predict", "dy_predict", "f_predict"]
        )
    if to_df:
        entries = chains * draws * N_ppc
        if entries >= 5 * 10**5:
            warnings.warn("You chose to return a dataframe. This might take a while, and the dataframe is pretty large.")
            warnings.warn("More than 500000 rows. Consider working with the arviz object.")
        return _gr_inference_to_df(intermediate, t_ppc, chains, draws)
    else:
        return intermediate


def _gr_inference_to_df(az_object, t_predict, chains, draws):

    #y_predict = az_object.posterior_predictive['y_predict'].values
    dy_predict = az_object.posterior_predictive['dy_predict'].values
    #f_predict = az_object.posterior_predictive['f_predict'].values

    df = pd.DataFrame(columns=["chain", "draw", "growth_rate", "t"])
    for i in range(chains):
        for j in range(draws):
            df = pd.concat(
                [df, 
                pd.DataFrame(
                    data={
                        "chain": np.ones(len(t_predict), dtype=int) * (i + 1), 
                        "draw": np.ones(len(t_predict), dtype=int) * (j + 1), 
                        "growth_rate": dy_predict[i, j, :],
                        "t": t_predict
                        }
                    )],
                    ignore_index=True
                )

    return df