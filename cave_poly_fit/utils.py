from contextlib import contextmanager,redirect_stderr,redirect_stdout
from dataclasses import dataclass, field
import numpy as np
from os import devnull
from scipy import interpolate
from typing import List

@dataclass
class ExperimentDetails:
    """Class for encapsulating details of CaVE experiments."""
    experiment_size: int
    poly_degree: int
    problem_type: str
    alpha_values: np.ndarray
    inf_rel_norm_stanard_lower: int
    inf_rel_norm_stanard_upper: int
    inf_rel_norm_outlier_lower: int
    inf_rel_norm_outlier_upper: int
    grads_per_experiment: List = field(default_factory=list)
    fits_per_experiment: List = field(default_factory=list)

@contextmanager
def nostdout():
    """
    A context manager that redirects stdout and stderr to devnull.
    Prevents those annoying PyEPO prints about optimizing datasets
    because I wanted to have a single progress bar for the entire
    CaVE experiment.
    """

    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def fill_nan(A):
    """
    interpolate to fill nan values
    """
    A = np.array(A)
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))

    if len(good[0] > 0):
        f = interpolate.interp1d(inds[good], A[good], bounds_error=False, fill_value='extrapolate')
        B = np.where(np.isfinite(A), A, f(inds))
        return B
    else:
        return A
