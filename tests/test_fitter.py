import numpy as np
from mercupix.fitter import LSQFit

def test_lsqfit():
    """ """
    rng = np.random.default_rng(0)
    npoints = 21
    times = np.arange(1, npoints, dtype="float32") * 2.8
    true_offset, true_slope, noise = 1_000., 50., 30.
    # generate noisy data.
    data = (true_offset + true_slope * times[:, None, None]
            + rng.normal(0, noise, (20, 3, 3))).astype("float32")

    # fit
    fitter = LSQFit(data, times=times, degree=1)
    _ = fitter.fit(inplace=True)           # stores the coefficients

    residuals = fitter.get_residuals()
    # model residuals
    assert ( np.abs(np.mean(residuals, axis=0)) < noise/np.sqrt(npoints) ).all()
    assert ( np.abs((noise - np.std(residuals, axis=0))) < noise ).all()
