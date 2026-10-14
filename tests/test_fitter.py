import numpy as np
import pytest
from numpy.polynomial import polynomial as P

from mercupix.fitter import LSQFit
from mercupix.mock import mock_ramps

FLUXES = np.array([20., 100., 300., 700.])         # ADU/s, one per row of the mock image
OFFSET = 1_000.
NL_LOSSES = [0., 0.01, 0.05, 0.1, 0.2, 0.4]          # fractional loss at full well (60 000 ADU)


def mock(nl_loss=0., npix=1_000, seed=0):
    """Mock ramps with one flux per row: times (ngroups,), data (ngroups, 4, npix)."""
    return mock_ramps(flux=FLUXES[:, None], shape=(FLUXES.size, npix), offset=OFFSET,
                      nl_loss=nl_loss, seed=seed)


def assert_unbiased(values, truth, nsigma=5):
    """The mean of each row is within nsigma standard errors of its truth."""
    values = np.asarray(values)
    mean = values.mean(axis=-1)
    stderr = values.std(axis=-1) / np.sqrt(values.shape[-1])
    pull = (mean - truth) / stderr
    assert np.all(np.abs(pull) < nsigma), f"pull {pull}, relative error {mean / truth - 1}"


def polyfit_reference(times, data, degree):
    """Independent per-pixel polynomial fit in t: value and slope at t = 0."""
    coefs = P.polyfit(times.astype("float64"), data.reshape(data.shape[0], -1).astype("float64"), degree)
    return coefs[0].reshape(data.shape[1:]), coefs[1].reshape(data.shape[1:])


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


# ========================== #
#  Mock: sanity checks       #
# ========================== #
def test_mock_ramps():
    """mock_ramps: shapes, noise-free truth, and the non-linearity it adds."""
    times, data = mock_ramps(flux=50., shape=(3, 4), ngroups=10, dt=2., offset=500.,
                             readnoise=0., poisson=False, nl_loss=0.)
    assert times.shape == (10,) and data.shape == (10, 3, 4) and data.dtype == np.float32
    assert np.allclose(data[:, 0, 0], 500. + 50. * times)

    _, curved = mock_ramps(flux=50., shape=(), ngroups=10, dt=2., offset=500.,
                           readnoise=0., poisson=False, nl_loss=0.1, full_well=1_000.)
    assert curved.shape == (10,)
    assert np.isclose(curved[-1], 500. + 1_000. * 0.9)         # 10 % loss at full well


# ========================== #
#  LSQFit: flux and offset   #
# ========================== #
@pytest.mark.parametrize("polytype", ["legendre", "hermite", "monomial"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_lsqfit_linear_ramps(polytype, degree):
    """Linear ramps: flux and offset are unbiased for every degree and basis."""
    times, data = mock(nl_loss=0.)
    res = LSQFit(data, times=times, degree=degree, polytype=polytype).fit()
    assert res.slope.shape == data.shape[1:]
    assert_unbiased(res.slope, FLUXES)
    assert_unbiased(res.offset, OFFSET)


@pytest.mark.parametrize("nl_loss", NL_LOSSES)
@pytest.mark.parametrize("polytype", ["legendre", "hermite", "monomial"])
@pytest.mark.parametrize("degree", [2, 3])
def test_lsqfit_nonlinear_ramps(nl_loss, polytype, degree):
    """Non-linear ramps: from degree 2, the flux is the slope at t = 0, whatever the non-linearity."""
    times, data = mock(nl_loss=nl_loss)
    res = LSQFit(data, times=times, degree=degree, polytype=polytype).fit()
    assert_unbiased(res.slope, FLUXES)
    assert_unbiased(res.offset, OFFSET)


@pytest.mark.parametrize("nl_loss", [0.05, 0.1, 0.2, 0.4])
def test_lsqfit_degree1_is_biased_by_nonlinearity(nl_loss):
    """Degree 1 cannot follow a curved ramp: the flux of bright pixels is underestimated."""
    times, data = mock(nl_loss=nl_loss)
    slope = LSQFit(data, times=times, degree=1).fit().slope.mean(axis=-1)
    assert slope[-1] < FLUXES[-1] * (1 - nl_loss / 4)          # brightest row clearly biased
    assert np.all(np.diff(slope / FLUXES) < 0)                  # the brighter, the more biased


@pytest.mark.parametrize("polytype", ["legendre", "hermite"])
@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_lsqfit_matches_polyfit(polytype, degree):
    """Pixel by pixel, same value and slope at t = 0 as an independent numpy polynomial fit."""
    times, data = mock(nl_loss=0.2, npix=200)
    res = LSQFit(data, times=times, degree=degree, polytype=polytype).fit()
    ref_offset, ref_slope = polyfit_reference(times, data, degree)
    # well below the statistical scatter of the estimates (float32 fit)
    assert np.abs(res.slope - ref_slope).max() < 1e-2 * res.slope.std(axis=-1).min()
    assert np.abs(res.offset - ref_offset).max() < 1e-2 * res.offset.std(axis=-1).min()


@pytest.mark.parametrize("shape, degree", [((50,), 2),          # 2D data (ngroups, npix)
                                           ((3, 40), 2),        # ny = degree + 1
                                           ((2, 2), 1),         # all axes of length degree + 1
                                           ((), 2)])            # a single ramp
def test_lsqfit_input_shapes(shape, degree):
    """Shapes that used to confuse the coefficient parsing."""
    times, data = mock_ramps(flux=300., shape=shape, nl_loss=0.1, seed=1)
    res = LSQFit(data, times=times, degree=degree).fit()
    ref_offset, ref_slope = polyfit_reference(times, data, degree)
    assert np.shape(res.slope) == shape
    assert np.allclose(res.slope, ref_slope, rtol=1e-3, atol=1e-2)


def test_lsqfit_herm_alias():
    """polytype='herm' (short name used elsewhere in LSQFit) works like 'hermite'."""
    times, data = mock(nl_loss=0.1, npix=100)
    herm = LSQFit(data, times=times, degree=2, polytype="herm").fit()
    hermite = LSQFit(data, times=times, degree=2, polytype="hermite").fit()
    assert np.allclose(herm.slope, hermite.slope)
