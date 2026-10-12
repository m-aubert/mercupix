# mercupix

`mercupix` processes up-the-ramp data from HxRG infrared detectors. It extracts
accurate per-pixel fluxes and returns a calibrated 2D image of the incoming flux.

The processing chain covers:

- reference pixel correction (up-down and left-right, plus a Fourier-domain SIRS-like correction),
- offset, dark, classical non-linearity and IPC corrections,
- data quality masking (saturation, charge migration, hot/warm and disconnected pixels),
- cosmic ray (jump) detection and correction,
- ramp fitting with several estimators (polynomial least-squares, Kubik et al. 2016, Brandt 2024).

## Installation

`mercupix` requires Python 3.11 or later.

Install the latest version from GitHub:

```bash
pip install git+https://github.com/m-aubert/mercupix.git
```

Or clone the repository and install it in editable mode for development:

```bash
git clone https://github.com/m-aubert/mercupix.git
cd mercupix
pip install -e .
```

## Quick start

```python
from mercupix.ramp import RawRamp
from mercupix.fitter import fit_ramp
from mercupix.image import Exposure

# 1. Load a raw up-the-ramp datacube.
#    datatype is one of 'pyxim', 'roman', 'ip2i' or None (any FITS file).
ramp = RawRamp.from_filename("my_ramp.fits", datatype="pyxim")
print(ramp.cube_shape)   # (ngroups, ny, nx), reference pixels included
print(ramp.times)        # time of each group

# 2. Preprocess the ramp: reference pixel correction and data quality mask.
#    The output is cropped to the active area: (ngroups, ny_act, nx_act).
data, var, mask = ramp.preprocess_ramp(
    corr_refpix=True,
    mask=True,
    mask_kwargs=dict(disconnected=False, saturated=True, cosmics=True),
)

# 3. Fit the ramp. fit_ramp preprocesses the ramp with `raw_ramp_kwargs`,
#    then fits it with the chosen method ('LSQ', 'Kubik' or 'Brandt').
result = fit_ramp(
    ramp,
    method="Brandt",
    raw_ramp_kwargs=dict(corr_refpix=True),
    fit_args=dict(jumps=True, jump_threshold=[20, 30]),  # detect and omit jumps
    readnoise=10,                                        # read noise in ADU
)

result.slope    # (ny_act, nx_act) count rate, in ADU per unit of `times`
result.var      # variance of the slope
result.qof      # chi-square of the fit
result.mask2d   # number of group differences kept after jump detection

# Alternatively, a polynomial least-squares fit:
result_lsq = fit_ramp(ramp, method="LSQ", degree=2)

# 4. Build and display the 2D exposure.
exposure = Exposure(data=result.slope, mask=result.mask2d)
fig = exposure.show()
```

Each correction can also be applied on its own: `RawRamp.correct_refpix`,
`subtract_offset`, `correct_dark`, `correct_nl`, `correct_ipc` and `get_mask`.
See the docstrings for details.

## Package layout

| Module      | Content                                                              |
|-------------|----------------------------------------------------------------------|
| `baseramp`  | `BaseRamp`: datacube storage, geometry, I/O and plotting             |
| `ramp`      | `RawRamp`: corrections applied before ramp fitting                   |
| `refpixels` | Up-down and left-right reference pixel corrections                   |
| `sirs`      | Fourier-domain (SIRS-like) reference pixel correction                |
| `mask`      | Data quality bitmasks, saturation and cosmic ray detection           |
| `fitter`    | Ramp fitters (`LSQFit`, `Kubik2015`, `Brandt2024`), `RampFitResults` |
| `image`     | `Exposure`: 2D images built from the fit results                     |
| `carac`     | Characterization tools: PTC, noise PSD, classical non-linearity      |
| `calib`     | Calibration effects: IPC, brighter-fatter, persistence               |
| `io`        | Readers for the supported formats and calibration products           |

## Documentation

The documentation is built with Sphinx from the `docs/` directory:

```bash
pip install -e ".[docs]"
cd docs && make html
```

## License

MIT. See [LICENSE](LICENSE).
