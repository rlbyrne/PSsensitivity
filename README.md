# PSsensitivity
Code that calculates the sensitivity of interferometric arrays for intensity mapping applications. For more information, see Byrne et al. 2024 "21 cm Intensity Mapping with the DSA-2000" (https://arxiv.org/abs/2311.00896).

## How to Use
The code underlying the sensitivity analysis is contained in `array_sensitivity.py`. </br>
</br>
`run_calculate_sensitivity.py` and `run_calculate_psf.py` feature scripts that perform the computationally expensive operations of calculating the thermal noise of a simulated array and generating a simulated PSF. The outputs of those operations are contained in the `simulation_outputs` directory. </br>
</br>
`plot_DSA2000_sensitivity.ipynb` generates all plots presented in Byrne et al. 2024 and serves as an example of how to use the code. Plots are saved to the `plots` directory. HTML and PDF renderings of the notebook are also provided. </br>

## Data
Data files contained in this repository are:
- `W2-17.cfg`: Configuration file containing the nominal DSA-2000 antenna positions.
- `W2-17_core.cfg`: Configuration file containing the nominal DSA-2000 antenna positions plus a randomized core of 200 additional antennas. The core antenna locations were generated with `generate_array_core.py`.
- `camb_49591724_matterpower_z0.5.dat`: Predicted power spectrum, generated with CAMB (Lewis et al. 2000).

## Dependencies
- [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata)
- numpy
- scipy
- matplotlib
