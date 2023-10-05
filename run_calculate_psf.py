import numpy as np
import pyuvdata
import array_sensitivity


c = 3e8
min_freq_hz = 0.7e9
max_freq_hz = c / 0.21
antenna_diameter_m = 5
freq_resolution_hz = 130.2e3
antpos_filepath = "W2-17.cfg"

antpos = array_sensitivity.get_antpos(antpos_filepath)
baselines_m = array_sensitivity.get_baselines(antpos)

psf, frequencies, ew_axis, ns_axis = array_sensitivity.calculate_psf(
    baselines_m=baselines_m,
    min_freq_hz=min_freq_hz,
    max_freq_hz=max_freq_hz,
    freq_resolution_hz=freq_resolution_hz,
    antenna_diameter_m=antenna_diameter_m,
)

f = open(f"simulation_outputs/psf.npy", "wb")
np.save(f, psf)
np.save(f, frequencies)
np.save(f, ew_axis)
np.save(f, ns_axis)
f.close()
