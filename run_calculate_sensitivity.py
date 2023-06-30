import numpy as np
import pyuvdata
import array_sensitivity


# Set instrument parameters
antpos_filepath = "20210226W.cfg"
c = 3e8
min_freq_hz = 0.7e9
max_freq_hz = c / 0.21
freq_hz = np.mean([min_freq_hz, max_freq_hz])
tsys_k = 25
aperture_efficiency = 0.62
field_of_view_deg2 = 10.6
antenna_diameter_m = 5
freq_resolution_hz = 162.5e3
int_time_s = 15.0 * 60  # 15 minutes in each survey field
max_bl_m = None

antpos = array_sensitivity.get_antpos(antpos_filepath)
baselines_m = array_sensitivity.get_baselines(antpos)

freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)
delay_array_s = np.fft.fftshift(
    np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
)
kpar_conv_factor = array_sensitivity.get_kpar_conversion_factor(freq_hz)
max_kpar = kpar_conv_factor * np.max(delay_array_s)

kperp_conv_factor = array_sensitivity.get_kperp_conversion_factor(freq_hz)
max_baseline_wl = np.max(np.sqrt(np.sum(baselines_m**2.0, axis=1))) * max_freq_hz / c
max_kperp = kperp_conv_factor * max_baseline_wl
max_k = np.sqrt(max_kpar**2.0 + max_kperp**2.0)

# Define bin edges:
k_bin_size = 0.2
bin_edges = np.arange(0, max_k, k_bin_size)
kpar_bin_edges = np.arange(0, max_kpar, k_bin_size)
kperp_bin_edges = np.arange(0, max_kperp, k_bin_size)

# Calculate zenith-pointed thermal noise
(
    nsamples,
    binned_ps_variance,
    true_bin_edges,
    true_bin_centers,
    nsamples_2d,
    binned_ps_variance_2d,
) = array_sensitivity.delay_ps_sensitivity_analysis(
    antpos_filepath=antpos_filepath,
    min_freq_hz=min_freq_hz,
    max_freq_hz=max_freq_hz,
    tsys_k=tsys_k,
    aperture_efficiency=aperture_efficiency,
    antenna_diameter_m=antenna_diameter_m,
    freq_resolution_hz=freq_resolution_hz,
    int_time_s=int_time_s,
    max_bl_m=max_bl_m,
    k_bin_edges_1d=bin_edges,
    kpar_bin_edges=kpar_bin_edges,
    kperp_bin_edges=kperp_bin_edges,
    zenith_angle=0.0,
)
f = open("simulation_outputs/zenith_thermal_noise.npy", "wb")
np.save(f, nsamples)
np.save(f, binned_ps_variance)
np.save(f, true_bin_edges)
np.save(f, true_bin_centers)
np.save(f, nsamples_2d)
np.save(f, binned_ps_variance_2d)
f.close()

# Calculate sample variance
# Get theory PS
f = open("camb_49591724_matterpower_z0.5.dat", "r")
file_data = f.readlines()
f.close()
model_k_axis = []
ps_model_unnorm = []
for line in file_data:
    model_k_axis.append(float(line.split()[0]))
    ps_model_unnorm.append(float(line.split()[1]))
ps_model = array_sensitivity.matter_ps_to_21cm_ps_conversion(
    np.array(model_k_axis),
    np.array(ps_model_unnorm),
    0.5
)
# Do sample variance calculation
sample_variance_cube, binned_ps_sample_variance = array_sensitivity.get_sample_variance(
    ps_model,  # Units mK^2
    model_k_axis,  # Units h/Mpc
    uv_extent=max_baseline_wl,
    field_of_view_deg2=field_of_view_deg2,
    min_freq_hz=min_freq_hz,
    max_freq_hz=max_freq_hz,
    freq_resolution_hz=freq_resolution_hz,
    k_bin_edges=bin_edges,
)
f = open("simulation_outputs/sample_variance.npy", "wb")
np.save(f, sample_variance_cube)
np.save(f, binned_ps_sample_variance)
f.close()

# Calculate off-zenith thermal noise
(
    nsamples_offzenith,
    binned_ps_variance_offzenith,
    true_bin_edges_offzenith,
    true_bin_centers_offzenith,
    nsamples_2d_offzenith,
    binned_ps_variance_2d_offzenith,
) = array_sensitivity.delay_ps_sensitivity_analysis(
    antpos_filepath=antpos_filepath,
    min_freq_hz=min_freq_hz,
    max_freq_hz=max_freq_hz,
    tsys_k=tsys_k,
    aperture_efficiency=aperture_efficiency,
    antenna_diameter_m=antenna_diameter_m,
    freq_resolution_hz=freq_resolution_hz,
    int_time_s=int_time_s,
    max_bl_m=max_bl_m,
    k_bin_edges_1d=bin_edges,
    kpar_bin_edges=kpar_bin_edges,
    kperp_bin_edges=kperp_bin_edges,
    zenith_angle=60.0,
)
f = open("simulation_outputs/off_zenith_thermal_noise.npy", "wb")
np.save(f, nsamples_offzenith)
np.save(f, binned_ps_variance_offzenith)
np.save(f, true_bin_edges_offzenith)
np.save(f, true_bin_centers_offzenith)
np.save(f, nsamples_2d_offzenith)
np.save(f, binned_ps_variance_2d_offzenith)
f.close()
