import numpy as np
import pyuvdata
import array_sensitivity


# Set instrument parameters
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


antpos_filepaths = ["20210226W.cfg", "20210226W_core_100.cfg", "20210226W_core_200.cfg"]
wedge_extents = [90.0, 1.8]
pointing_angles = [0., 60.]

for array_config_ind, antpos_filepath in enumerate(antpos_filepaths):
    ant_config_name = (["no_core", "core100", "core2000"])[array_config_ind]

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
    k_bin_size = 0.1
    min_k = 0.02
    bin_edges = np.arange(min_k, max_k, k_bin_size)
    kpar_bin_edges = np.arange(0, max_kpar, k_bin_size)
    kperp_bin_edges = np.arange(0, max_kperp, k_bin_size)
    
    for wedge_ext_ind, wedge_ext in enumerate(wedge_extents):
        wedge_cut_name = (["horizon", "fov"])[wedge_ext_ind]
        for pointing_ang in pointing_angles:
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
                zenith_angle=pointing_ang,
                wedge_extent_deg=wedge_ext,
            )
            f = open(f"simulation_outputs/hermal_noise_{ant_config_name}_wedge_cut_{wedge_cut_name}_za_{pointing_ang}.npy", "wb")
            np.save(f, nsamples)
            np.save(f, binned_ps_variance)
            np.save(f, true_bin_edges)
            np.save(f, true_bin_centers)
            np.save(f, nsamples_2d)
            np.save(f, binned_ps_variance_2d)
            f.close()
