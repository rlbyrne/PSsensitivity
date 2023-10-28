import numpy as np
import pyuvdata
import array_sensitivity


# Set instrument parameters
c = 3e8
min_freq_hz = 0.7e9
max_freq_hz = c / 0.21
freq_hz = np.mean([min_freq_hz, max_freq_hz])
tsys_k = 25
aperture_efficiency = 0.7
field_of_view_deg2 = 30.0
antenna_diameter_m = 5
freq_resolution_hz = 130.2e3
int_time_s = 15.0 * 60  # 15 minutes in each survey field
max_bl_m = None

run_params = [
    {
        "run name": "no_core_wedge_cut_horizon_za_0.0",
        "antpos": "W2-17.cfg",
        "wedge extent": 90.0,
        "pointing angle": 0.0,
    },
    {
        "run name": "no_core_wedge_cut_fov_za_0.0",
        "antpos": "W2-17.cfg",
        "wedge extent": 3.09,
        "pointing angle": 0.0,
    },
    {
        "run name": "no_core_wedge_cut_fov_za_60.0",
        "antpos": "W2-17.cfg",
        "wedge extent": 3.09,
        "pointing angle": 60.0,
    },
    {
        "run name": "core_wedge_cut_fov_za_0.0",
        "antpos": "W2-17_core.cfg",
        "wedge extent": 3.09,
        "pointing angle": 0.0,
    },
]

for run_ind in range(len(run_params)):

    use_params = run_params[use_params]
    antpos_filepath = use_params["antpos"]
    wedge_ext = use_params["wedge extent"]
    pointing_ang = use_params["pointing angle"]
    run_name = use_params["run name"]

    antpos = array_sensitivity.get_antpos(antpos_filepath)
    baselines_m = array_sensitivity.get_baselines(antpos)

    freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)
    delay_array_s = np.fft.fftshift(
        np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
    )
    kpar_conv_factor = array_sensitivity.get_kpar_conversion_factor(freq_hz)
    max_kpar = kpar_conv_factor * np.max(delay_array_s)

    kperp_conv_factor = array_sensitivity.get_kperp_conversion_factor(freq_hz)
    max_baseline_wl = (
        np.max(np.sqrt(np.sum(baselines_m**2.0, axis=1))) * max_freq_hz / c
    )
    max_kperp = kperp_conv_factor * max_baseline_wl
    max_k = np.sqrt(max_kpar**2.0 + max_kperp**2.0)

    # Define bin edges:
    k_bin_size = 0.1
    min_k = 0.02
    bin_edges = np.arange(min_k, max_k, k_bin_size)
    kpar_bin_edges = np.arange(0, max_kpar, k_bin_size)
    kperp_bin_edges = np.arange(0, max_kperp, k_bin_size)

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
    f = open(f"simulation_outputs/thermal_noise_{run_name}.npy", "wb")
    np.save(f, nsamples)
    np.save(f, binned_ps_variance)
    np.save(f, true_bin_edges)
    np.save(f, true_bin_centers)
    np.save(f, nsamples_2d)
    np.save(f, binned_ps_variance_2d)
    f.close()
