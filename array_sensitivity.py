import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate
from tqdm import tqdm


c = 3e8


def get_cosmological_parameters():

    cosmological_parameter_dict = {
        "HI rest frame wavelength": 0.21,  # 21 cm wavelength, units m
        "omega_M": 0.3111,  # Matter density
        "omega_k": 0,  # Curvature
        "omega_Lambda": 0.6889,  # Cosmological constant
        "omega_B": 0.0490,  # Baryon density
        "mass_frac_HI": 0.015,  # Mass fraction of neutral hydrogen
        "bias": 0.75,  # Bias between matter PS and HI PS
        "h": 0.6766,  # Dimensionless Hubble constant
    }
    # Derived quantities
    cosmological_parameter_dict["H_0"] = (
        cosmological_parameter_dict["h"] * 1.0e5
    )  # Hubble constant, units m / s / Mpc
    cosmological_parameter_dict["D_H"] = (
        c / cosmological_parameter_dict["H_0"]
    )  # Hubble distance, units Mpc

    return cosmological_parameter_dict


def get_antpos(antpos_filepath):

    f = open(antpos_filepath, "r")
    antpos_data = f.readlines()
    f.close()

    Nants = len(antpos_data) - 2
    antpos = np.zeros((Nants, 2))
    for ant_ind, ant in enumerate(antpos_data[2:]):
        line = ant.split(" ")
        antpos[ant_ind, 0] = line[0]
        antpos[ant_ind, 1] = line[1]

    return antpos


def get_baselines(antpos):

    Nants = np.shape(antpos)[0]
    Nbls = int((Nants**2 - Nants) / 2)
    baselines_m = np.zeros((Nbls, 2))
    bl_ind = 0
    for ant1 in range(Nants):
        for ant2 in range(ant1 + 1, Nants):
            bl_coords = antpos[ant1, :] - antpos[ant2, :]
            if bl_coords[1] < 0:
                bl_coords *= -1
            baselines_m[bl_ind, :] = bl_coords
            bl_ind += 1

    return baselines_m


def calculate_psf(
    baselines_m=None,
    min_freq_hz=None,
    max_freq_hz=None,
    freq_resolution_hz=None,
    antenna_diameter_m=None,
):

    resolution_deg = 5.0 / 60.0 / 1000
    image_extent_deg = 5.0 / 60.0
    ew_axis = np.arange(0, image_extent_deg, resolution_deg)
    ew_axis = np.append(-(ew_axis[::-1])[:-1], ew_axis)
    ns_axis = np.copy(ew_axis)
    ew_vals, ns_vals = np.meshgrid(ew_axis, ns_axis)

    frequencies = np.arange(
        min_freq_hz, max_freq_hz + freq_resolution_hz / 2, freq_resolution_hz
    )
    frequencies = np.array([np.mean(frequencies)])  # Use just one frequency
    psf = np.zeros_like(ew_vals)
    if len(frequencies) == 1:
        psf = psf[np.newaxis, :, :]
    else:
        psf = np.repeat(psf, len(frequencies), axis=0)
    chunk_size = 500
    for freq_ind, freq in enumerate(frequencies):
        print(f"Calculating frequency {freq_ind + 1} of {len(frequencies)}")
        wl = c / freq
        bessel_argument = (
            np.pi
            / wl
            * antenna_diameter_m
            * np.sin(np.radians(np.sqrt(ew_vals**2.0 + ns_vals**2.0)))
        )
        beam = (2.0 * scipy.special.jv(1, bessel_argument) / bessel_argument) ** 2.0
        baselines_wl = baselines_m / wl
        psf_no_beam = np.zeros_like(ew_vals)
        bl_ind_start = 0
        while bl_ind_start < np.shape(baselines_wl)[0]:
            psf_no_beam += np.sum(
                np.cos(
                    2
                    * np.pi
                    * baselines_wl[
                        bl_ind_start : bl_ind_start + chunk_size,
                        0,
                        np.newaxis,
                        np.newaxis,
                    ]
                    * np.radians(ew_vals[np.newaxis, :, :])
                )
                * np.cos(
                    2
                    * np.pi
                    * baselines_wl[
                        bl_ind_start : bl_ind_start + chunk_size,
                        1,
                        np.newaxis,
                        np.newaxis,
                    ]
                    * np.radians(ns_vals[np.newaxis, :, :])
                ),
                axis=0,
            )
            bl_ind_start += chunk_size
            if bl_ind_start % 10000 == 0:
                print(
                    f"{round(float(bl_ind_start) / float(len(baselines_wl)) * 100)}% completed"
                )
        psf[freq_ind, :, :] = psf_no_beam * beam

    return psf, frequencies, ew_axis, ns_axis


def get_visibility_stddev(
    freq_hz=None,
    tsys_k=None,
    aperture_efficiency=None,
    antenna_diameter_m=None,
    freq_resolution_hz=None,
    int_time_s=None,
):

    wavelength_m = c / freq_hz
    eff_collecting_area = (
        np.pi * antenna_diameter_m**2.0 / 4 * aperture_efficiency
    )  # Assumes a circular aperture. Uses the single antenna aperture efficiency. Is this right?
    visibility_rms = (
        wavelength_m**2.0
        * tsys_k
        / (eff_collecting_area * np.sqrt(freq_resolution_hz * int_time_s))
    )
    visibility_stddev_k = visibility_rms / np.sqrt(2)
    visibility_stddev_mk = 1e3 * visibility_stddev_k  # Convert from K to mK

    return visibility_stddev_mk


def get_kpar_conversion_factor(avg_freq_hz):

    cosmological_parameter_dict = get_cosmological_parameters()
    hubble_dist = cosmological_parameter_dict["D_H"]
    h = cosmological_parameter_dict["h"]
    rest_frame_wl = cosmological_parameter_dict["HI rest frame wavelength"]
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_k = cosmological_parameter_dict["omega_k"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]

    avg_wl = c / avg_freq_hz
    z = avg_wl / rest_frame_wl - 1
    rest_frame_freq = c / rest_frame_wl
    e_func = np.sqrt(omega_M * (1 + z) ** 3.0 + omega_k * (1 + z) ** 2.0 + omega_Lambda)

    kpar_conv_factor = (2 * np.pi * rest_frame_freq * e_func) / (
        hubble_dist * (1 + z) ** 2.0
    )  # Units Mpc^-1
    kpar_conv_factor /= h  # Units h/Mpc

    return kpar_conv_factor


def get_kperp_conversion_factor(avg_freq_hz):

    cosmological_parameter_dict = get_cosmological_parameters()
    hubble_dist = cosmological_parameter_dict["D_H"]
    h = cosmological_parameter_dict["h"]
    rest_frame_wl = cosmological_parameter_dict["HI rest frame wavelength"]
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_k = cosmological_parameter_dict["omega_k"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]

    avg_wl = c / avg_freq_hz
    z = avg_wl / rest_frame_wl - 1

    dist_comoving_func = lambda z, omega_M, omega_k, omega_Lambda: 1 / np.sqrt(
        omega_M * (1 + z) ** 3.0 + omega_k * (1 + z) ** 2.0 + omega_Lambda
    )
    dist_comoving_int, err = scipy.integrate.quad(
        dist_comoving_func,
        0,
        7,
        args=(
            omega_M,
            omega_k,
            omega_Lambda,
        ),
    )
    dist_comoving = hubble_dist * dist_comoving_int
    kperp_conv_factor = 2 * np.pi / dist_comoving  # Units Mpc^-1
    kperp_conv_factor /= h  # Units h/Mpc

    return kperp_conv_factor


def uvn_to_cosmology_axis_transform(
    u_coords_wl,
    v_coords_wl,
    delay_array_s,
    avg_freq_hz,
):

    # Line-of-sight conversion
    kpar_conv_factor = get_kpar_conversion_factor(avg_freq_hz)
    kz = kpar_conv_factor * delay_array_s  # units h/Mpc

    # Perpendicular to the line-of-sight conversion
    kperp_conv_factor = get_kperp_conversion_factor(avg_freq_hz)
    kx = kperp_conv_factor * u_coords_wl
    ky = kperp_conv_factor * v_coords_wl

    return kx, ky, kz


def get_brightness_temp(z):

    cosmological_parameter_dict = get_cosmological_parameters()
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]
    omega_B = cosmological_parameter_dict["omega_B"]
    mass_frac_HI = cosmological_parameter_dict["mass_frac_HI"]
    h = cosmological_parameter_dict["h"]

    brightness_temp = (
        0.084
        * h
        * (1 + z) ** 2.0
        * (omega_M * ((1 + z) ** 3.0) + omega_Lambda) ** -0.5
        * (omega_B / 0.044)
        * (mass_frac_HI / 0.01)
    )  # Units mK

    return brightness_temp


def matter_ps_to_21cm_ps_conversion(
    k_axis,  # Units h/Mpc
    matter_ps,
    z,  # redshift
):
    # See Pober et al. 2013

    cosmological_parameter_dict = get_cosmological_parameters()
    bias = cosmological_parameter_dict["bias"]

    brightness_temp = get_brightness_temp(z)
    ps = brightness_temp**2.0 * bias**2.0 * matter_ps  # Units mK^2(Mpc/h)^3
    # Convert to a dimensionless PS
    ps *= k_axis**3.0 / (2 * np.pi**2.0)

    return ps


def get_wedge_mask_array(
    baselines_m,
    delay_array_s,
    wedge_slope=1,
):

    uv_dist_s = (
        np.sqrt(np.abs(baselines_m[:, 0]) ** 2.0 + np.abs(baselines_m[:, 1]) ** 2.0) / c
    )
    delay_array_lower_bounds = np.array(
        [
            np.mean([delay_array_s[ind], delay_array_s[ind + 1]])
            for ind in range(len(delay_array_s) - 1)
        ]
    )
    # Add the lower bound for the lowest delay bin
    delay_array_lower_bounds = np.append(
        np.array([delay_array_s[0] - (delay_array_s[1] - delay_array_s[0]) / 2]),
        delay_array_lower_bounds,
    )

    wedge_dist = (
        np.abs(delay_array_lower_bounds[np.newaxis, :])
        - wedge_slope * uv_dist_s[:, np.newaxis]
    )
    wedge_mask_array = wedge_dist > 0

    return wedge_mask_array


def delay_ps_sensitivity_analysis(
    antpos_filepath=None,
    min_freq_hz=None,
    max_freq_hz=None,
    tsys_k=None,
    aperture_efficiency=None,
    antenna_diameter_m=None,
    freq_resolution_hz=None,
    int_time_s=None,
    max_bl_m=None,
    k_bin_edges_1d=None,  # required if calculate_1d is True
    kpar_bin_edges=None,  # required if calculate_2d is True
    kperp_bin_edges=None,  # required if calculate_2d is True
    wedge_extent_deg=90.0,
    zenith_angle=0,
    calculate_2d=True,
    calculate_1d=True,
    verbose=True,
):

    mean_freq_hz = np.mean([min_freq_hz, max_freq_hz])

    visibility_stddev_mk = get_visibility_stddev(
        freq_hz=mean_freq_hz,
        tsys_k=tsys_k,
        aperture_efficiency=aperture_efficiency,
        antenna_diameter_m=antenna_diameter_m,
        freq_resolution_hz=freq_resolution_hz,
        int_time_s=int_time_s,
    )
    freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)
    delay_visibility_variance = visibility_stddev_mk**2.0 * len(freq_array_hz)
    ps_variance = 4.0 * delay_visibility_variance**2.0

    antpos = get_antpos(antpos_filepath)
    baselines_m = get_baselines(antpos)
    if max_bl_m is not None:
        baselines_m = baselines_m[
            np.where(np.sqrt(np.sum(np.abs(baselines_m) ** 2.0, axis=1)) < max_bl_m)[0],
            :,
        ]
    if zenith_angle != 0:
        baselines_m[:, 0] *= np.cos(np.radians(zenith_angle))
    wavelength = c / mean_freq_hz
    baselines_wl = baselines_m / wavelength

    delay_array_s = np.fft.fftshift(
        np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
    )
    kx, ky, kz = uvn_to_cosmology_axis_transform(
        baselines_wl[:, 0],
        baselines_wl[:, 1],
        delay_array_s,
        np.mean(freq_array_hz),
    )

    # 2d binning
    if calculate_2d:
        if verbose:
            print("Calculating 2D-binned sensitivity.")
        kperp_dist = np.sqrt(np.abs(kx) ** 2.0 + np.abs(ky) ** 2.0)
        n_kbins_kpar = len(kpar_bin_edges) - 1
        n_kbins_kperp = len(kperp_bin_edges) - 1

        nsamples_kpar = np.zeros(n_kbins_kpar, dtype=int)
        for kpar_bin in range(n_kbins_kpar):
            use_values_kpar = np.where(
                (np.abs(kz) > kpar_bin_edges[kpar_bin])
                & (np.abs(kz) <= kpar_bin_edges[kpar_bin + 1])
            )
            nsamples_kpar[kpar_bin] = len(use_values_kpar[0])

        nsamples_kperp = np.zeros(n_kbins_kperp, dtype=int)
        for kperp_bin in range(n_kbins_kperp):
            use_values_kperp = np.where(
                (kperp_dist > kperp_bin_edges[kperp_bin])
                & (kperp_dist < kperp_bin_edges[kperp_bin + 1])
            )
            nsamples_kperp[kperp_bin] = len(use_values_kperp[0])
        nsamples_2d = np.outer(nsamples_kperp, nsamples_kpar)

        binned_ps_variance_2d = ps_variance / nsamples_2d
    else:
        binned_ps_variance_2d = None
        nsamples_2d = None

    # 1d binning
    if calculate_1d:
        wedge_mask_array = get_wedge_mask_array(
            baselines_m,
            delay_array_s,
            wedge_slope=np.sin(np.radians(wedge_extent_deg)),
        )
        n_kbins = len(k_bin_edges_1d) - 1
        nsamples = np.zeros(n_kbins, dtype=int)
        distance_mat = np.sqrt(
            np.abs(kx[:, np.newaxis]) ** 2.0
            + np.abs(ky[:, np.newaxis]) ** 2.0
            + np.abs(kz[np.newaxis, :]) ** 2.0
        )
        binned_ps_variance = np.full(n_kbins, np.nan, dtype=float)
        true_bin_edges = np.full((n_kbins, 2), np.nan, dtype=float)
        true_bin_centers = np.full(n_kbins, np.nan, dtype=float)
        if verbose:
            print("Calculating 1D-binned sensitivity.")
        for bin in tqdm(range(n_kbins), disable=not verbose):
            use_values = np.where(
                (distance_mat > k_bin_edges_1d[bin])
                & (distance_mat <= k_bin_edges_1d[bin + 1])
                & wedge_mask_array
            )
            nsamples[bin] = len(use_values[0])
            if nsamples[bin] > 0:
                true_bin_edges[bin, 0] = np.min(distance_mat[use_values])
                true_bin_edges[bin, 1] = np.max(distance_mat[use_values])
                true_bin_centers[bin] = np.mean(distance_mat[use_values])

        binned_ps_variance = ps_variance / nsamples
    else:
        nsamples = None
        binned_ps_variance = None
        true_bin_edges = None
        true_bin_centers = None

    return (
        nsamples,
        binned_ps_variance,
        true_bin_edges,
        true_bin_centers,
        nsamples_2d,
        binned_ps_variance_2d,
    )


def get_nsamples(
    field_of_view_deg2=None,
    min_freq_hz=None,
    max_freq_hz=None,
    freq_resolution_hz=None,
    k_bin_edges=None,
    wedge_extent_deg=90.0,
):

    field_of_view_diameter = 2 * np.sqrt(field_of_view_deg2 / np.pi)
    uv_corr_len = 90.0 / field_of_view_diameter  # Nyquist sample the FoV
    delay_corr_len = 1 / (max_freq_hz - min_freq_hz)
    freq_hz = np.mean([min_freq_hz, max_freq_hz])
    kpar_conv_factor = get_kpar_conversion_factor(freq_hz)
    kperp_conv_factor = get_kperp_conversion_factor(freq_hz)
    corr_volume = (
        kpar_conv_factor * delay_corr_len * (kperp_conv_factor * uv_corr_len) ** 2.0
    )
    print(f"Kpar correlation length: {kpar_conv_factor * delay_corr_len}")
    print(f"Kperp correlation length: {kperp_conv_factor * uv_corr_len}")
    print(f"Correlation volume: {corr_volume}")

    # Calculate sampled volume
    sampling_volumes = np.zeros(len(k_bin_edges) - 1, dtype=float)
    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2.0
    k_bin_sizes = k_bin_edges[1:] - k_bin_edges[:-1]
    below_delay_cut_inds = np.where(
        k_bin_edges[1:] <= kpar_conv_factor / (2 * freq_resolution_hz)
    )[0]
    wedge_slope = np.sin(np.radians(wedge_extent_deg))
    if len(below_delay_cut_inds) > 0:
        sampling_volumes[below_delay_cut_inds] = (
            2.0
            * np.pi
            * k_bin_centers[below_delay_cut_inds] ** 2.0
            * k_bin_sizes[below_delay_cut_inds]
            + np.pi / 6.0 * k_bin_sizes[below_delay_cut_inds] ** 3.0
        ) * (
            1.0
            - wedge_slope
            * kpar_conv_factor
            / np.sqrt(
                wedge_slope**2.0 * kpar_conv_factor**2.0
                + kperp_conv_factor**2.0 * freq_hz**2.0
            )
        )
    above_delay_cut_inds = np.where(
        k_bin_edges[:-1] >= kpar_conv_factor / (2 * freq_resolution_hz)
    )[0]
    if len(above_delay_cut_inds) > 0:
        sampling_volumes[above_delay_cut_inds] = np.pi * k_bin_centers[
            above_delay_cut_inds
        ] * k_bin_sizes[
            above_delay_cut_inds
        ] * kpar_conv_factor / freq_resolution_hz - (
            2.0
            * np.pi
            * k_bin_centers[above_delay_cut_inds] ** 2.0
            * k_bin_sizes[above_delay_cut_inds]
            + np.pi / 6.0 * k_bin_sizes[above_delay_cut_inds] ** 3.0
        ) * wedge_slope * kpar_conv_factor / np.sqrt(
            wedge_slope**2.0 * kpar_conv_factor**2.0
            + kperp_conv_factor**2.0 * freq_hz**2.0
        )
    span_delay_cut_inds = np.where(
        (k_bin_edges[:-1] < kpar_conv_factor / (2 * freq_resolution_hz))
        & (k_bin_edges[1:] > kpar_conv_factor / (2 * freq_resolution_hz))
    )[0]
    if len(span_delay_cut_inds) > 0:
        sampling_volumes[span_delay_cut_inds] = (
            np.pi
            / 8.0
            * kpar_conv_factor
            / freq_resolution_hz
            * (
                2.0 * k_bin_centers[span_delay_cut_inds]
                + k_bin_sizes[span_delay_cut_inds]
            )
            ** 2.0
            - wedge_slope
            * kpar_conv_factor
            / np.sqrt(
                wedge_slope**2.0 * kpar_conv_factor**2.0
                + kperp_conv_factor**2.0 * freq_hz**2.0
            )
            * (
                2.0
                * np.pi
                * k_bin_centers[span_delay_cut_inds] ** 2.0
                * k_bin_sizes[span_delay_cut_inds]
                + np.pi / 6.0 * k_bin_sizes[span_delay_cut_inds] ** 3.0
            )
            + np.pi
            / 12.0
            * (
                k_bin_sizes[span_delay_cut_inds]
                - 2.0 * k_bin_centers[span_delay_cut_inds]
            )
            ** 3.0
            - np.pi / 24.0 * kpar_conv_factor**3.0 / freq_resolution_hz**3.0
        )

    nsamples = sampling_volumes / corr_volume
    return nsamples


def get_sample_variance(
    ps_model,  # Units mK^2
    model_k_axis,  # Units h/Mpc
    field_of_view_deg2=None,
    min_freq_hz=None,
    max_freq_hz=None,
    freq_resolution_hz=None,
    k_bin_edges=None,
    wedge_extent_deg=90.0,
):

    nsamples = get_nsamples(
        field_of_view_deg2=field_of_view_deg2,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        freq_resolution_hz=freq_resolution_hz,
        k_bin_edges=k_bin_edges,
        wedge_extent_deg=wedge_extent_deg,
    )

    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2.0
    ps_model_interp = np.interp(k_bin_centers, model_k_axis, ps_model)
    sample_var = ps_model_interp**2.0 / nsamples

    return sample_var


def get_shot_noise(
    p_shot=53.5,  # Units Mpc^3/h^3
    field_of_view_deg2=None,
    min_freq_hz=None,
    max_freq_hz=None,
    freq_resolution_hz=None,
    k_bin_edges=None,
    wedge_extent_deg=90.0,
):

    nsamples = get_nsamples(
        field_of_view_deg2=field_of_view_deg2,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        freq_resolution_hz=freq_resolution_hz,
        k_bin_edges=k_bin_edges,
        wedge_extent_deg=wedge_extent_deg,
    )

    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2.0

    cosmological_parameter_dict = get_cosmological_parameters()
    rest_frame_wl = cosmological_parameter_dict["HI rest frame wavelength"]
    avg_wl = c / np.mean([min_freq_hz, max_freq_hz])
    z = avg_wl / rest_frame_wl - 1
    brightness_temp = get_brightness_temp(z)

    noise = k_bin_centers**3.0 / (2.0 * np.pi**2.0) * brightness_temp**2.0 * p_shot
    shot_noise_var = noise**2.0 / nsamples

    return shot_noise_var
