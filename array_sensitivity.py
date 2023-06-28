import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate


c = 3e8


def get_cosmological_parameters():

    cosmological_parameter_dict = {
        "D_H": 3000,  # Hubble distance, units Mpc/h
        "HI rest frame wavelength": 0.21,  # 21 cm wavelength, units m
        "omega_M": 0.27,  # Matter density
        "omega_k": 0,  # Curvature
        "omega_Lambda": 0.73,  # Cosmological constant
        "omega_B": 0.04,  # Baryon density
        "mass_frac_HI": 0.015,  # Mass fraction of neutral hydrogen
        "bias": 0.75,  # Bias between matter PS and HI PS
        "h": 0.71,
    }
    # Derived quantities
    cosmological_parameter_dict["H_0"] = (
        c / cosmological_parameter_dict["D_H"]
    )  # Hubble constant, units h*s/(m Mpc)

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
    # See "Cosmological Parameters and Conversions Memo"

    cosmological_parameter_dict = get_cosmological_parameters()
    hubble_dist = cosmological_parameter_dict["D_H"]
    hubble_const = cosmological_parameter_dict["H_0"]
    rest_frame_wl = cosmological_parameter_dict["HI rest frame wavelength"]
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_k = cosmological_parameter_dict["omega_k"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]

    avg_wl = c / avg_freq_hz
    z = avg_wl / rest_frame_wl - 1

    # Line-of-sight conversion
    e_func = np.sqrt(omega_M * (1 + z) ** 3.0 + omega_k * (1 + z) ** 2.0 + omega_Lambda)
    kpar_conv_factor = (2 * np.pi * hubble_const * e_func) / (
        (1 + z) ** 2.0 * rest_frame_wl
    )
    return kpar_conv_factor


def get_kperp_conversion_factor(avg_freq_hz):
    # See "Cosmological Parameters and Conversions Memo"

    cosmological_parameter_dict = get_cosmological_parameters()
    hubble_dist = cosmological_parameter_dict["D_H"]
    hubble_const = cosmological_parameter_dict["H_0"]
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
    kperp_conv_factor = 2 * np.pi / dist_comoving
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


def matter_ps_to_21cm_ps_conversion(
    k_axis,  # Units h/Mpc
    matter_ps,  # Units (Mpc/h)^3
    z,  # redshift
):
    # See Pober et al. 2013
    # Produces a slight difference from the paper results. Why?

    cosmological_parameter_dict = get_cosmological_parameters()
    omega_M = cosmological_parameter_dict["omega_M"]
    omega_Lambda = cosmological_parameter_dict["omega_Lambda"]
    omega_B = cosmological_parameter_dict["omega_B"]
    mass_frac_HI = cosmological_parameter_dict["mass_frac_HI"]
    bias = cosmological_parameter_dict["bias"]
    h = cosmological_parameter_dict["h"]

    brightness_temp = (
        0.084
        * h
        * (1 + z) ** 2.0
        * (omega_M * ((1 + z) ** 3.0) + omega_Lambda) ** -0.5
        * (omega_B / 0.044)
        * (mass_frac_HI / 0.01)
    )  # Units mK
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
    delay_array_lower_bounds = np.array([
        np.mean([delay_array_s[ind], delay_array_s[ind + 1]])
        for ind in range(len(delay_array_s) - 1)
    ])
    # Add the lower bound for the lowest delay bin
    delay_array_lower_bounds = np.append(
        np.array([delay_array_s[0] - (delay_array_s[1] - delay_array_s[0]) / 2]),
        delay_array_lower_bounds
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
    k_bin_edges_1d=None,
    kpar_bin_edges=None,
    kperp_bin_edges=None,
    zenith_angle=0,
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
            np.where(np.sqrt(np.sum(np.abs(baselines_m) ** 2.0, axis=1)) < max_bl_m)[0], :
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

    # 1d binning
    wedge_mask_array = get_wedge_mask_array(
        baselines_m,
        delay_array_s,
        wedge_slope=1,
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
    for bin in range(n_kbins):
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

    return (
        nsamples,
        binned_ps_variance,
        true_bin_edges,
        true_bin_centers,
        nsamples_2d,
        binned_ps_variance_2d,
    )


def get_sample_variance(
    ps_model,  # Units mK^2
    model_k_axis,  # Units h/Mpc
    uv_extent=None,
    field_of_view_deg2=None,
    min_freq_hz=None,
    max_freq_hz=None,
    freq_resolution_hz=None,
    k_bin_edges=None,
):

    field_of_view_diameter = 2 * np.sqrt(field_of_view_deg2 / np.pi)
    uv_spacing = 0.5 * 180 / field_of_view_diameter  # Nyquist sample the FoV
    freq_array_hz = np.arange(min_freq_hz, max_freq_hz, freq_resolution_hz)

    u_coords_wl = np.arange(0, uv_extent, uv_spacing)
    u_coords_wl = np.append(-np.flip(u_coords_wl[1:]), u_coords_wl)
    v_coords_wl = np.arange(0, uv_extent, uv_spacing)

    delay_array_s = np.fft.fftshift(
        np.fft.fftfreq(len(freq_array_hz), d=freq_resolution_hz)
    )
    kx, ky, kz = uvn_to_cosmology_axis_transform(
        u_coords_wl, v_coords_wl, delay_array_s, np.mean(freq_array_hz)
    )

    # Get wedge masking
    v_coords_meshed, u_coords_meshed = np.meshgrid(v_coords_wl, u_coords_wl)
    uv_locs = np.stack((u_coords_meshed.flatten(), v_coords_meshed.flatten()), axis=-1)
    uv_locs_m = uv_locs * c / np.mean(freq_array_hz)
    wedge_mask_array = get_wedge_mask_array(
        uv_locs_m,
        delay_array_s,
        wedge_slope=1,
    )
    wedge_mask_array = np.reshape(
        wedge_mask_array, (len(u_coords_wl), len(v_coords_wl), len(delay_array_s))
    )

    distance_mat = np.sqrt(
        np.abs(kx[:, np.newaxis, np.newaxis]) ** 2.0
        + np.abs(ky[np.newaxis, :, np.newaxis]) ** 2.0
        + np.abs(kz[np.newaxis, np.newaxis, :]) ** 2.0
    )
    sample_variance_cube = np.interp(distance_mat, model_k_axis, ps_model)

    n_kbins = len(k_bin_edges) - 1
    binned_ps_variance = np.full(n_kbins, np.nan, dtype=float)
    for bin in range(n_kbins):
        use_values = np.where(
            (distance_mat > k_bin_edges[bin]) & (distance_mat <= k_bin_edges[bin + 1]) & wedge_mask_array
        )
        if len(use_values[0]) > 0:
            binned_ps_variance[bin] = (
                np.nansum(sample_variance_cube[use_values]) / len(use_values[0]) ** 2.0
            )

    return sample_variance_cube, binned_ps_variance
