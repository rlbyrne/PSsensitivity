import numpy as np
import pyuvdata
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate
from tqdm import tqdm
import warnings
from numpy.typing import NDArray


class Sensitivity:

    def __init__(
        self,
        antpos_filepath: str = None,
        min_freq_hz: float = None,
        max_freq_hz: float = None,
        freq_resolution_hz: float = None,
        tsys_k: float = None,
        int_time_s: float = None,
        dish_antenna: bool = None,
        aperture_efficiency: float | None = None,  # Required if dish_antenna is True
        antenna_diameter_m: float | None = None,  # Required if dish_antenna is True
        use_max_bl_m: float | None = None,
        k_bin_edges_1d: NDArray[float] | None = None,
        kpar_bin_edges: NDArray[float] | None = None,
        kperp_bin_edges: NDArray[float] | None = None,
        wedge_extent_deg: float = 90.0,
        zenith_angle: float = 0,
        field_of_view_deg2: float = None,
    ):

        self.cosmological_params = get_cosmological_parameters()

        if max_freq_hz > self.cosmological_params["HI rest frame freq"]:
            raise ValueError(f"Frequency range exceeds rest frame frequency.")

        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.freq_resolution_hz = freq_resolution_hz
        if (self.min_freq_hz is not None) and (self.max_freq_hz is not None):
            self.avg_freq_hz = np.mean([min_freq_hz, max_freq_hz])
            self.wavelength_m = self.cosmological_params["c"] / self.avg_freq_hz
            self.redshift = (
                self.wavelength_m / self.cosmological_params["HI rest frame wavelength"]
            ) - 1
        else:
            self.avg_freq_hz = None
            self.wavelength_m = None
            self.redshift = None
        if self.freq_resolution_hz is None:
            self.freq_array_hz = None
            self.delay_array_s = None
        else:
            self.freq_array_hz = np.arange(
                self.min_freq_hz,
                self.max_freq_hz + self.freq_resolution_hz,
                self.freq_resolution_hz,
            )
            self.delay_array_s = np.fft.fftshift(
                np.fft.fftfreq(len(self.freq_array_hz), d=self.freq_resolution_hz)
            )

        self.tsys_k = tsys_k
        self.int_time_s = int_time_s
        self.dish_antenna = dish_antenna
        self.aperture_efficiency = aperture_efficiency
        self.antenna_diameter_m = antenna_diameter_m

        self.use_max_bl_m = use_max_bl_m
        if antpos_filepath is None:
            self.antpos = None
            self.baselines_m = None
        else:
            if antpos_filepath.endswith(".txt"):
                self.antpos = get_antpos_from_txt(antpos_filepath)
            else:
                self.antpos = get_antpos(antpos_filepath)
            self.get_baselines()

        self.tsys_k = tsys_k
        self.int_time_s = int_time_s
        self.dish_antenna = dish_antenna
        self.aperture_efficiency = aperture_efficiency
        self.antenna_diameter_m = antenna_diameter_m

        self.visibility_stddev_mk = None
        self.kpar_conv_factor = None
        self.kperp_conv_factor = None

        self.k_bin_edges_1d = k_bin_edges_1d
        self.kpar_bin_edges = kpar_bin_edges
        self.kperp_bin_edges = kperp_bin_edges
        self.wedge_extent_deg = wedge_extent_deg
        self.zenith_angle = zenith_angle

        self.kx = None
        self.ky = None
        self.kz = None

        self.nsamples_2d = None
        self.nsamples = None
        self.true_bin_centers = None
        self.true_bin_edges = None
        self.ps_thermal_variance = None
        self.ps_thermal_variance_2d = None

        self.field_of_view_deg2 = field_of_view_deg2
        self.kspace_corr_volume = None
        self.kspace_sampling_volume = None
        self.kspace_nsamples = None

        self.shot_noise_var = None

    def get_baselines(self):
        """
        Populates attribute baselines_m using antpos.
        """

        Nants = self.antpos.shape[0]
        i, j = np.triu_indices(Nants, k=1)
        baselines_m = self.antpos[i, :] - self.antpos[j, :]
        baselines_m[baselines_m[:, 1] < 0] *= -1

        if self.use_max_bl_m is not None:
            max_bl_sq = self.use_max_bl_m**2
            baselines_m = baselines_m[
                np.einsum("ij,ij->i", baselines_m, baselines_m) < max_bl_sq
            ]

        self.baselines_m = baselines_m

    def get_visibility_stddev(self):
        """
        Populates attribute visibility_stddev_mk.
        """

        if self.dish_antenna:
            eff_collecting_area = (
                np.pi * self.antenna_diameter_m**2.0 / 4 * self.aperture_efficiency
            )  # Assumes a circular aperture. Uses the single antenna aperture efficiency.
        else:
            eff_collecting_area = 3 * self.wavelength_m**2 / (8 * np.pi)
        visibility_rms = (
            self.wavelength_m**2.0
            * self.tsys_k
            / (eff_collecting_area * np.sqrt(self.freq_resolution_hz * self.int_time_s))
        )
        visibility_stddev_k = visibility_rms / np.sqrt(2)
        visibility_stddev_mk = 1e3 * visibility_stddev_k  # Convert from K to mK

        self.visibility_stddev_mk = visibility_stddev_mk

    def get_kpar_conversion_factor(self):

        e_func = np.sqrt(
            self.cosmological_params["omega_M"] * (1 + self.redshift) ** 3.0
            + self.cosmological_params["omega_k"] * (1 + self.redshift) ** 2.0
            + self.cosmological_params["omega_Lambda"]
        )
        kpar_conv_factor = (
            2 * np.pi * self.cosmological_params["HI rest frame freq"] * e_func
        ) / (
            self.cosmological_params["D_H"] * (1 + self.redshift) ** 2.0
        )  # Units Mpc^-1
        kpar_conv_factor /= self.cosmological_params["h"]  # Units h/Mpc

        self.kpar_conv_factor = kpar_conv_factor

    def get_kperp_conversion_factor(self):

        dist_comoving_func = lambda z, omega_M, omega_k, omega_Lambda: 1 / np.sqrt(
            omega_M * (1 + z) ** 3.0 + omega_k * (1 + z) ** 2.0 + omega_Lambda
        )
        dist_comoving_int, err = scipy.integrate.quad(
            dist_comoving_func,
            0,
            self.redshift,
            args=(
                self.cosmological_params["omega_M"],
                self.cosmological_params["omega_k"],
                self.cosmological_params["omega_Lambda"],
            ),
        )
        dist_comoving = self.cosmological_params["D_H"] * dist_comoving_int
        kperp_conv_factor = 2 * np.pi / dist_comoving  # Units Mpc^-1
        kperp_conv_factor /= self.cosmological_params["h"]  # Units h/Mpc

        self.kperp_conv_factor = kperp_conv_factor

    def get_k_coords(self):

        if self.use_max_bl_m is not None:
            use_baselines_m = self.baselines_m[
                np.where(
                    np.sqrt(np.sum(np.abs(self.baselines_m) ** 2.0, axis=1))
                    < self.use_max_bl_m
                )[0],
                :,
            ]
        else:
            use_baselines_m = self.baselines_m
        if self.zenith_angle != 0:
            use_baselines_m[:, 0] *= np.cos(np.radians(self.zenith_angle))
        baselines_wl = use_baselines_m / self.wavelength_m

        # Line-of-sight conversion
        kz = self.kpar_conv_factor * self.delay_array_s  # units h/Mpc

        # Perpendicular to the line-of-sight conversion
        kx = self.kperp_conv_factor * baselines_wl[:, 0]
        ky = self.kperp_conv_factor * baselines_wl[:, 1]

        self.kx = kx
        self.ky = ky
        self.kz = kz

    def bin_2d(self):

        kperp_dist = np.sqrt(np.abs(self.kx) ** 2.0 + np.abs(self.ky) ** 2.0)
        n_kbins_kpar = len(self.kpar_bin_edges) - 1
        n_kbins_kperp = len(self.kperp_bin_edges) - 1

        nsamples_kpar, _ = np.histogram(np.abs(self.kz), bins=self.kpar_bin_edges)
        nsamples_kperp, _ = np.histogram(kperp_dist, bins=self.kperp_bin_edges)
        nsamples_2d = np.outer(nsamples_kperp, nsamples_kpar)
        self.nsamples_2d = nsamples_2d

    def bin_1d(self):

        wedge_mask_array = get_wedge_mask_array(
            self.baselines_m,
            self.delay_array_s,
            wedge_slope=np.sin(np.radians(self.wedge_extent_deg)),
        )
        n_kbins = len(self.k_bin_edges_1d) - 1

        distance_mat = np.sqrt(
            np.abs(self.kx[:, np.newaxis]) ** 2.0
            + np.abs(self.ky[:, np.newaxis]) ** 2.0
            + np.abs(self.kz[np.newaxis, :]) ** 2.0
        )
        distances_masked = distance_mat[wedge_mask_array]

        idx = np.searchsorted(self.k_bin_edges_1d, distances_masked, side="left") - 1
        valid = (idx >= 0) & (idx < n_kbins)
        distances_masked, idx = distances_masked[valid], idx[valid]

        nsamples = np.bincount(idx, minlength=n_kbins)
        sums = np.bincount(idx, weights=distances_masked, minlength=n_kbins)

        true_bin_edges = np.full((n_kbins, 2), np.nan, dtype=float)
        true_bin_centers = np.full(n_kbins, np.nan, dtype=float)

        nonempty = nsamples > 0
        true_bin_centers[nonempty] = sums[nonempty] / nsamples[nonempty]

        order = np.argsort(idx, kind="stable")
        distances_masked_sorted = distances_masked[order]
        starts = (np.cumsum(nsamples) - nsamples)[nonempty]
        true_bin_edges[nonempty, 0] = np.minimum.reduceat(
            distances_masked_sorted, starts
        )
        true_bin_edges[nonempty, 1] = np.maximum.reduceat(
            distances_masked_sorted, starts
        )

        self.nsamples = nsamples
        self.true_bin_centers = true_bin_centers
        self.true_bin_edges = true_bin_edges

    def get_ps_sensitivity_thermal(self):

        delay_visibility_variance = self.visibility_stddev_mk**2.0 * len(
            self.freq_array_hz
        )
        ps_variance = 4.0 * delay_visibility_variance**2.0

        if self.nsamples is not None:
            self.ps_thermal_variance = ps_variance / self.nsamples

        if self.nsamples_2d is not None:
            self.ps_thermal_variance_2d = ps_variance / self.nsamples_2d

    def get_sampling_volume(self):

        field_of_view_diameter = 2 * np.sqrt(self.field_of_view_deg2 / np.pi)
        uv_corr_len = 90.0 / field_of_view_diameter  # Nyquist sample the FoV
        delay_corr_len = 1 / (self.max_freq_hz - self.min_freq_hz)
        corr_volume = (
            self.kpar_conv_factor
            * delay_corr_len
            * (self.kperp_conv_factor * uv_corr_len) ** 2.0
        )

        # Calculate sampled volume
        sampling_volumes = np.zeros(len(self.k_bin_edges_1d) - 1, dtype=float)
        k_bin_centers = (self.k_bin_edges_1d[:-1] + self.k_bin_edges_1d[1:]) / 2.0
        k_bin_sizes = self.k_bin_edges_1d[1:] - self.k_bin_edges_1d[:-1]
        below_delay_cut_inds = np.where(
            self.k_bin_edges_1d[1:]
            <= self.kpar_conv_factor / (2 * self.freq_resolution_hz)
        )[0]
        wedge_slope = np.sin(np.radians(self.wedge_extent_deg))
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
                * self.kpar_conv_factor
                / np.sqrt(
                    wedge_slope**2.0 * self.kpar_conv_factor**2.0
                    + self.kperp_conv_factor**2.0 * self.avg_freq_hz**2.0
                )
            )
        above_delay_cut_inds = np.where(
            k_bin_edges[:-1] >= self.kpar_conv_factor / (2 * self.freq_resolution_hz)
        )[0]
        if len(above_delay_cut_inds) > 0:
            sampling_volumes[above_delay_cut_inds] = np.pi * k_bin_centers[
                above_delay_cut_inds
            ] * k_bin_sizes[
                above_delay_cut_inds
            ] * self.kpar_conv_factor / self.freq_resolution_hz - (
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
            (k_bin_edges[:-1] < self.kpar_conv_factor / (2 * self.freq_resolution_hz))
            & (k_bin_edges[1:] > self.kpar_conv_factor / (2 * self.freq_resolution_hz))
        )[0]
        if len(span_delay_cut_inds) > 0:
            sampling_volumes[span_delay_cut_inds] = (
                np.pi
                / 8.0
                * self.kpar_conv_factor
                / self.freq_resolution_hz
                * (
                    2.0 * k_bin_centers[span_delay_cut_inds]
                    + k_bin_sizes[span_delay_cut_inds]
                )
                ** 2.0
                - wedge_slope
                * self.kpar_conv_factor
                / np.sqrt(
                    wedge_slope**2.0 * self.kpar_conv_factor**2.0
                    + kperp_conv_factor**2.0 * self.avg_freq_hz**2.0
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

        self.kspace_corr_volume = corr_volume
        self.kspace_sampling_volume = sampling_volumes
        self.kspace_nsamples = nsamples

    def get_shot_noise(
        self,
    ):
        """
        Calculated at z=0.5
        """

        k_bin_centers = (self.k_bin_edges[:-1] + self.k_bin_edges[1:]) / 2.0
        brightness_temp = get_brightness_temp(self.redshift)

        noise = (
            k_bin_centers**3.0
            / (2.0 * np.pi**2.0)
            * brightness_temp**2.0
            * self.cosmological_params["shot noise PS at z=0.5"]
        )
        shot_noise_var = noise**2.0 / self.kspace_nsamples

        self.shot_noise_var = shot_noise_var


def calculate_thermal_var(
    antpos_filepath: str = None,
    min_freq_hz: float = None,
    max_freq_hz: float = None,
    freq_resolution_hz: float = None,
    tsys_k: float = None,
    int_time_s: float = None,
    dish_antenna: bool = None,
    aperture_efficiency: float | None = None,  # Required if dish_antenna is True
    antenna_diameter_m: float | None = None,  # Required if dish_antenna is True
    use_max_bl_m: float | None = None,
    k_bin_edges_1d: NDArray[float] | None = None,
    kpar_bin_edges: NDArray[float] | None = None,
    kperp_bin_edges: NDArray[float] | None = None,
    wedge_extent_deg: float = 90.0,
    zenith_angle: float = 0,
):

    sensitivity_obj = Sensitivity(
        antpos_filepath=antpos_filepath,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        freq_resolution_hz=freq_resolution_hz,
        tsys_k=tsys_k,
        int_time_s=int_time_s,
        dish_antenna=dish_antenna,
        aperture_efficiency=aperture_efficiency,  # Required if dish_antenna is True
        antenna_diameter_m=antenna_diameter_m,  # Required if dish_antenna is True
        use_max_bl_m=use_max_bl_m,
        k_bin_edges_1d=k_bin_edges_1d,
        kpar_bin_edges=kpar_bin_edges,
        kperp_bin_edges=kperp_bin_edges,
        wedge_extent_deg=wedge_extent_deg,
        zenith_angle=zenith_angle,
    )
    sensitivity_obj.get_baselines()
    sensitivity_obj.get_visibility_stddev()
    sensitivity_obj.get_kpar_conversion_factor()
    sensitivity_obj.get_kperp_conversion_factor()
    sensitivity_obj.get_k_coords()
    sensitivity_obj.bin_1d()
    sensitivity_obj.bin_2d()
    sensitivity_obj.get_ps_sensitivity_thermal()
    return sensitivity_obj


def get_cosmological_parameters():

    cosmological_parameter_dict = {
        "c": 3e8,  # Speed of light in m/s
        "HI rest frame wavelength": 0.21,  # 21 cm wavelength, units m
        "omega_M": 0.3111,  # Matter density
        "omega_k": 0,  # Curvature
        "omega_Lambda": 0.6889,  # Cosmological constant
        "omega_B": 0.0490,  # Baryon density
        "mass_frac_HI": 0.015,  # Mass fraction of neutral hydrogen
        "bias": 0.75,  # Bias between matter PS and HI PS
        "h": 0.6766,  # Dimensionless Hubble constant
        "shot noise PS at z=0.5": 53.5,  # Units Mpc^3/h^3, interpolated from Boylan-kolchin+ 2009 and Spinelli+ 2020
    }
    # Derived quantities
    cosmological_parameter_dict["H_0"] = (
        cosmological_parameter_dict["h"] * 1.0e5
    )  # Hubble constant, units m / s / Mpc
    cosmological_parameter_dict["D_H"] = (
        cosmological_parameter_dict["c"] / cosmological_parameter_dict["H_0"]
    )  # Hubble distance, units Mpc
    cosmological_parameter_dict["HI rest frame freq"] = (
        cosmological_parameter_dict["c"]
        / cosmological_parameter_dict["HI rest frame wavelength"]
    )  # 21 cm rest frame frequency, units Hz

    return cosmological_parameter_dict


def get_antpos(antpos_filepath):

    f = open(antpos_filepath, "r")
    antpos_data = f.readlines()
    f.close()

    Nants = len(antpos_data) - 2
    antpos = np.zeros((Nants, 2), dtype=float)
    for ant_ind, ant in enumerate(antpos_data[2:]):
        line = ant.split(" ")
        antpos[ant_ind, 0] = line[0]
        antpos[ant_ind, 1] = line[1]

    return antpos


def get_antpos_from_txt(antpos_filepath):

    f = open(antpos_filepath, "r")
    antpos_data = f.readlines()
    f.close()

    Nants = len(antpos_data) - 1
    antpos = np.zeros((Nants, 3))
    for ant_ind, ant in enumerate(antpos_data[1:]):
        line = ant.split(",")
        antpos[ant_ind, 0] = line[1]
        antpos[ant_ind, 1] = line[2]
        antpos[ant_ind, 2] = line[3]

    # Find the best fit plane for the points and rotate into that plane
    antpos -= np.mean(antpos, axis=0)  # Center on the origin

    # SVD of the covariance matrix
    # The normal is the last singular vector
    _, _, vh = np.linalg.svd(antpos)
    normal = vh[-1]

    # Compute rotation matrix
    target = np.array([0, 0, 1])
    v = np.cross(normal, target)
    c = np.dot(normal, target)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    antpos = antpos @ R.T

    return antpos[:, :2]


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
    c=3e8,
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
