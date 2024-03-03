import numpy as np
import array_sensitivity
import sys


c = 3e8


def generate_ant_locs(
    antpos_filepath="W2-17.cfg",
    min_freq_hz=0.7e9,
    max_freq_hz=c / 0.21,
    n_core_antennas=200,
    core_radius_m=50.0,
    min_ant_separation=3.0,
    outfile_path="W2-17_core.cfg",
):

    if min_ant_separation**2.0 * n_core_antennas > np.pi * core_radius_m**2.0:
        print("ERROR: Core radius too small to accommodate all antennas. Exiting")
        sys.exit()

    freq_hz = np.mean([min_freq_hz, max_freq_hz])
    antpos = array_sensitivity.get_antpos(antpos_filepath)

    new_antpos = np.full((n_core_antennas, 2), np.nan, dtype=float)
    ant_ind = 0
    while ant_ind < n_core_antennas:
        ant_coords = np.random.uniform(-core_radius_m, core_radius_m, 2)
        if np.sqrt(ant_coords[0] ** 2.0 + ant_coords[1] ** 2.0) < core_radius_m:
            ant_dists = antpos - ant_coords[np.newaxis, :]
            min_dist = np.min(np.sqrt(ant_dists[:, 0] ** 2.0 + ant_dists[:, 1] ** 2.0))
            if ant_ind > 0:
                ant_dists_new = new_antpos - ant_coords[np.newaxis, :]
                min_dist_new = np.nanmin(
                    np.sqrt(ant_dists_new[:, 0] ** 2.0 + ant_dists_new[:, 1] ** 2.0)
                )
                min_dist = np.nanmin([min_dist, min_dist_new])
            if min_dist > min_ant_separation:
                new_antpos[ant_ind, :] = ant_coords
                ant_ind += 1

    f = open(antpos_filepath, "r")
    antpos_data = f.readlines()
    f.close()
    for ant_ind in range(n_core_antennas):
        antpos_data.append(
            f"{new_antpos[ant_ind, 0]} {new_antpos[ant_ind, 1]} 0 5 dsa-core-{ant_ind + 1} \n"
        )
    f = open(outfile_path, "w")
    antpos_data = f.writelines(antpos_data)
    f.close()


if __name__ == "__main__":
    # generate_ant_locs()
    #generate_ant_locs(n_core_antennas=100, core_radius_m=30, outfile_path="W2-17_core_test_100.cfg")
    #generate_ant_locs(n_core_antennas=110, core_radius_m=30, outfile_path="W2-17_core_test_110.cfg")
    #generate_ant_locs(n_core_antennas=120, core_radius_m=30, outfile_path="W2-17_core_test_120.cfg")
    #generate_ant_locs(n_core_antennas=130, core_radius_m=30, outfile_path="W2-17_core_test_130.cfg")
    #generate_ant_locs(n_core_antennas=140, core_radius_m=30, outfile_path="W2-17_core_test_140.cfg")
    generate_ant_locs(n_core_antennas=150, core_radius_m=30, outfile_path="W2-17_core_test_150.cfg")
    generate_ant_locs(n_core_antennas=160, core_radius_m=30, outfile_path="W2-17_core_test_160.cfg")
    generate_ant_locs(n_core_antennas=170, core_radius_m=30, outfile_path="W2-17_core_test_170.cfg")
    #generate_ant_locs(n_core_antennas=180, core_radius_m=30, outfile_path="W2-17_core_test_180.cfg")
    generate_ant_locs(n_core_antennas=190, core_radius_m=30, outfile_path="W2-17_core_test_190.cfg")
    generate_ant_locs(n_core_antennas=200, core_radius_m=30, outfile_path="W2-17_core_test_200.cfg")
