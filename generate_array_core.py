import numpy as np
import array_sensitivity

antpos_filepath = "20210226W.cfg"
c = 3e8
min_freq_hz = 0.7e9
max_freq_hz = c / 0.21
freq_hz = np.mean([min_freq_hz, max_freq_hz])
bao_scales_deg = [1.5, 4.6]
n_core_antennas = 200
core_radius_m = 50.0
min_ant_separation = 3.0

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
            min_dist_new = np.nanmin(np.sqrt(ant_dists_new[:, 0] ** 2.0 + ant_dists_new[:, 1] ** 2.0))
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
f = open("20210226W_core.cfg", "w")
antpos_data = f.writelines(antpos_data)
f.close()
