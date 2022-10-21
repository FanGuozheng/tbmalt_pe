import numpy as np


cpdef feature_dist_ij(double[:, :, :] distances,
                      long[:, :] numbers,
                      double[:, :, :, :] out_tensor,
                      long[:] n_atoms,
                      unique_number_dict,
                      double cutoff):
    cdef long ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for jj in range(n_atoms[ibatch]):

                dist_ij = distances[ibatch, ii, jj]
                decay_ij = decay(dist_ij, cutoff)
                out_tensor[ibatch, ii, jj, 0] = decay_ij

                for kk in range(n_atoms[ibatch]):
                    if kk != ii and kk != jj:
                        dist_ik = distances[ibatch, ii, kk]
                        dist_jk = distances[ibatch, jj, kk]
                        if dist_ik < cutoff and dist_jk < cutoff:
                            decay_ik = decay(dist_ik, cutoff)
                            decay_jk = decay(dist_jk, cutoff)
                            ind = unique_number_dict[numbers[ibatch, kk]]
                            out_tensor[ibatch, ii, jj, ind] += decay_ij*decay_ik*decay_jk

    return out_tensor


cpdef feature_exp_dist_ij(double[:, :, :] position_i,
                          double[:, :, :, :] position_j,
                          long[:, :] number_i,
                          long[:, :, :] number_j,
                          double[:, :, :, :, :] out_tensor,
                          long[:] n_atom_i,
                          long[:, :] n_atom_j,
                          long[:] n_image,
                          unique_number_dict,
                          unique_pairs_dict,
                          double[:] rs,
                          double alpha,
                          double[:, :] cutoff,
                          double min_distance,
                          int n_uan,
                          int n_uap,
                          ):
    cdef long ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk, cos_ijk
    cdef double[3] dist_vec_ik, dist_vec_jk

    for ibatch in range(len(n_atom_i)):
        for im in range(len(n_image)):
            for ii in range(n_atom_i[ibatch]):
                for jj in range(n_atom_j[ibatch, im]):

                    dvec_ij_x, dvec_ij_y, dvec_ij_z = subtract(position_i[ibatch, ii], position_j[ibatch, jj, im])
                    dist_ij = get_distance_xyz(dvec_ij_x, dvec_ij_y, dvec_ij_z)

                    if dist_ij < cutoff[-1, -1] and dist_ij > min_distance:
                        decay_ij = decay(dist_ij, cutoff[-1, -1])
                        out_tensor[ibatch, ii, jj, 0] = decay_ij

                        for kk in range(n_atom_j[ibatch, im]):
                            if kk != ii and kk != jj:

                                dvec_ik_x, dvec_ik_y, dvec_ik_z = subtract(position_i[ibatch, ii], position_j[ibatch, kk, im])
                                dist_ik = get_distance_xyz(dvec_ik_x, dvec_ik_y, dvec_ik_z)
                                dvec_jk_x, dvec_jk_y, dvec_jk_z = subtract(position_j[ibatch, jj, im], position_j[ibatch, kk, im])
                                dist_jk = get_distance_xyz(dvec_jk_x, dvec_jk_y, dvec_jk_z)

                                if dist_ik < cutoff[-1, -1] and dist_jk < cutoff[-1, -1]:
                                    for irs, _rs in enumerate(rs):

                                        bool_ik = dist_ik > cutoff[irs, 0]  # and dist_ik < cutoff[irs, 1]
                                        bool_jk = dist_jk > cutoff[irs, 0]  # and dist_jk < cutoff[irs, 1]

                                        if bool_ik and bool_jk:
                                            decay_ik = decay(dist_ik, cutoff[irs, 1])
                                            decay_jk = decay(dist_jk, cutoff[irs, 1])
                                            cos_ijk = (dvec_ik_x * dvec_jk_x + dvec_ik_y * dvec_jk_y + dvec_ik_z * dvec_jk_z) / (dist_ik * dist_jk)
                                            exp_ik = exp_rs(dist_ik, _rs, alpha)
                                            exp_jk = exp_rs(dist_jk, _rs, alpha)
                                            ind = unique_number_dict[number_j[ibatch, kk]]
                                            ind_ang = unique_pairs_dict[(number_j[ibatch, jj], number_j[ibatch, kk])]
                                            out_tensor[ibatch, ii, jj, im, ind + n_uan * irs] += exp_ik*exp_jk*decay_ik*decay_jk
                                            out_tensor[ibatch, ii, jj, im, ind_ang + n_uap * irs] += cos_ijk*decay_ik*decay_jk

    return out_tensor

cpdef feature_exp_dist_ij2(double[:, :, :] distances,
                          long[:, :] numbers,
                          double[:, :, :, :, :] out_tensor,
                          long[:] n_atoms,
                          unique_number_dict,
                          double[:] rs,
                          double alpha,
                          double[:, :] cutoff):
    cdef long ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for jj in range(n_atoms[ibatch]):

                dist_ij = distances[ibatch, ii, jj]
                out_tensor[ibatch, ii, jj, 0] = decay_ij

                if dist_ij < cutoff[-1, -1]:
                    for kk in range(n_atoms[ibatch]):
                        if kk != ii and kk != jj:
                            dist_ik = distances[ibatch, ii, kk]
                            dist_jk = distances[ibatch, jj, kk]
                            for irs, _rs in enumerate(rs):
                                decay_ij = decay(dist_ij, cutoff[irs, 1])
                                bool_ik = dist_ik > cutoff[irs, 0] and dist_ik < cutoff[irs, 1]
                                bool_jk = dist_jk > cutoff[irs, 0] and dist_jk < cutoff[irs, 1]
                                exp_ij = exp_rs(dist_ij, _rs, alpha)
                                if bool_ik or bool_jk:
                                    decay_ik = decay(dist_ik, cutoff[irs, 1])
                                    decay_jk = decay(dist_jk, cutoff[irs, 1])
                                    exp_ik = exp_rs(dist_ik, _rs, alpha)
                                    exp_jk = exp_rs(dist_jk, _rs, alpha)
                                    ind = unique_number_dict[numbers[ibatch, kk]]
                                    out_tensor[ibatch, ii, jj, ind, irs] += exp_ij*exp_ik*exp_jk*decay_ij*decay_ik*decay_jk

    return out_tensor


cpdef feature_dist_i(double[:, :, :] distances,
                     long[:, :] numbers,
                     double[:, :, :] out_tensor,
                     long[:] n_atoms,
                     unique_number_dict,
                     double cutoff):
    cdef long ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for jj in range(n_atoms[ibatch]):

                if jj != ii:
                    dist_ij = distances[ibatch, ii, jj]

                    if dist_ij < cutoff:
                        decay_ij = decay(dist_ij, cutoff)
                        out_tensor[ibatch, ii, 0] = decay_ij
                        ind = unique_number_dict[numbers[ibatch, jj]]
                        out_tensor[ibatch, ii, ind] += decay_ij

    return out_tensor


cpdef feature_exp_dist_i(double[:, :, :] position_i,
                         double[:, :, :, :] position_j,
                         long[:, :, :] numbers,
                         double[:, :, :, :] out_tensor,
                         long[:] n_atoms,
                         long[:] n_atom_j,
                         long[:] n_image,
                         unique_number_dict,
                         double[:] rs,
                         double alpha,
                         double[:, :] cutoff):
    cdef long ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for im in range(len(n_atoms)):
            for ii in range(n_atoms[ibatch]):
                for jj in range(n_atom_j[ibatch]):

                    if jj != ii:
                        dvec_ij_x, dvec_ij_y, dvec_ij_z = subtract(position_i[ibatch, ii], position_j[ibatch, jj, im])
                        dist_ij = get_distance_xyz(dvec_ij_x, dvec_ij_y, dvec_ij_z)

                        for irs, _rs in enumerate(rs):
                            if dist_ij < cutoff[irs, 1]:
                                decay_ij = decay(dist_ij, cutoff[irs, 1])
                                out_tensor[ibatch, ii, 0] = decay_ij
                                exp_ij = exp_rs(dist_ij, _rs, alpha)
                                ind = unique_number_dict[numbers[ibatch, jj, im]]
                                out_tensor[ibatch, ii, ind, irs] += exp_ij*decay_ij

    return out_tensor


cpdef get_distance(double[:] position_i, double[:] position_j):
    cdef double[3] pos_ij
    cdef double dist_ij

    dist_ij = 0.0
    for ii in range(3):
        pos_ij[ii] = position_i[ii] - position_j[ii]
        dist_ij += pos_ij[ii] ** 2
    return np.sqrt(dist_ij)


cpdef get_distance_xyz(double dvec_x, double dvec_y, double dvec_z):
    return (dvec_x ** 2 + dvec_y ** 2 + dvec_z ** 2) ** 0.5


cdef subtract(double[:] pos_i, double[:] pos_j):
    return pos_i[0] - pos_j[0], pos_i[1] - pos_j[1], pos_i[2] - pos_j[2]


cpdef decay(double distance, double cutoff):
    if distance > cutoff:
        return 0.0
    else:
        return 0.5 * np.cos(np.pi * distance / cutoff) + 0.5


cpdef exp_rs(double distance, double rs, double alpha):
    return np.exp(-alpha * (distance - rs) ** 2)
