from libc.math cimport cos, exp


cpdef decay(double distance, cutoff):
    if distance > cutoff:
        return 0.0
    else:
        return 0.5 * cos(3.141592653589793 * distance / cutoff) + 0.5


cpdef exp_dist(double distance, eta):
    return exp(-eta * distance ** 2)


cpdef g_pe_pair(double[:, :, :, :, :] out_tensor1,
                double[:, :, :, :, :] out_tensor2,
                double[:, :, :, :, :] out_tensor3,
                double[:, :, :, :, :] out_tensor4,
                long[:, :] numbers,
                double[:, :, :] positions,
                double[:, :, :, :] positions_pe,
                long[:] n_atoms,
                double cutoff,
                double eta,
                double lamda,
                double zeta,
                idx_dict):
    cdef int ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for j1 in range(n_atoms[ibatch]):
                for k1 in range(positions_pe.shape[1]):

                    dist_ij = (positions[ibatch, ii, 0] - positions_pe[ibatch, k1, j1, 0]) ** 2 +\
                              (positions[ibatch, ii, 1] - positions_pe[ibatch, k1, j1, 1]) ** 2 +\
                              (positions[ibatch, ii, 2] - positions_pe[ibatch, k1, j1, 2]) ** 2
                    dist_ij = dist_ij ** 0.5
                    f_ij = decay(dist_ij, cutoff)
                    fe_ij = f_ij * exp_dist(dist_ij, eta)

                    if cutoff > dist_ij > 0.0:
                        out_tensor1[ibatch, ii, jj, k1, 0] = f_ij

                        for k2 in range(positions_pe.shape[1]):
                            for j2 in range(n_atoms[ibatch]):

                                dist_ik = (positions[ibatch, ii, 0] - positions_pe[ibatch, k2, j2, 0]) ** 2 +\
                                          (positions[ibatch, ii, 1] - positions_pe[ibatch, k2, j2, 1]) ** 2 +\
                                          (positions[ibatch, ii, 2] - positions_pe[ibatch, k2, j2, 2]) ** 2
                                dist_ik = dist_ik ** 0.5
                                dist_jk = (positions_pe[ibatch, k1, j1, 0] - positions_pe[ibatch, k2, j2, 0]) ** 2 +\
                                          (positions_pe[ibatch, k1, j1, 1] - positions_pe[ibatch, k2, j2, 1]) ** 2 +\
                                          (positions_pe[ibatch, k1, j1, 2] - positions_pe[ibatch, k2, j2, 2]) ** 2
                                dist_jk = dist_jk ** 0.5
                                if cutoff > dist_ik > 0.0 and cutoff > dist_jk > 0.0:

                                    f_ik = decay(dist_ik, cutoff)
                                    f_jk = decay(dist_jk, cutoff)

                                    idx_k = idx_dict[numbers[ibatch, j2]]
                                    out_tensor2[ibatch, ii, j1, k1, idx_k] += f_ij * (f_ik + f_jk)

                                    fe_ik = f_ik * exp_dist(dist_ik, eta)
                                    fe_jk = f_jk * exp_dist(dist_jk, eta)
                                    out_tensor3[ibatch, ii, j1, k1, idx_k] += fe_ij * (fe_ik + fe_jk)

                                    pos_ik0 = positions[ibatch, ii, 0] - positions_pe[ibatch, k1, j1, 0]
                                    pos_jk0 = positions_pe[ibatch, k1, j1, 0] - positions_pe[ibatch, k2, j2, 0]
                                    pos_ik1 = positions[ibatch, ii, 1] - positions_pe[ibatch, k1, j1, 1]
                                    pos_jk1 = positions_pe[ibatch, k1, j1, 1] - positions_pe[ibatch, k2, j2, 1]
                                    pos_ik2 = positions[ibatch, ii, 2] - positions_pe[ibatch, k2, j2, 2]
                                    pos_jk2 = positions_pe[ibatch, k1, j1, 2] - positions_pe[ibatch, k2, j2, 2]
                                    cos_ijk = (pos_ik0 * pos_jk0 + pos_ik1 * pos_jk1 + pos_ik2 * pos_jk2) / (dist_ik * dist_jk)
                                    fe_jk = f_jk * exp_dist(dist_jk, eta)
                                    out_tensor4[ibatch, ii, j1, k1, idx_k] += \
                                        fe_ij * 2 ** (1 - zeta) * f_ik * f_jk * \
                                        (1 + lamda) * exp_dist(dist_ik ** 2 + dist_jk ** 2, eta)

    return out_tensor1, out_tensor2, out_tensor3, out_tensor4


cpdef g_pe(double[:, :, :] out_tensor1,
           double[:, :, :] out_tensor2,
           double[:, :, :] out_tensor4,
           long[:, :] numbers,
           double[:, :, :] positions,
           double[:, :, :, :] positions_pe,
           long[:] n_atoms,
           double cutoff,
           double eta,
           double lamda,
           double zeta,
           idx_dict):
    cdef int ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for j1 in range(n_atoms[ibatch]):
                for k1 in range(positions_pe.shape[1]):

                    dist_ij = (positions[ibatch, ii, 0] - positions_pe[ibatch, k1, j1, 0]) ** 2 +\
                              (positions[ibatch, ii, 1] - positions_pe[ibatch, k1, j1, 1]) ** 2 +\
                              (positions[ibatch, ii, 2] - positions_pe[ibatch, k1, j1, 2]) ** 2
                    dist_ij = dist_ij ** 0.5
                    f_ij = decay(dist_ij, cutoff)
                    fe_ij = f_ij * exp_dist(dist_ij, eta)

                    if cutoff > dist_ij > 0.0 and 120 > numbers[ibatch, j1] > 0:
                        idx_k = idx_dict[(numbers[ibatch, j1])]
                        out_tensor1[ibatch, ii, idx_k] += f_ij
                        out_tensor2[ibatch, ii, idx_k] += fe_ij

                        for k2 in range(positions_pe.shape[1]):
                            for j2 in range(n_atoms[ibatch]):

                                dist_ik = (positions[ibatch, ii, 0] - positions_pe[ibatch, k2, j2, 0]) ** 2 +\
                                          (positions[ibatch, ii, 1] - positions_pe[ibatch, k2, j2, 1]) ** 2 +\
                                          (positions[ibatch, ii, 2] - positions_pe[ibatch, k2, j2, 2]) ** 2
                                dist_ik = dist_ik ** 0.5
                                dist_jk = (positions_pe[ibatch, k1, j1, 0] - positions_pe[ibatch, k2, j2, 0]) ** 2 +\
                                          (positions_pe[ibatch, k1, j1, 1] - positions_pe[ibatch, k2, j2, 1]) ** 2 +\
                                          (positions_pe[ibatch, k1, j1, 2] - positions_pe[ibatch, k2, j2, 2]) ** 2
                                dist_jk = dist_jk ** 0.5
                                if cutoff > dist_ik > 0.0 and cutoff > dist_jk > 0.0 and 120 > numbers[ibatch, j2] > 0:

                                    f_ik = decay(dist_ik, cutoff)
                                    f_jk = decay(dist_jk, cutoff)

                                    idx_k = idx_dict[(numbers[ibatch, j1], numbers[ibatch, j2])]

                                    pos_ik0 = positions[ibatch, ii, 0] - positions_pe[ibatch, k1, j1, 0]
                                    pos_jk0 = positions_pe[ibatch, k1, j1, 0] - positions_pe[ibatch, k2, j2, 0]
                                    pos_ik1 = positions[ibatch, ii, 1] - positions_pe[ibatch, k1, j1, 1]
                                    pos_jk1 = positions_pe[ibatch, k1, j1, 1] - positions_pe[ibatch, k2, j2, 1]
                                    pos_ik2 = positions[ibatch, ii, 2] - positions_pe[ibatch, k2, j2, 2]
                                    pos_jk2 = positions_pe[ibatch, k1, j1, 2] - positions_pe[ibatch, k2, j2, 2]
                                    cos_ijk = (pos_ik0 * pos_jk0 + pos_ik1 * pos_jk1 + pos_ik2 * pos_jk2) / (dist_ik * dist_jk)
                                    fe_ijk = exp_dist(dist_ij ** 2 + dist_ik ** 2 + dist_jk ** 2, eta)

                                    out_tensor4[ibatch, ii, idx_k] += 2 ** (1 - zeta) * (1 + lamda * cos_ijk) ** zeta *\
                                        fe_ijk * f_ij * f_ik * f_jk


    return out_tensor1, out_tensor2, out_tensor4


#
# cpdef g3(double[:, :, :, :] out_tensor,
#          double[:, :, :] positions,
#          double[:, :, :] exp_dist,
#          long[:] n_atoms,
#          int len_mbtr,
#          double cutoff):
#     cdef int ibatch, ii, jj, kk, im
#     cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk
#
#     for ibatch in range(len(n_atoms)):
#         for ii in range(n_atoms[ibatch]):
#             for jj in range(n_atoms[ibatch]):
#
#                 dist_ij = sum((positions[ibatch, ii] - positions[ibatch, jj]) ** 2) ** 0.5
#                 fe_ij = decay(dist_ij, cutoff) * exp_dist[ibatch, ii, jj]
#
#                 if cutoff > dist_ij > 0.0:
#                     for kk in range(n_atoms[ibatch]):
#
#                         if cutoff > dist_ik > 0.0 and cutoff > dist_jk > 0.0:
#                             # dist_ik = distances[ibatch, ii, kk]
#                             # dist_jk = distances[ibatch, jj, kk]
#                             dist_ik = sum((positions[ibatch, ii] - positions[ibatch, kk]) ** 2) ** 0.5
#                             dist_jk = sum((positions[ibatch, jj] - positions[ibatch, kk]) ** 2) ** 0.5
#                             fe_ik = decay(dist_ik, cutoff) * exp_dist[ibatch, ii, kk]
#                             fe_jk = decay(dist_jk, cutoff) * exp_dist[ibatch, jj, kk]
#
#                             for im in range(len_mbtr):
#                                 out_tensor[ibatch, ii, jj, im] += (fe_ik+fe_jk)*fe_ij
#
#     return out_tensor
#
#
# cpdef g4(double[:, :, :, :] out_tensor,
#          double[:, :, :] distances,
#          double[:, :, :] distances_vect,
#          double[:, :, :] exp_dist,
#          long[:] n_atoms,
#          int len_mbtr,
#          double cutoff):
#     cdef int ibatch, ii, jj, kk, im
#     cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk
#
#     for ibatch in range(len(n_atoms)):
#         for ii in range(n_atoms[ibatch]):
#             for jj in range(n_atoms[ibatch]):
#
#                 dist_ij = distances[ibatch, ii, jj]
#                 fe_ij = decay(dist_ij, cutoff) * exp_dist[ibatch, ii, jj]
#
#                 for kk in range(n_atoms[ibatch]):
#                     dist_ik = distances[ibatch, ii, kk]
#                     dist_jk = distances[ibatch, jj, kk]
#                     decay_ik = decay(dist_ik, cutoff)
#                     decay_jk = decay(dist_jk, cutoff)
#
#                     for im in range(len_mbtr):
#                         out_tensor[ibatch, ii, jj, im] += decay_ij*decay_ik*decay_jk
#
#     return out_tensor
#
