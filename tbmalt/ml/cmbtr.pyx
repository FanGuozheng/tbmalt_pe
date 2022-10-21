import numpy as np


cpdef decay_ijk(double[:, :, :, :] mbtr_tensor,
                double[:, :, :, :] out_tensor,
                double[:, :, :] distances,
                long[:] n_atoms,
                int len_mbtr,
                double cutoff):
    cdef int ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for jj in range(n_atoms[ibatch]):
                dist_ij = distances[ibatch, ii, jj]
                decay_ij = decay(dist_ij, cutoff)
                for kk in range(n_atoms[ibatch]):
                    dist_ik = distances[ibatch, ii, kk]
                    dist_jk = distances[ibatch, jj, kk]
                    decay_ik = decay(dist_ik, cutoff)
                    decay_jk = decay(dist_jk, cutoff)
                    for im in range(len_mbtr):
                        out_tensor[ibatch, ii, jj, im] += mbtr_tensor[ibatch, ii, kk, im]*decay_ij*decay_ik*decay_jk
                        out_tensor[ibatch, ii, jj, im+len_mbtr] += mbtr_tensor[ibatch, jj, kk, im]*decay_ij*decay_ik*decay_jk
    return out_tensor


cpdef decay_jk(double[:, :, :, :] mbtr_tensor,
                double[:, :, :, :] out_tensor,
                double[:, :, :] distances,
                long[:] n_atoms,
                int len_mbtr,
                double cutoff):
    cdef int ibatch, ii, jj, kk, im
    cdef double dist_ij, dist_ik, dist_jk, decay_ij, decay_ik, decay_jk

    for ibatch in range(len(n_atoms)):
        for ii in range(n_atoms[ibatch]):
            for jj in range(n_atoms[ibatch]):
                dist_ij = distances[ibatch, ii, jj]
                decay_ij = decay(dist_ij, cutoff)
                for kk in range(n_atoms[ibatch]):
                    dist_ik = distances[ibatch, ii, kk]
                    dist_jk = distances[ibatch, jj, kk]
                    decay_ik = decay(dist_ik, cutoff)
                    decay_jk = decay(dist_jk, cutoff)
                    for im in range(len_mbtr):
                        out_tensor[ibatch, ii, jj, im] += mbtr_tensor[ibatch, ii, kk, im]*decay_ik*decay_jk
                        out_tensor[ibatch, ii, jj, im+len_mbtr] += mbtr_tensor[ibatch, jj, kk, im]*decay_ik*decay_jk
    return out_tensor


cpdef decay(double distance,                       cutoff):
    if distance > cutoff:
        return 0.0
    else:
        return 0.5 * np.cos(np.pi * distance / cutoff) + 0.5
