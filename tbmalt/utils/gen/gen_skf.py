"""Write skf to hdf5 binary file.

The skf include normal skf files or skf with a list of compression radii.
"""
import h5py
import torch
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import LoadSKF


class WriteSK:
    """Transfer SKF files from skf files to hdf binary type.

    Arguments:
        path: Path to SKF files.
        element: The global elements to read and write.

    Keyword Args:
        sk_type: The SKF types.
        output_name: Name of new h5py type file.
    """

    def __init__(self, path: str, element: list, mode='w', **kwargs):
        """Read skf, smooth the tail, write to hdf."""
        self.path = path
        self.element = element
        self.mode = mode
        self.element_number = System.to_element_number(self.element).squeeze()
        self.sk_type = kwargs.get('sk_type', 'normal')
        self.output_name = kwargs.get('output_name', 'skf.hdf')
        self._write(**kwargs)

    def _write(self, **kwargs):
        """Read and Write WriteSK."""
        if self.sk_type == 'normal':
            self._write_sk_normal(**kwargs)
        elif self.sk_type == 'compression_radii':
            self._write_sk_compression_radii(**kwargs)

    def _write_sk_normal(self, **kwargs):
        """Read skf data from SKF files and write into h5py binary data."""
        repulsive = kwargs.get('repulsive', False)
        with h5py.File(self.output_name, self.mode) as f:

            # loop over all global elements
            for iele, iele_n in zip(self.element, self.element_number):
                for jele, jele_n in zip(self.element, self.element_number):
                    homo = kwargs.get('homo', iele == jele)

                    # read SKF files
                    sk = LoadSKF.read(self.path, [iele, jele],
                                      torch.tensor([iele_n, jele_n]), **kwargs)

                    # create and write into h5py type
                    g = f.create_group(iele + jele)
                    g.attrs['version'] = sk.version
                    g.create_dataset('hamiltonian', data=sk.hamiltonian)
                    g.create_dataset('overlap', data=sk.overlap)
                    g.create_dataset('hs_grid', data=sk.hs_grid)
                    g.create_dataset('hs_cutoff', data=sk.hs_cutoff)
                    # g.create_dataset('version', data=sk.version)
                    g.create_dataset('rep_poly', data=sk.rep_poly)
                    g.create_dataset('g_step', data=sk.g_step)
                    g.create_dataset('n_points', data=sk.n_points)

                    # write onsite if homo
                    if homo:
                        g.create_dataset('onsite', data=sk.onsite)
                        g.create_dataset('U', data=sk.U)

                    if repulsive:
                        g.create_dataset('n_repulsive', data=sk.n_repulsive)
                        g.create_dataset('rep_table', data=sk.rep_table)
                        g.create_dataset('rep_grid', data=sk.rep_grid)
                        g.create_dataset('rep_short', data=sk.rep_short)
                        g.create_dataset('rep_long_grid', data=sk.rep_long_grid)
                        g.create_dataset('rep_long_c', data=sk.rep_long_c)
                        g.create_dataset('rep_cutoff', data=sk.rep_cutoff)
                        g.create_dataset('repulsive', data=True)

    def _write_sk_compression_radii(self, **kwargs):
        """Read .skf data from skgen with various compR."""
        repulsive = kwargs.get('repulsive', False)
        with h5py.File(self.output_name, self.mode) as f:

            # loop over all global elements
            for iele, iele_n in zip(self.element, self.element_number):
                for jele, jele_n in zip(self.element, self.element_number):
                    homo = kwargs.get('homo', iele == jele)

                    # read SKF files
                    sk = LoadSKF.read(self.path, [iele, jele],
                                      torch.tensor([iele_n, jele_n]), **kwargs)

                    # create and write into h5py type
                    g = f.create_group(iele + jele)
                    g.attrs['version'] = sk.version
                    g.create_dataset('hamiltonian', data=sk.hamiltonian)
                    g.create_dataset('overlap', data=sk.overlap)
                    g.create_dataset('hs_grid', data=sk.hs_grid)
                    g.create_dataset('hs_cutoff', data=sk.hs_cutoff)
                    g.create_dataset('rep_poly', data=sk.rep_poly)
                    g.create_dataset('g_step', data=sk.g_step)
                    g.create_dataset('n_points', data=sk.n_points)

                    # write onsite if homo
                    if homo:
                        g.create_dataset('onsite', data=sk.onsite)
                        g.create_dataset('U', data=sk.U)

                    if repulsive:  # TODOOOOO
                        pass
