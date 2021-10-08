"""Write Silicon systems to hdf5 binary file."""
import h5py
import torch
import numpy as np
from tbmalt.common.batch import pack
from ase.io.aims import read_aims
import os
torch.set_default_dtype(torch.float64)
torch.set_printoptions(20)
_number = {'Si': 14}


def silicon_data_writer(path_to_input, file_name):
    """Write database."""
    # Find the path of input file
    path = path_to_input
    inputfile = []
    for root, ds, fs in os.walk(path):
        for file in fs:
            if file == 'geometry.in':
                inputfile.append(os.path.join(root, file))

    # Read geometry file
    cells = []
    positions = []
    species = []

    for ip in inputfile:
        geo = read_aims(ip, apply_constraints=False)
        cells.append(torch.from_numpy(geo.cell[:]))
        positions.append(torch.from_numpy(geo.get_positions()))
        species.append(geo.get_chemical_symbols())

    numbers = [torch.tensor([_number[isp] for isp in ibatch]) for ibatch in species]

    # Check whether have same geometries
    aa = []
    for ii in range(len(positions)):
        for jj in range(len(positions)):
            if ii != jj:
                if (positions[ii] == positions[jj]).all() & (cells[ii] == cells[jj]).all():
                    _aa = torch.tensor([ii, jj])
                    aa.append(_aa)

    # write hdf5 file
    with h5py.File(file_name, "w") as f:
        for ii, isys in enumerate(species):
            if ''.join(isys) not in f.keys():
                g = f.create_group(''.join(isys))
                g.attrs['specie'] = isys
                g.attrs['numbers'] = numbers[ii]
                g.attrs['size_molecule'] = len(isys)
                g.attrs['n_molecule'] = 0
            else:
                g = f[''.join(isys)]

            n_system = g.attrs['n_molecule']
            g.attrs['n_molecule'] = n_system + 1
        g.create_dataset('cells', data=pack(cells))
        g.create_dataset('coordinates', data=pack(positions))
        dt = h5py.special_dtype(vlen=str)
        text = g.create_dataset('species', (64,), dtype=dt)
        text[:] = species[0]
