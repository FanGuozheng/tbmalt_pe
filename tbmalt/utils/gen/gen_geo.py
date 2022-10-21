import os
from typing import Union, List
from ase import Atoms
import ase.io as io
import numpy as np

from tbmalt import Geometry
from tbmalt.io.dataset import GeometryTo


class GenGeo:

    def __init__(self):
        pass


    @classmethod
    def band(cls, in_geometries: Union[str, List[str]],
        template: Union[List[str], str] = None,

    ):
        path_dftb = (
            "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery"
        )
        geometry_in_files = [os.path.join(in_geometries, ii) for ii in os.listdir(in_geometries)]
        path_input_template = path_input_template
        to_geometry_type = to_geometry_type
        to_geometry_path = to_geometry_path

        geot = GeometryTo(
            in_geometry_files=geometry_in_files,
            path_to_input_template=path_input_template,
            to_geometry_type=to_geometry_type,
            to_geometry_path=to_geometry_path,
            calculation_properties=["band"],
            labels=labels,
            set_constraint=set_constraint,
        )
        geot()

    @classmethod
    def scale(
        in_geometries: Union[str, List[str]],
        to_path: Union[str, list] = './',
        to_type: str = 'dftb',
        template: Union[List[str], str] = None,
    ):
        """Generate volume scaling geometries with template."""
        geo = io.read(template)
        _in = [
            Atoms(
                positions=geo.positions * ii,
                numbers=geo.numbers,
                cell=geo.cell * ii,
                pbc=True,
            )
            for ii in in_geometries
        ]

        if isinstance(to_path, str):
            to_path = [to_path] * len(in_geometries)

        if to_type == "aims":
            print("rm " + to_path[0] + "/geometry.in.*")
            os.system("rm " + to_path[0] + "/geometry.in.*")
            [
                io.write(os.path.join(path, "geometry.in." + str(ii)), iin, format="aims")
                for ii, (iin, path) in enumerate(zip(_in, to_path))
            ]
        elif to_type == "dftb":
            os.system("rm " + to_path[0] + "/geo.gen.*")
            [
                io.write(os.path.join(path, "geo.gen." + str(ii)), iin, format="dftb")
                for ii, (iin, path) in enumerate(zip(_in, to_path))
            ]
        elif to_type == "vasp":
            [
                io.write(os.path.join(path, "POSCAR." + str(ii)), iin, format="vasp")
                for ii, (iin, path) in enumerate(zip(_in, to_path))
            ]

        return _in

    @staticmethod
    def dftbplus_geo(obj, file):
        pass

    @staticmethod
    def surface(ase_obj, indices, layers, vacuum=None, tol=1e-10, periodic=False):
        """Create surface from a given lattice and Miller indices.

        lattice: Atoms object or str
            Bulk lattice structure of alloy or pure metal.  Note that the
            unit-cell must be the conventional cell - not the primitive cell.
            One can also give the chemical symbol as a string, in which case the
            correct bulk lattice will be generated automatically.
        indices: sequence of three int
            Surface normal in Miller indices (h,k,l).
        layers: int
            Number of equivalent layers of the slab.
        vacuum: float
            Amount of vacuum added on both sides of the slab.
        periodic: bool
            Whether the surface is periodic in the normal to the surface
        """

        indices = np.asarray(indices)

        if indices.shape != (3,) or not indices.any() or indices.dtype != int:
            raise ValueError('%s is an invalid surface type' % indices)

        if isinstance(ase_obj, str):
            lattice = bulk(ase_obj, cubic=True)

        h, k, l = indices  # noqa (E741, the variable l)
        h0, k0, l0 = (indices == 0)

        if h0 and k0 or h0 and l0 or k0 and l0:  # if two indices are zero
            if not h0:
                c1, c2, c3 = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
            if not k0:
                c1, c2, c3 = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
            if not l0:
                c1, c2, c3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            p, q = ext_gcd(k, l)
            a1, a2, a3 = ase_obj.cell

            # constants describing the dot product of basis c1 and c2:
            # dot(c1,c2) = k1+i*k2, i in Z
            k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                        l * a2 - k * a3)
            k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                        l * a2 - k * a3)

            if abs(k2) > tol:
                i = -int(round(k1 / k2))  # i corresponding to the optimal basis
                p, q = p + i * l, q - i * k

            a, b = ext_gcd(p * k + q * l, h)

            c1 = (p * k + q * l, -p * h, -q * h)
            c2 = np.array((0, l, -k)) // abs(gcd(l, k))
            c3 = (b, a * p, a * q)

        surf = build(ase_obj, np.array([c1, c2, c3]), layers, tol, periodic)
        if vacuum is not None:
            surf.center(vacuum=vacuum, axis=2)
        return surf



    def build(lattice, basis, layers, tol, periodic):
        surf = lattice.copy()
        scaled = solve(basis.T, surf.get_scaled_positions().T).T
        scaled -= np.floor(scaled + tol)
        surf.set_scaled_positions(scaled)
        surf.set_cell(np.dot(basis, surf.cell), scale_atoms=True)
        surf *= (1, 1, layers)

        a1, a2, a3 = surf.cell
        surf.set_cell([a1, a2,
                       np.cross(a1, a2) * np.dot(a3, np.cross(a1, a2)) /
                       norm(np.cross(a1, a2))**2])

        # Change unit cell to have the x-axis parallel with a surface vector
        # and z perpendicular to the surface:
        a1, a2, a3 = surf.cell
        surf.set_cell([(norm(a1), 0, 0),
                       (np.dot(a1, a2) / norm(a1),
                        np.sqrt(norm(a2)**2 - (np.dot(a1, a2) / norm(a1))**2), 0),
                       (0, 0, norm(a3))],
                      scale_atoms=True)

        surf.pbc = (True, True, periodic)

        # Move atoms into the unit cell:
        scaled = surf.get_scaled_positions()
        scaled[:, :2] %= 1
        surf.set_scaled_positions(scaled)

        if not periodic:
            surf.cell[2] = 0.0

        return surf


def ext_gcd(a, b):
    if b == 0:
        return 1, 0
    elif a % b == 0:
        return 0, 1
    else:
        x, y = ext_gcd(b, a % b)
        return y, x - y * (a // b)

