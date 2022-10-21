#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DFTB based molecular dynamics simulations."""
from typing import Union
import torch
import numpy as np
from tbmalt.physics.force import DftbGradient
from tbmalt.physics.dftb.dftb import Dftb1, Dftb2
from tbmalt import Basis, Geometry
from tbmalt.data.units import mass_units, time_units, _Boltzmann
from tbmalt.common.batch import pack
from torch import Tensor


class Md:
    """Molecular dynamics simulations.

    Arguments:
        geometry: Geometry object in TBMaLT.
        path_to_skf: Path to Slater-Koster files.
        shell_dict: Shell angular momentum dictionary.
        dftb: Dftb object in TBMaLT.
        descriptor: Descriptor for machine learning.
        ml_model: Machine learning model type.
        q0: Initial charges.
        skf_type: Slater-Koster files type.

    """

    def __init__(self,
                 geometry: Geometry,
                 path_to_skf: str,
                 shell_dict: dict,
                 dftb_type: float = 'dftb2',
                 descriptor: object = None,
                 ml_model: object = None,
                 q0: Tensor = None,
                 skf_type: str = 'h5', **kwargs):
        self.geometry = geometry
        self.path_to_skf = path_to_skf
        self.shell_dict = shell_dict
        self.descriptor = descriptor
        self.ml_model = ml_model
        self.basis = Basis(self.geometry.atomic_numbers, self.shell_dict)
        self.q0 = q0
        self.skf_type = skf_type

        # mask of non-padding positions
        self.mask = self.geometry.atomic_numbers.ne(0.0)
        self.a = torch.zeros(*self.geometry.atomic_numbers.shape, 3)
        self.temperature = kwargs.get('temperature', 273.15)
        self.n_freedom = (self.geometry.n_atoms - 1) * 3.0
        init_velocity = kwargs.get('init_velocity', None)

        # Initialize DFTB calculator
        self.dftb_type = dftb_type
        if dftb_type == 'dftb2':
            self.dftb = Dftb2(
                self.geometry, self.shell_dict, self.path_to_skf,
                repulsive=True, skf_type=self.skf_type)
        elif dftb_type == 'dftb3':
            self.dftb = Dftb3(
                self.geometry, self.shell_dict, self.path_to_skf,
                repulsive=True, skf_type=self.skf_type,
                u_derivative=kwargs.get('u_derivative'),
                damp_exp=kwargs.get('damp_exp'),
            )

        self.mass = self.dftb.skparams.mass * mass_units['atomic']

        if init_velocity is None:
            self.velocity = self._init_velocity(**kwargs)
        else:
            self.velocity = init_velocity

    def __call__(self, steps: int = 2, deltat: float = 0.1, **kwargs):
        """Run molecular dynamics.

        Arguments:
            steps: Molecular dynamics steps.

        Keyword Args:
            seed: Random seed for initial velocity.

        """
        self._algorithm = kwargs.get('md_driver', 'verlet')
        self.extra_charge = kwargs.get('extra_charge', 0.0)
        self.deltat = deltat * time_units[kwargs.get('time_unit', 'femtoseconds')]
        self._md_energy = []
        self._dftb_energy = []
        self._md_charge = []
        self._md_positions = []

        # molecular dynamics steps
        for istep in range(steps):

            # get ML charge
            if self.descriptor is not None and self.ml_model is not None:
                self.descriptor(self.geometry)
                x_pred = self.descriptor.g
                charge = self.ml_model(x_pred, self.geometry.n_atoms) + self.q0
            else:
                charge = None
            # Run DFTB calculations
            self._dftb(charge, **kwargs)

            # Get the deraviative from DFTB
            self.grad = self.get_gradient()

            # molecular dynamics calculations
            self.next_md(istep)
            self._dftb_energy.append(self.dftb.total_energy)

    def _init_velocity(self, **kwargs):
        """Get initial velocity."""
        torch.manual_seed(kwargs.get('seed', 0))
        _velocity = torch.rand(*self.geometry.atomic_numbers.shape, 4)
        velocity = torch.zeros(*self.geometry.atomic_numbers.shape, 3)
        kt = self.temperature * _Boltzmann

        # Box–Muller transformation
        vx, vy = boxmuller(
            _velocity[..., 0][self.mask], _velocity[..., 1][self.mask])
        vz, _ = boxmuller(
            _velocity[..., 2][self.mask], _velocity[..., 3][self.mask])
        # vx[vx.lt(0.5)] = -vx[vx.lt(0.5)]
        # vy[vy.lt(0.5)] = -vy[vy.lt(0.5)]
        # vz[vz.lt(0.5)] = -vz[vz.lt(0.5)]

        velocity[self.mask] = torch.stack([vx, vy, vz]).T
        velocity[self.mask] = velocity[self.mask] * torch.sqrt(
            kt / self.mass[self.mask]).unsqueeze(1)

        # to make the total verlocity as 0, mv shape: [3, n_batch]
        mv = (velocity.permute(2, 0, 1) * self.mass).sum(-1)
        mv = mv / self.mass.sum(-1)
        velocity = velocity.permute(1, 2, 0) - mv

        # to make sure velocity satisfy initial temperature
        # the input shape of velocity (after permute): [3, n_batch, max_atom]
        e_k = self.kinetic_energy(self.mass, velocity.permute(1, 2, 0))
        _kt = 2.0 * e_k / self.n_freedom
        velocity = velocity * torch.sqrt(kt / _kt)

        # return velocity shape: [n_batch, max_atom, 3]
        return velocity.permute(2, 0, 1)

    def _dftb(self, charge, **kwargs):
        """Run DFTB calculation."""
        self._charge_potential()
        self.dftb(charge, self.geometry)
        self._density = self.dftb.density
        self.density_e = self.dftb.energy_weighted_density
        self.eigenvalue = self.dftb.eigenvalue
        self.deltaq = self.dftb.deltaq

    def _charge_potential(self):
        self.qzero_norm = self.dftb.qzero.T * (
            self.extra_charge / self.dftb.qzero.sum(-1))
        self.deltaq = self.qzero_norm.T - self.dftb.qzero

    def get_gradient(self):
        """Calculate gradient from DFTB calculations."""
        self.grad_instance = DftbGradient(
            self.geometry, self.basis, self.dftb.h_feed, self.dftb.s_feed,
                self.shell_dict, self.dftb.skparams, dftb_type=self.dftb_type)

        kwargs = {}
        if self.dftb_type == 'dftb3':
            kwargs.update({
                'h': self.dftb.h,
                'damp_exp': self.dftb.damp_exp,
                's': self.dftb.short_gamma,
                's_u': -self.dftb.shortgamma_u,
                'h_u': self.dftb.h_u,
                'u_derivative': self.dftb.u_derivative,
            })

        _grad = self.grad_instance(
            self._density, self.density_e, self.eigenvalue, self.deltaq,
            self.dftb.shift_mat, self.dftb.U, **kwargs)

        return _grad

    def next_md(self, step):
        """"""
        self.a[self.mask] = -(self.grad[self.mask].T / self.mass[self.mask]).T

        self.geometry, new_v, self.velocity = _md_driver[self._algorithm](
            step, self.geometry.atomic_numbers,
            self.geometry.positions, self.velocity, self.a, self.deltat)

        # the input shape of velocity (after permute): [3, n_batch, max_atom]
        e_k = self.kinetic_energy(self.mass, new_v.permute(2, 0, 1))
        self.temperature = 2.0 * e_k / self.n_freedom / _Boltzmann
        self._md_energy.append(e_k.detach())
        self._md_charge.append(self.dftb.charge)
        self._md_positions.append(self.geometry.positions)

    def kinetic_energy(self, mass, velocity=None):
        v = velocity if velocity is not None else self.velocity
        return 0.5 * (v ** 2.0 * mass).sum(-1).sum(0)

    # @property
    # def H0_grad(self):
    #     return self.grad_instance.H0_grad()

    # @property
    # def coulomb_grad(self):
    #     return self.grad_instance.coulomb_grad()

    # @property
    # def repulsive_grad(self):
    #     return self.grad_instance.repulsive_grad()

    @property
    def density(self):
        return self.dftb.density

    @property
    def energy_weighted_density(self):
        return self.dftb.energy_weighted_density

    @property
    def charge(self):
        """Charge for all MD steps."""
        return pack(self._md_charge)

    @property
    def dftb_energy(self):
        """DFTB energy (without MD energy) for all MD steps."""
        return pack(self._dftb_energy)

    @property
    def md_energy(self):
        """MD kinetic energy for all MD steps."""
        return pack(self._md_energy)

    @property
    def total_energy(self):
        """Total energy for all MD steps."""
        return self.dftb_energy + self.md_energy


def boxmuller(u1, u2):
    """Box–Muller transform."""
    assert (u1.ge(0.0) * u1.le(1.0)).all(), 'u1 should be range from 0 to 1'
    assert (u2.ge(0.0) * u2.le(1.0)).all(), 'u1 should be range from 0 to 1'

    tmp1 = torch.sqrt(-2.0 * torch.log(u1))
    tmp2 = 2.0 * np.pi * u2

    return tmp1 * torch.cos(tmp2), tmp1 * torch.sin(tmp2)


def verlet(step, atomic_numbers, positions, v, a, deltat):
    """"""
    if step > 0:
        new_v = v + 0.5 * a * deltat
    else:
        new_v = v

    v = new_v + 0.5 * a * deltat
    positions = positions + v * deltat

    return Geometry(atomic_numbers, positions), new_v, v


def leap_frog(step, atomic_numbers, positions, v, a, deltat):
    """"""
    if step > 0:
        new_v = v + 0.5 * a * deltat
    else:
        new_v = v

    v = new_v + 0.5 * a * deltat
    positions = positions + v * deltat

    return Geometry(atomic_numbers, positions), new_v, v


_md_driver = {'verlet': verlet, 'leap_frog': leap_frog}
