#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store data used throughout the simulation."""

from milo import containers
from milo import enumerations as enums
from milo import random_number_generator as rng

import numpy as np


class ProgramState:
    """Contain all the data used throughout the simulation."""

    def __init__(self):
        """Create all variables and sets some to default values."""
        # Basic job data
        self.job_name = None
        self.spin = None
        self.charge = None
        self.current_step = 0  # current_step is 0 for the first step
        self.step_size = containers.Time(1.00, enums.TimeUnits.FEMTOSECOND)
        self.max_steps = None  # If None, no limit
        self.temperature = 298.15  # kelvin
        self.input_structure = containers.Positions()

        # Enums for job control
        self.propagation_algorithm = enums.PropagationAlgorithm.VERLET
        self.oscillator_type = enums.OscillatorType.QUASICLASSICAL
        self.add_rotational_energy = enums.RotationalEnergy.NO
        self.geometry_displacement_type = enums.GeometryDisplacement.NONE
        self.phase_direction = enums.PhaseDirection.RANDOM
        self.phase = None  # (1, 2)
        self.fixed_mode_directions = dict()  # maps mode number to 1 or -1
        self.fixed_vibrational_quanta = dict()  # maps mode number to quanta

        # Data added to at every time step
        self.structures = list()
        self.velocities = list()
        self.forces = list()
        self.accelerations = list()
        self.energies = list()
        self.state_energies = list()

        # Frequency data
        self.frequencies = containers.Frequencies()
        self.mode_displacements = list()  # mode[frequency index][atom index]
        self.force_constants = containers.ForceConstants()
        self.reduced_masses = containers.Masses()
        self.zero_point_energy = None
        self.zero_point_correction = None

        # Energy Boost
        self.energy_boost = enums.EnergyBoost.OFF
        self.energy_boost_min = None
        self.energy_boost_max = None

        # Random Number Generation
        # self.random_seed = 0  # this is in self.random as self.random.seed
        self.random = rng.RandomNumberGenerator()

        # level of theory
        self.theory = None

        self.program_id = enums.ProgramID.GAUSSIAN

        # Program information
        self.processor_count = None
        self.memory_amount = None

        # Final output files
        self.output_xyz_file = True
        
        # surface hopping data
        self.nacmes = []
        self.socmes = []
        self.current_electronic_state = 0
        self.initial_electronic_state = 0 # ground state
        self.number_of_electronic_states = 1
        self.intersystem_crossing = False
        self.number_of_basis_functions = None
        self.number_of_alpha_occupied = None
        self.number_of_alpha_virtual = None
        self.current_mo_coefficients = None
        self.previous_mo_coefficients = None
        self.current_ci_coefficients = None
        self.previous_ci_coefficients = None
        self.electronic_propogation_steps = 20
        self.electronic_propogation_type = "coefficient"
        self.decoherence_correction = None
        self.ecd_parameter = containers.Energies()
        self.ecd_parameter.append(0.1, enums.EnergyUnits.HARTREE)
        self.rho = np.zeros(
            (
                self.number_of_electronic_states,
                self.number_of_electronic_states,
            ),
            dtype=np.cdouble
        )
        self.state_coefficients = np.zeros(
            self.number_of_electronic_states, dtype=np.cdouble
        )
        # XXX: what did these do? and why are they arrays?
        # AFIAK, they are just used as the bounds on loops in C code,
        # but there should be enough data passed in the other variables
        # to those C subroutines to infer values for orb_ini and orb_final
        self.orb_ini = np.zeros(1, dtype=np.int32)
        self.orb_final = np.zeros(1, dtype=np.int32)
        
        self.molecule = None
        

    @property
    def number_atoms(self):
        return self.molecule.num_atoms
    
    @property
    def atoms(self):
        return self.molecule.atoms

    def set_number_of_electronic_states(self, nstates):
        """
        sets the number of electronic states to nstates and adjusts the
        size of the state coefficient arrays
        """
        if nstates < self.number_of_electronic_states:
            self.rho = self.rho[nstates:, nstates:]
            self.state_coefficients = self.state_coefficients[nstates:]
        elif nstates > self.number_of_electronic_states:
            for i in range(0, nstates - self.number_of_electronic_states):
                self.rho = np.append(self.rho, np.zeros((1, self.number_of_electronic_states + i), dtype=np.cdouble), axis=0)
                self.rho = np.append(self.rho, np.zeros((self.number_of_electronic_states + i + 1, 1), dtype=np.cdouble), axis=1)
            state_coefficients = np.zeros(
                nstates, dtype=np.cdouble
            )
            state_coefficients[self.number_of_electronic_states:] = self.state_coefficients
            self.state_coefficients = state_coefficients
        self.number_of_electronic_states = nstates
