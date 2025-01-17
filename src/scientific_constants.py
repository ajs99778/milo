#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declare scientific constants used throughout Milo."""

# constants
h = 6.62607015E-34  # planck's constant in joule * second
c = 29979245800  # speed of light in cm / second
CLASSICAL_SPACING = 2  # cm^-1
GAS_CONSTANT_KCAL = 0.00198720425864083  # gas constant: kcal / (mol * kelvin)
AVOGADROS_NUMBER = 6.02214076E23  # number of particles in one mole

# conversions
# prefixes
FROM_KILO = 1.0E3
TO_KILO = 1 / FROM_KILO
TO_MILLI = 1.0E3
FROM_MILLI = 1 / TO_MILLI
TO_CENTI = 1.0E2
FROM_CENTI = 1 / TO_CENTI
# number of particles
MOLE_TO_PARTICLE = AVOGADROS_NUMBER
PARTICLE_TO_MOLE = 1 / AVOGADROS_NUMBER
# distance
ANGSTROM_TO_METER = 1.0E-10
METER_TO_ANGSTROM = 1 / ANGSTROM_TO_METER
BOHR_TO_ANGSTROM = 0.52917721090380
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
# mass
AMU_TO_KG = 1.66053878E-27
KG_TO_AMU = 1 / AMU_TO_KG
# force
HARTREE_PER_BOHR_TO_NEWTON = 8.2387234983E-8
NEWTON_TO_HARTREE_PER_BOHR = 1 / HARTREE_PER_BOHR_TO_NEWTON
NEWTON_TO_DYNE = 1.0E5
DYNE_TO_NEWTON = 1 / NEWTON_TO_DYNE
# time
SECOND_TO_FEMTOSECOND = 1.0E15
FEMTOSECOND_TO_SECOND = 1 / SECOND_TO_FEMTOSECOND
ATOMIC_TO_SECOND = 2.4188843265857e-17
SECOND_TO_ATOMIC = 1 / ATOMIC_TO_SECOND
# energy
CALORIE_TO_JOULE = 4.184
JOULE_TO_CALORIE = 1 / CALORIE_TO_JOULE
JOULE_TO_KCAL_PER_MOLE = TO_KILO * JOULE_TO_CALORIE / PARTICLE_TO_MOLE
KCAL_PER_MOLE_TO_JOULE = 1 / JOULE_TO_KCAL_PER_MOLE
JOULE_TO_MILLIDYNE_ANGSTROM = TO_MILLI * NEWTON_TO_DYNE * METER_TO_ANGSTROM
MILLIDYNE_ANGSTROM_TO_JOULE = 1 / JOULE_TO_MILLIDYNE_ANGSTROM
HARTREE_TO_JOULE = 4.359744722207185E-18
JOULE_TO_HARTREE = 1 / HARTREE_TO_JOULE
ELECTRON_VOLT_TO_JOULE = 1.602176634e-19
JOULE_TO_ELECTRON_VOLT = 1 / ELECTRON_VOLT_TO_JOULE
WAVENUMBER_TO_HARTREE = h * c * JOULE_TO_HARTREE
