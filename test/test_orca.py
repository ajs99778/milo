#!/usr/bin/env python

import os
import unittest

from AaronTools.fileIO import FileReader
from AaronTools.test import TestWithTimer

from milo import containers
from milo import enumerations as enums
from milo.molecule import Molecule
from milo.program_state import ProgramState
from milo.electronic_structure_program_handler import ORCASurfaceHopHandler
from milo.test import prefix

import numpy as np

WITH_UNIXMD = False
try:
    import unixmd
    WITH_UNIXMD = True
except (ImportError, ModuleNotFoundError):
    pass

class TestORCA(TestWithTimer):
    step_0_orca_out = os.path.join(
        prefix, "test_files", "orca", "_0_force.out"
    )
    step_1_orca_out = os.path.join(
        prefix, "test_files", "orca", "_1_force.out"
    )
    step_0_orca_cis = os.path.join(
        prefix, "test_files", "orca", "_0_force.cis"
    )
    step_1_orca_cis = os.path.join(
        prefix, "test_files", "orca", "_1_force.cis"
    )
    ref_ao_overlap = os.path.join(
        prefix, "ref_files", "orca", "ao_overlap.npy"
    )
    ref_nac = os.path.join(
        prefix, "ref_files", "orca", "nac.npy"
    )
    ref_soc = os.path.join(
        prefix, "ref_files", "orca", "soc.npy"
    )
    _ps = None
    _sh_ps = None

    @classmethod
    def setUpClass(cls):
        """set up ProgramState using two iteration from an MD simulation"""
        super(TestORCA, cls).setUpClass()

        if not WITH_UNIXMD:
            return

        # initial surface hopping setup
        ps = ProgramState()
        ps.step_size = containers.Time(0.5, enums.TimeUnits.FEMTOSECOND)
        ps.intersystem_crossing = True
        ps.set_number_of_electronic_states(5)
        ps.electronic_structure_handler = ORCASurfaceHopHandler(None)
        # pretend we've just finished the second iteration
        ps.current_step = 1
        fr0 = FileReader(cls.step_0_orca_out)
        fr1 = FileReader(cls.step_1_orca_out)

        # read structures - these would be calculated
        structure_0 = containers.Positions()
        structure_1 = containers.Positions()
        for atom_0, atom_1 in zip(fr0.atoms, fr1.atoms):
            structure_0.append(
                *atom_0.coords, enums.DistanceUnits.ANGSTROM
            )
            structure_1.append(
                *atom_1.coords, enums.DistanceUnits.ANGSTROM
            )
        ps.structures.append(structure_0)
        ps.structures.append(structure_1)
        ps.molecule = Molecule(fr1.atoms)

        # read other data
        # forces, energy, state energies, mo and ci coefficients, soc
        f, e, se, mo, ci, soc = ps.electronic_structure_handler._grab_data(cls.step_0_orca_out, ps)
        ps.previous_mo_coefficients = mo
        ps.previous_ci_coefficients = ci
        ps.socmes.append(soc)
        f, e, se, mo, ci, soc = ps.electronic_structure_handler._grab_data(cls.step_1_orca_out, ps)
        ps.current_mo_coefficients = mo
        ps.current_ci_coefficients = ci
        ps.socmes.append(soc)

        # compute AO overlap
        ps.electronic_structure_handler._compute_overlaps(ps)

        cls._sh_ps = ps

    @unittest.skipUnless(WITH_UNIXMD, "requires modified UNIX-MD")
    def test_ao_overlap(self):
        """AO overlap"""
        # AO overlap is calculated by Milo
        # check the ORCASurfaceHopHandler class methods if this fails
        ref = np.load(self.ref_ao_overlap)

        ao_diff = np.sqrt(np.sum((self._sh_ps.atomic_orbital_overlap - ref)  ** 2))
        self.assertTrue(ao_diff < 1e-8)

    @unittest.skipUnless(WITH_UNIXMD, "requires modified UNIX-MD")
    def test_nac(self):
        """NAC"""
        # this is calculated using UNIX-MD mode
        # check unixmd.qm.cioverlap in tdnac.c or cioverlap.pyx if this fails
        nac = self._sh_ps.electronic_structure_handler._compute_nacme(self._sh_ps)

        ref = np.load(self.ref_nac)
        nac_diff = np.sqrt(np.sum((nac - ref)  ** 2))
        self.assertTrue(nac_diff < 1e-8)

    @unittest.skipUnless(WITH_UNIXMD, "requires modified UNIX-MD")
    def test_soc(self):
        """SOC"""
        # this is read from the ORCA output file
        # check the parse if this fails
        soc = self._sh_ps.socmes[-1]

        ref = np.load(self.ref_soc)
        soc_diff = np.sqrt(np.sum((soc - ref)  ** 2))
        self.assertTrue(soc_diff < 1e-8)

if __name__ == "__main__":
    unittest.main()

