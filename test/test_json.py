#!/usr/bin/env python

import json
import os
import unittest

from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer

from milo import containers
from milo import enumerations as enums
from milo.json_extension import MiloEncoder, MiloDecoder
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

class TestJSON(TestWithTimer):
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
        super(TestJSON, cls).setUpClass()

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
        ps.molecule = Geometry(fr1.atoms)

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

    def test_json(self, debug=False):
        """ProgramState JSON"""
        s = json.dumps(self._sh_ps.__dict__, indent=2, cls=MiloEncoder)
        ps = ProgramState()
        d = json.loads(s, cls=MiloDecoder)
        for key, value in d.items():
            setattr(ps, key, value)

        for key in ps.__dict__.keys():
            if key == "electronic_structure_handler":
                continue
            if key == "program_id":
                continue
            if debug:
                print(key)
            try:
                a = getattr(self._sh_ps, key)
                b = getattr(ps, key)
                if debug:
                    print("a")
                    print(a)
                    print("b")
                    print(b)
                self.assertEqual(a, b)
            except ValueError:
                pass

if __name__ == "__main__":
    unittest.main()

