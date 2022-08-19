#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle calling ESPs and parsing output."""

import os
import subprocess
from sys import platform

from milo import containers
from milo import enumerations as enums
from milo import exceptions
from AaronTools.fileIO import FileReader


def get_program_handler(program_state):
    """Return the configured electronic structure program handler."""
    print(program_state.program_id)
    if program_state.program_id is enums.ProgramID.GAUSSIAN:
        return GaussianHandler(program_state.executable)
    elif program_state.program_id is enums.ProgramID.ORCA:
        return ORCAHandler(program_state.executable)
    else:
        raise ValueError(f'Unknown electronic structure program '
                         f'"{program_state.program_id}"')


class ProgramHandler:
    """base class for handling different QM software packages"""
    def __init__(self, executable):
        self.executable = executable

    def compute_forces(self, program_state):
        """
        program_state: milo.program_state.ProgramState instance
        returns True if forces were compute successfully
        """
        raise NotImplementedError


class GaussianHandler(ProgramHandler):
    """handler for Gaussian 16 and 09."""

    def generate_forces(self, program_state):
        """Preform computation and append forces to list in program state."""
        log_file = self._run_job(
            f"_{program_state.current_step}",
            program_state,
        )
        self._grab_forces(log_file, program_state)
        return True

    def _run_job(self, job_name, program_state, debug=False):
        """Call Gaussian and return a string with the name of the log file."""
        job_com_file = f"{job_name}.com"
        job_log_file = f"{job_name}.log"
        self._prepare_com_file(
            job_com_file,
            program_state.theory,
            program_state,
        )
        args = [self.executable, job_com_file, job_log_file]
        kwargs = dict()
        if platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        if debug:
            print("executing %s\n" % " ".join(args))
        proc = subprocess.Popen(
            args,
            **kwargs,
        )
        proc.communicate()
        if debug:
            print("done executing %s\n" % " ".join(args))
        return job_log_file

    @staticmethod
    def _prepare_com_file(file_name, route_section, program_state):
        """Prepare a .com file for a Gaussian run."""
        program_state.molecule.write(
            outfile=file_name,
            theory=program_state.theory,
            style="gaussian",
        )

    @staticmethod
    def _grab_forces(log_file_name, program_state):
        """Parse forces into program_state from the given log file."""
        fr = FileReader(log_file_name, just_geom=False)
        if not fr["finished"]:
            raise exceptions.ElectronicStructureProgramError(
                "Gaussian force calculation log file was not valid. Gaussian "
                "returned an error or could not be called correctly."
            )
        
        forces = containers.Forces()
        energy = containers.Energies()
        
        energy.append(fr["energy"], enums.EnergyUnits.HARTREE)
        program_state.energies.append(energy)
        for v in fr["forces"]:
            forces.append(*v, enums.ForceUnits.HARTREE_PER_BOHR)
        program_state.forces.append(forces)


class ORCAHandler(ProgramHandler):
    """handler for ORCA."""

    def generate_forces(self, program_state):
        """Preform computation and append forces to list in program state."""
        out_file = self._run_job(
            f"_{program_state.current_step}",
            program_state,
        )
        self._grab_forces(out_file, program_state)
        return True

    def _run_job(self, job_name, program_state, debug=False):
        """Call Gaussian and return a string with the name of the log file."""
        job_inp_file = f"{job_name}.inp"
        job_out_file = f"{job_name}.out"
        self._prepare_inp_file(
            job_inp_file,
            program_state.theory,
            program_state,
        )
        stdout = open(job_out_file, "w")
        args = [self.executable, job_inp_file]
        kwargs = {
            "stdout": stdout,
        }
        if platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        if debug:
            print("executing %s\n" % " ".join(args))
        proc = subprocess.Popen(
            args,
            **kwargs,
        )
        proc.communicate()
        if debug:
            print("done executing %s\n" % " ".join(args))
        stdout.close()
        return job_out_file

    @staticmethod
    def _prepare_inp_file(file_name, route_section, program_state):
        """Prepare a .inp file for an ORCA run."""
        program_state.molecule.write(
            outfile=file_name,
            theory=program_state.theory,
            style="orca",
        )

    @staticmethod
    def _grab_forces(out_file_name, program_state):
        """Parse forces into program_state from the given out file."""
        fr = FileReader(out_file_name, just_geom=False)
        if not fr["finished"]:
            raise exceptions.ElectronicStructureProgramError(
                "ORCA force calculation out file was not valid. ORCA "
                "returned an error or could not be called correctly."
            )
        
        forces = containers.Forces()
        energy = containers.Energies()
        
        energy.append(fr["energy"], enums.EnergyUnits.HARTREE)
        program_state.energies.append(energy)
        for v in fr["forces"]:
            forces.append(*v, enums.ForceUnits.HARTREE_PER_BOHR)
        program_state.forces.append(forces)
