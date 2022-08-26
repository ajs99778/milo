#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle calling ESPs and parsing output."""

import os
import subprocess
from sys import platform

from milo import containers
from milo import enumerations as enums
from milo import exceptions
from milo.scientific_constants import ELECTRON_VOLT_TO_JOULE, JOULE_TO_HARTREE
from milo.molecule import Molecule

from AaronTools.fileIO import FileReader
from AaronTools.theory import ForceJob, TDDFTJob

from unixmd.qm.ci_overlap import wf_overlap
from unixmd.mqc.el_propagator import el_run

import numpy as np

def get_program_handler(program_state, nonadiabatic=False):
    """Return the configured electronic structure program handler."""
    print(program_state.program_id)
    if program_state.program_id is enums.ProgramID.GAUSSIAN:
        if nonadiabatic:
            return GaussianSurfaceHopHandler(program_state.executable)
        else:
            return GaussianHandler(program_state.executable)
    elif program_state.program_id is enums.ProgramID.ORCA:
        if not nonadiabatic:
            return ORCAHandler(program_state.executable)
        raise ValueError("nonadiabatic dynamics not currently supported with ORCA")
    
    raise ValueError(f'Unknown electronic structure program '
                     f'"{program_state.program_id}"')


def _faulhaber_inverse(n):
    """returns the number of positive integers that add up to n starting from 1"""
    out = 0
    total = 0
    while total < n:
        out += 1
        total += out

    if total != n:
        raise ValueError("could not determine Faulhaber sum for %i" % n)

    return out


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



class GaussianSurfaceHopHandler(ProgramHandler):
    """handler for Gaussian 16 and 09 with Unix-MD surface hopping algorithms"""
    def __init__(self, executable):
        self.executable = executable
        gaussian_root, gaussian_exe = os.path.split(executable)
        gaussian_name, extension = os.path.splitext(gaussian_exe)
        self.rwfdump = os.path.join(gaussian_root, "rwfdump" + extension)

    def generate_forces(self, program_state):
        """
        Preform computation and append forces to list in program state.
        also determines if the state should change using Unix-MD
        """
        # fix up the theory to read SCF orbitals from
        # the previous iteration
        # this might help with SCF convergence issues
        force_theory = program_state.theory.copy()
        force_theory.add_kwargs(
            link0={
                "chk": ["force_job.chk"],
                "rwf": ["single.rwf"],
            },
            route={"NoSymm": []},
        )
        if program_state.current_step > 0:
            force_theory.add_kwargs(
                route={"guess": ["read"]},
            )
        force_theory.job_type = [
            ForceJob(), 
            TDDFTJob(
                program_state.number_of_electronic_states - 1,
                root_of_interest=program_state.current_electronic_state,
            ),
        ]

        # run force and excited state computation
        force_log_file = self._run_job(
            f"_{program_state.current_step}_force",
            program_state.molecule,
            force_theory,
        )
        self._grab_data(force_log_file, program_state)
        
        self._propogate_electronic(program_state)
        
        # surface hopping calculation
        # calculate overlap with the wavefunction from the previous
        # iteration
        # TODO: decoherence 
        self._compute_overlaps(program_state)
        program_state.nacmes.append(self._compute_nacme(program_state))
        self._propogate_electronic(program_state)
        hop = self._check_hop(program_state)
        if hop:
            self._scale_velocities(program_state)

        return True

    def _propogate_electronic(self, program_state):
        """
        propogate electrons
        """
        el_run(
            program_state.number_of_electronic_states,
            program_state.electronic_propogation_steps,
            program_state.step_size.as_atomic(),
            program_state.state_energies[-1],
            program_state.state_energies[-2],
            program_state.nacmes[-1],
            program_state.nacmes[-2],
            program_state.state_coefficient,
            program_state.rho,
            "density",
        )

    def _scale_velocities(self, program_state):
        """scale velocities to conserve energy after hopping"""
        # TODO
        pass

    def _check_hop(self, program_state):
        """
        determine if the simulation should hop surfaces
        """

        # calculate probability of hopping to a state
        state_probability = np.zeros(program_state.number_of_electronic_states)
        cummulative_probability = np.zeros(program_state.number_of_electronic_states + 1)
        running_state = program_state.current_electronic_state
        
        accumulator = 0.
        for i_state in range(0, program_state.number_of_electronic_states):
            if i_state == running_state:
                continue
            state_probability[i_state] = -2 * (
                program_state.rho.real[i_state, running_state] * \
                program_state.nacmes[-1][i_state, running_state] * \
                program_state.step_size.as_atomic() / program_state.rho.real[running_state, running_state]
            )
            accumulator += state_probability[i_state]
            cummulative_probability[i_state + 1] = accumulator
        total_probability = cummulative_probability[program_state.number_of_electronic_states]
        
        if total_probability > 1:
            state_probability /= total_probability
            cummulative_probability /= total_probability

        random_number = program_state.random.uniform()
        for i_state in range(0, program_state.number_of_electronic_states):
            if i_state == running_state:
                continue
            if random_number > cummulative_probability[i_state] and random_number <= cummulative_probability[i_state + 1]:
                program_state.current_electronic_state = i_state
                return True
        
        return False
    
    def _compute_overlaps(self, program_state):
        """
        compute atomic orbital overlap and overlap between wavefunctions
        from this iteration and the previous one
        """
        overlap_theory = program_state.theory.copy()
        # XXX: do we need to account for center of mass motion/rotation?
        # kill job after link 302 (overlap integral + some others)
        # and ignore that atoms are very close together
        overlap_theory.add_kwargs(
            link0={"rwf": ["overlap.rwf"], "kjob": ["l302"]},
            route={"NoSymm": [], "IOp": ["2/12=3"]},
        )
        overlap_theory.job_type = "energy"
        # double the molecules, double the charge
        overlap_theory.charge *= 2
        # make a combined structure with the structure from this iteration and
        # the structure from the previous iteration
        prev_mol = program_state.molecule.copy()
        prev_mol.coords = program_state.structures[-1].as_angstrom()
        overlap_mol = Molecule(program_state.molecule.atoms, prev_mol.atoms)
        overlap_rwf = self._run_job(
            f"_{program_state.current_step}_overlap",
            overlap_mol,
            overlap_theory,
        )
        program_state.mo_coefficients.append(
            self._read_mo_coeff("single.rwf")
        )
        ao_overlap = self._read_ao_overlap("overlap.rwf")
        # we only care about the overlap between the current AOs with the previous AOs
        dim = len(ao_overlap)
        program_state.ao_overlap.append(
            ao_overlap[:dim, dim:]
        )
        program_state.ci_coefficients.append(
            self._read_ci_coeff(
                "single.rwf",
                program_state.number_of_electronic_states,
                program_state.number_of_alpha_occupied,
                program_state.number_of_alpha_virtual,
            )
        )

    def _compute_nacme(self, program_state):
        nacme = wf_overlap(
            program_state.number_of_electronic_states,
            program_state.number_of_basis_functions,
            program_state.number_of_alpha_occupied,
            program_state.number_of_alpha_virtual,
            program_state.mo_coefficients[-2],
            program_state.mo_coefficients[-1],
            program_state.ci_coefficients[-2],
            program_state.ci_coefficients[-1],
            program_state.current_step,
            program_state.step_size.as_atomic(),
            program_state.orb_ini,
            program_state.orb_final,
        )
        return nacme

    def _run_rwfdump(self, rwf_file, output_file, code, debug=False):
        """
        runs `rwfdump rwf_file output file code`
        e.g. `rwfdump some.rwf ao_overlap.dat 514R`
        """
        args = [self.rwfdump, rwf_file, output_file, code]
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

        return True

    def _read_mo_coeff(self, rwf_file):
        """runs rwfdump and reads molecular orbital coefficients matrix"""
        self._run_rwfdump(rwf_file, "mo_coeff.dat", "524R")
        values = self._read_rwf_output("mo_coeff.dat")

        # TODO adjust for frozen core
        dim = np.sqrt(len(values))
        if dim % 1 != 0:
            raise RuntimeError("unexpected length of MO coefficient array: %i" % len(values))
        dim = int(dim)
        mo_coeff = np.reshape(values, (dim, dim))
    
    # TODO try out spin flip states
    def _read_ci_coeff(self, rwf_file, num_roots, num_occupied, num_virtual):
        """runs rwfdump and reads XY coefficients"""
        self._run_rwfdump(rwf_file, "ci_coeff.dat", "635R")

        # excited state coefficients are num_occupied x num_virtual
        # Gaussian calculates 4x more roots than requested
        # x2 for spin degrees of freedom
        # x2 for root solutions
        # I guess there's other stuff in the file that we don't need to read
        num_used_coeff = np.prod([4 * num_roots, 2, 2, num_occupied, num_virtual])
        values = self._read_rwf_output("ci_coeff.dat", limit=num_used_coeff + 12)
        # I guess the first 12 values are all zero and we don't need them
        values = values[12:]

        x_plus_y, x_minus_y = np.reshape(values, (2, num_roots, 2, -1))
        x = 0.5 * (x_plus_y + x_minus_y)

        # removes beta coefficients
        x = x[:num_roots, 0, :]
        return x.reshape(-1, num_occupied, num_virtual)        

    def _read_ao_overlap(self, rwf_file):
        """runs rwfdump and reads atomic orbital overlap matrix"""
        self._run_rwfdump(rwf_file, "ao_overlap", "514R")

        # ao_overlap is only a triangle of the full matrix
        values = self._read_rwf_output("ao_overlap.dat")
        ao_overlap_dim = _faulhaber_inverse(len(values))
        # start with array of zeros and fill the upper triangle
        ao_overlap = np.zeros((ao_overlap_dim, ao_overlap_dim))
        ao_overlap[np.triu_indices(ao_overlap_dim)] = values
        # when we fill the lower triangle by adding the transpose, also
        # double the diagonal, which we don't want
        ao_overlap = ao_overlap + ao_overlap.T - np.diagonal(ao_overlap)
        return ao_overlap

    @staticmethod
    def _read_rwf_output(filename, limit=None):
        """
        read values from filename and return a vector of the values
        limit: only read this many values
        """
        values = []
        with open(filename, "r") as f:
            # output starts with other data
            line = f.readline()
            while not line.startswith(" Dump of file") and line:
                line = f.readline()

            if not line:
                raise RuntimeError("could not read rwf file: %s" % filename)

            # read the actual data
            line = f.readline()
            while line:
                values.extend([float(x.replace("D", "e")) for x in line.split()])
                if limit and len(values) >= limit:
                    return values[:limit]
                line = f.readline()

        return values

    def _run_job(self, job_name, molecule, theory, debug=False):
        """Call Gaussian and return a string with the name of the log file."""
        job_com_file = f"{job_name}.com"
        job_log_file = f"{job_name}.log"
        self._prepare_com_file(
            job_com_file,
            molecule,
            theory,
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
    def _prepare_com_file(file_name, molecule, theory):
        """Prepare a .com file for a Gaussian run."""
        molecule.write(
            outfile=file_name,
            theory=theory,
            style="gaussian",
        )

    @staticmethod
    def _grab_data(log_file_name, program_state):
        """Parse forces into program_state from the given log file."""
        fr = FileReader(log_file_name, just_geom=False)
        if not fr["finished"]:
            raise exceptions.ElectronicStructureProgramError(
                "Gaussian force calculation log file was not valid. Gaussian "
                "returned an error or could not be called correctly."
            )
        
        forces = containers.Forces()
        energy = containers.Energies()
        state_nrg = containers.Energies()
        state_nrg.append(fr["energy"], enums.EnergyUnits.HARTREE)
        # state energy is ground state energy + excitation energy
        for i_state in range(1, program_state.number_of_electronic_states):
            state_nrg.append(
                fr["energy"] + (
                    JOULE_TO_HARTREE * ELECTRON_VOLT_TO_JOULE * \
                    fr["uv_vis"].data[i_state - 1]
                ),
                enums.EnergyUnits.HARTREE
            )
        program_state.state_energies.append(state_nrg)
        
        energy.append(
            state_nrg.as_hartree(index=-1)[program_state.current_electronic_state],
            enums.EnergyUnits.HARTREE
        )
        program_state.energies.append(energy)
        for v in fr["forces"]:
            forces.append(*v, enums.ForceUnits.HARTREE_PER_BOHR)
        program_state.forces.append(forces)
        program_state.number_of_basis_functions = fr["n_basis"]
        program_state.number_of_frozen_core = fr["n_frozen"]
        program_state.number_of_alpha_occupied = fr["n_occupied_alpha"]
        program_state.number_of_alpha_virtual = fr["n_virtual_alpha"]
        program_state.number_of_beta_occupied = fr["n_occupied_beta"]
        program_state.number_of_beta_virtual = fr["n_virtual_beta"]


