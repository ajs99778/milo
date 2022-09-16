#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle calling ESPs and parsing output."""

import os
import subprocess
from sys import platform

from milo import containers
from milo import enumerations as enums
from milo import exceptions
from milo.scientific_constants import ELECTRON_VOLT_TO_JOULE, JOULE_TO_HARTREE, AMU_TO_KG
from milo.scientific_constants import h as PLANCKS_CONSTANT
from milo.molecule import Molecule

from AaronTools.fileIO import FileReader
from AaronTools.theory import ForceJob, TDDFTJob

from unixmd.qm.cioverlap import wf_overlap
from unixmd.mqc.el_propagator import el_run

import numpy as np

import struct

from scipy.special import factorial2
from scipy.spatial import distance_matrix

def get_program_handler(program_state, nonadiabatic=False):
    """Return the configured electronic structure program handler."""
    print(program_state.program_id)
    if program_state.program_id is enums.ProgramID.GAUSSIAN:
        if nonadiabatic:
            return GaussianSurfaceHopHandler(program_state.executable)
        return GaussianHandler(program_state.executable)
    
    elif program_state.program_id is enums.ProgramID.ORCA:
        if nonadiabatic:
            return ORCASurfaceHopHandler(program_state.executable)
        return ORCAHandler(program_state.executable)
    
    elif program_state.program_id is enums.ProgramID.QCHEM:
        if not nonadiabatic:
            return QChemHandler(program_state.executable)
        raise ValueError("nonadiabatic dynamics not currently supported with Q-Chem")
    
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
        """Call ORCA and return a string with the name of the log file."""
        job_inp_file = f"{job_name}.inp"
        job_out_file = f"{job_name}.out"
        self._prepare_inp_file(
            job_inp_file,
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
    def _prepare_inp_file(file_name, program_state):
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


class QChemHandler(ProgramHandler):
    """handler for Q-Chem."""

    def generate_forces(self, program_state):
        """Preform computation and append forces to list in program state."""
        out_file = self._run_job(
            f"_{program_state.current_step}",
            program_state,
        )
        self._grab_forces(out_file, program_state)
        return True

    def _run_job(self, job_name, program_state, debug=False):
        """Call Q-Chem and return a string with the name of the log file."""
        job_inp_file = f"{job_name}.inq"
        job_out_file = f"{job_name}.qout"
        self._prepare_inq_file(
            job_inp_file,
            program_state,
        )
        args = [self.executable, "-nt", "%i" % int(program_state.theory.processors), job_inp_file, job_out_file]
        kwargs = {"stdout": subprocess.DEVNULL}
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

        return job_out_file

    @staticmethod
    def _prepare_inq_file(file_name, program_state):
        """Prepare a .inq file for an Q-Chem run."""
        program_state.molecule.write(
            outfile=file_name,
            theory=program_state.theory,
            style="qchem",
            rem={"SYM_IGNORE": "TRUE"},
        )

    @staticmethod
    def _grab_forces(out_file_name, program_state):
        """Parse forces into program_state from the given out file."""
        fr = FileReader(out_file_name, just_geom=False)
        if not fr["finished"]:
            raise exceptions.ElectronicStructureProgramError(
                "Q-Chem force calculation out file was not valid. Q-Chem "
                "returned an error or could not be called correctly."
            )
        
        forces = containers.Forces()
        energy = containers.Energies()
        
        energy.append(fr["energy"], enums.EnergyUnits.HARTREE)
        program_state.energies.append(energy)
        for v in fr["forces"]:
            forces.append(*v, enums.ForceUnits.HARTREE_PER_BOHR)
        program_state.forces.append(forces)



class NumericalNonAdiabaticSurfaceHopHandler(ProgramHandler):
    """general handler for nonadiabatic dynamics with numerical NAC calculations"""

    def _propogate_electronic(self, program_state):
        """
        propogate wavefunction
        """
        el_run(
            program_state,
            program_state.electronic_propogation_type,
        )

    def _scale_velocities(self, program_state):
        """scale velocities to conserve energy after hopping"""
        current_kinetic = 0
        for i in range(0, program_state.number_atoms):
            mass = program_state.atoms[i].mass
            velocity = program_state.velocities[-1].as_meter_per_sec(index=i)
            current_kinetic += mass + np.dot(velocity, velocity)
        current_kinetic /= 2
        current_kinetic *= AMU_TO_KG
        # current_electronic_state has already been moved to previous_electronic_state
        current_potential = program_state.state_energies[-1].as_joules(
            index=program_state.previous_electronic_state
        )
        target_potential = program_state.state_energies[-1].as_joules(
            index=program_state.current_electronic_state
        )
        target_kinetic = current_kinetic + current_potential - target_potential
        scale_factor = np.sqrt(target_kinetic / current_kinetic)
        print("scaling velocities by %.2f" % scale_factor, flush=True)
        new_velocities = containers.Velocities()
        for i in range(0, program_state.number_atoms):
            old_velocity = program_state.velocities[-1].as_meter_per_sec(index=i)
            new_velocity = [scale_factor * v for v in old_velocity]
            new_velocities.append(*new_velocity, enums.VelocityUnits.METER_PER_SEC)
        program_state.velocities[-1] = new_velocities

    def _decohere(self, program_state):
        """apply decoherence correction"""
        current_state = program_state.current_electronic_state
        number_of_states = program_state.number_of_electronic_states
        # IDC
        if program_state.decoherence_correction == "instantaneous":
            program_state.rho = np.zeros(
                (number_of_states, number_of_states), dtype=np.cdouble,
            )
            program_state.rho[current_state, current_state] = 1. + 0.j
        
            program_state.state_coefficients = np.zeros(number_of_states, dtype=np.cdouble)
            program_state.state_coefficients[current_state] = 1. + 0.j

        # EDC
        elif program_state.decoherence_correction == "energy-based":

            for i in range(0, program_state.number_atoms):
                mass = program_state.atoms[i].mass
                velocity = program_state.velocities[-1].as_meter_per_sec(index=i)
                current_kinetic += mass * np.dot(velocity, velocity)
            current_kinetic /= 2
            current_kinetic *= AMU_TO_KG
            current_kinetic *= JOULE_TO_HARTREE

            tau = np.zeros(number_of_states)
            for i in range(0, number_of_states):
                if i == program_state.current_electronic_state:
                    continue
                nrg_difference = abs(
                    program_state.state_energies[-1].as_joules(index=i) - program_state.state_energies[-1].as_joules(index=current_state)
                )
                tau[i] = PLANCKS_CONSTANT * (1 + program_state.edc_parameter.as_hartree()[0] / current_kinetic) / (2 * np.pi * nrg_difference)

            exp_tau = np.exp(tau)

            rho_update = 1.
            if program_state.electronic_propogation_type == "coefficient":
                for i_state in range(0, number_of_states):
                    if (i_state != current_state):
                        program_state.state_coefficients[i_state] *= exp_tau[i_state]
                        rho_update -= program_state.state_coefficients[i_state].conjugate() * program_state.state_coefficients[i_state]

                program_state.state_coefficients[current_state] = np.sqrt(
                    rho_update / program_state.rho[current_state, current_state]
                )

                for i_state in range(0, number_of_states):
                    for j_state in range(i_state, number_of_states):
                        program_state.rho[i_state, j_state] = (
                            program_state.state_coefficients[i_state].conjugate() * program_state.state_coefficients[j_state]
                        )
                        program_state.rho[j_state, i_state] = program_state.rho[i_state, j_state].conjugate()

            elif program_state.electronic_propogation_type == "density":
                old_rho_running_state = program_state.rho[current_state, current_state]
                for i_state in range(number_of_states):
                    for j_state in range(i_state, number_of_states):
                        program_state.rho[i_state, j_state] *= exp_tau[i_state] * exp_tau[j_state]
                        program_state.rho[j_state, i_state] = program_state.rho[i_state, j_state].conjugate()

                    if (i_state != current_state):
                        rho_update -= program_state.rho[i_state, i_state]

                for i_state in range(number_of_states):
                    program_state.rho[i_state, current_state] *= np.sqrt(rho_update / old_rho_running_state)
                    program_state.rho[current_state, i_state] *= np.sqrt(rho_update / old_rho_running_state)
        
        # if there is not decoherence correction, do nothing

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
                cummulative_probability[i_state + 1] = accumulator
                continue
            state_probability[i_state] = -2 * (
                program_state.rho.real[i_state, running_state] * \
                program_state.nacmes[-1][i_state, running_state] * \
                program_state.step_size.as_atomic() / program_state.rho.real[running_state, running_state]
            )
            if state_probability[i_state] < 0:
                state_probability[i_state] = 0
            accumulator += state_probability[i_state]
            cummulative_probability[i_state + 1] = accumulator
        total_probability = cummulative_probability[program_state.number_of_electronic_states]

        if total_probability > 1:
            state_probability /= total_probability
            cummulative_probability /= total_probability

        print("total probabilities", total_probability)
        print("probabilities", state_probability, flush=True)

        random_number = program_state.random.uniform()
        for i_state in range(0, program_state.number_of_electronic_states):
            if i_state == running_state:
                continue
            if random_number > cummulative_probability[i_state] and random_number <= cummulative_probability[i_state + 1]:
                # probability condition is met, but we still need to ensure energy is conserved
                # determine if current PE + KE < new PE
                # if so, this is a 'frustrated' hop, and the state does not change
                current_kinetic = 0
                for i in range(0, program_state.number_atoms):
                    mass = program_state.atoms[i].mass
                    velocity = program_state.velocities[-1].as_meter_per_sec(index=i)
                    current_kinetic += mass * np.dot(velocity, velocity)
                current_kinetic /= 2
                current_kinetic *= AMU_TO_KG
                current_potential = program_state.energies[-1].as_joules()
                target_potential = program_state.state_energies[-1].as_joules(index=i_state)
                if current_kinetic + current_potential < target_potential:
                    print("frustated hop from state %i to %i" % (running_state, i_state), flush=True)
                    continue
                program_state.previous_electronic_state = program_state.current_electronic_state
                program_state.current_electronic_state = i_state
                return True
        
        return False
    
    def _compute_nacme(self, program_state):
        nacme = np.zeros(
            (program_state.number_of_electronic_states, program_state.number_of_electronic_states),
            dtype=np.double
        )
        wf_overlap(
            program_state,
            program_state.step_size.as_atomic(),
            nacme,
        )
        return nacme




class GaussianSurfaceHopHandler(NumericalNonAdiabaticSurfaceHopHandler):
    """handler for Gaussian 16 and 09 with Unix-MD surface hopping algorithms"""
    def __init__(self, executable):
        self.executable = executable
        gaussian_root, gaussian_exe = os.path.split(executable)
        gaussian_name, extension = os.path.splitext(gaussian_exe)
        # rwfdump should be in the same place as the gaussian executable
        # if g09/g16 is in the $PATH, so is rwfdump
        self.rwfdump = os.path.join(gaussian_root, "rwfdump" + extension)

    def generate_forces(self, program_state):
        """
        Preform computation and append forces to list in program state.
        also determines if the state should change using Unix-MD
        """

        print("current state:", program_state.current_electronic_state)
            
        # fix up the theory to read SCF orbitals from
        # the previous iteration
        # this might help with SCF convergence issues
        if not program_state.force_theory:
            force_theory = program_state.theory.copy()
            force_theory.add_kwargs(
                link0={
                    "chk": ["force_job.chk"],
                    "rwf": ["single.rwf"],
                },
                route={"NoSymm": []},
            )
        else:
            force_theory = program_state.force_theory
        
        if os.path.exists("force_job.chk") and "guess" not in force_theory.kwargs["route"]:
            force_theory.add_kwargs(
                route={"guess": ["read"]},
            )
            program_state.force_theory = force_theory
        
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
            retry=3,
        )
        forces, energy, state_energy = self._grab_data(force_log_file, program_state)
        program_state.state_energies.append(state_energy)
        program_state.forces.append(forces)
        
        self._compute_overlaps(program_state)
      
        # need to have a dx/dt before we can calculate the NAC
        # so we don't do it the first iteration
        if program_state.current_step > 0:
            # surface hopping calculation
            # calculate overlap with the wavefunction from the previous
            # iteration
            # TODO: decoherence 
            program_state.nacmes.append(self._compute_nacme(program_state))

            self._propogate_electronic(program_state)
            hop = self._check_hop(program_state)
            if hop:
                self._decohere(program_state)
                self._scale_velocities(program_state)
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
                # state energies shouldn't change, but force and energy would
                forces, energy, _ = self._grab_data(force_log_file, program_state)
       
        else:
            program_state.nacmes.append(
                np.zeros((program_state.number_of_electronic_states, program_state.number_of_electronic_states), dtype=np.double)
            )

        program_state.forces[-1] = forces
        program_state.energies.append(energy)
        
        return True

    def _compute_overlaps(self, program_state):
        """
        compute atomic orbital overlap and overlap between wavefunctions
        from this iteration and the previous one
        """
        if not program_state.overlap_theory:
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
            program_state.overlap_theory = overlap_theory
        else:
            overlap_theory = program_state.overlap_theory
        
        # coefficients from the previous iteration are needed for NAC
        # and propogating the wave function
        program_state.previous_mo_coefficients = program_state.current_mo_coefficients
        program_state.previous_ci_coefficients = program_state.current_ci_coefficients
        
        if program_state.current_step > 0:
            # the structure from the previous iteration
            prev_mol = program_state.molecule.copy()
            prev_mol.coords = program_state.structures[-2].as_angstrom()
            overlap_mol = Molecule([*program_state.molecule.atoms, *prev_mol.atoms])
            overlap_rwf = self._run_job(
                f"_{program_state.current_step}_overlap",
                overlap_mol,
                overlap_theory,
            )
            ao_overlap = self._read_ao_overlap("overlap.rwf", program_state.number_of_basis_functions)
            # we only care about the overlap between the current AOs with the previous AOs
            # the upper left quadrant is the overlap of the current iteration overlap only
            # the lower right quadrant is the previous iteration overlap only
            dim = program_state.number_of_basis_functions
            program_state.atomic_orbital_overlap = ao_overlap[:dim, dim:]
        
        program_state.current_mo_coefficients = self._read_mo_coeff("single.rwf")
        program_state.current_mo_coefficients = program_state.current_mo_coefficients[
            program_state.number_of_frozen_core:program_state.number_of_basis_functions
        ]
        program_state.current_ci_coefficients = np.zeros((
            program_state.number_of_electronic_states,
            program_state.number_of_alpha_occupied,
            program_state.number_of_alpha_virtual
        ))
        program_state.current_ci_coefficients[1:] = self._read_ci_coeff(
            "single.rwf",
            program_state.number_of_electronic_states - 1,
            program_state.number_of_alpha_occupied,
            program_state.number_of_alpha_virtual,
        )

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

        dim = np.sqrt(len(values))
        if dim % 1 != 0:
            raise RuntimeError("unexpected length of MO coefficient array: %i" % len(values))
        dim = int(dim)
        mo_coeff = np.reshape(values, (dim, dim))
   
        return mo_coeff

    def _read_ci_coeff(self, rwf_file, num_roots, num_occupied, num_virtual):
        """runs rwfdump and reads XY coefficients"""
        self._run_rwfdump(rwf_file, "ci_coeff.dat", "635R")

        # excited state coefficients are num_occupied x num_virtual
        # Gaussian calculates 4x more roots than requested
        # x2 for spin degrees of freedom
        # x2 for root solutions
        # I guess there's other stuff in the file that we don't need to read
        num_used_coeff = np.prod([4, num_roots, 2, 2, num_occupied, num_virtual])
        values = self._read_rwf_output("ci_coeff.dat", limit=num_used_coeff + 12)
        # I guess the first 12 values are all zero and we don't need them
        values = values[12:]

        x_plus_y, x_minus_y = np.reshape(values, (2, 4 * num_roots, 2, -1))
        x = 0.5 * (x_plus_y + x_minus_y)

        # removes beta coefficients
        x = x[:num_roots, 0, :]
        return x.reshape(-1, num_occupied, num_virtual) 

    def _read_ao_overlap(self, rwf_file, num_basis):
        """runs rwfdump and reads atomic orbital overlap matrix"""
        self._run_rwfdump(rwf_file, "ao_overlap.dat", "514R")

        # ao_overlap is only a triangle of the full matrix
        values = self._read_rwf_output("ao_overlap.dat")
        # this has the AOs from the previous iteration
        # and the current iteration, so 2x the basis functions
        ao_overlap_dim = 2 * num_basis
        # start with array of zeros and fill the upper triangle
        ao_overlap = np.zeros((ao_overlap_dim, ao_overlap_dim))
        ao_overlap[np.tril_indices(ao_overlap_dim)] = values
        # when we fill the lower triangle by adding the transpose, it will
        # also double the diagonal, which we don't want
        # XXX we probably don't need to do this, as we only use one quadrant
        # of the AO overlap
        ao_overlap = ao_overlap + ao_overlap.T
        ao_overlap[np.diag_indices(ao_overlap_dim)] *= 0.5
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
                # D = 10^, but python uses e for 10^
                values.extend([float(x.replace("D", "e")) if "D" in x else float(x.replace("-", "e-")) for x in line.split()])
                if limit and len(values) >= limit:
                    return values[:limit]
                line = f.readline()

        return values

    def _run_job(self, job_name, molecule, theory, debug=False, retry=False):
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
        
        # SCF errors seem to be common
        # try to resolve errors automatically
        # AaronTools will request the XQC SCF algorithm if the SCF does not converge
        if retry:
            fr = FileReader(job_log_file, just_geom=False)
            if fr["error"]:
                print("encountered a gaussian error: %s\nattempting to resolve" % fr["error_msg"])
                new_theory = theory.copy()
                fix_attempted = False
                for job in theory.job_type:
                    try:
                        new_theory = job.resolve_error(
                            fr["error"], new_theory, "gaussian",
                        )
                        fix_attempted = True
                    except NotImplementedError:
                        pass
                
                if fix_attempted:
                    new_theory.remove_kwargs(route={"guess": ["read"]})
                    
                    old_header = theory.make_header(
                        geom=molecule, style="gaussian"
                    )
                    new_header = new_theory.make_header(
                        geom=molecule, style="gaussian"
                    )
                    print("trying a potential fix")
                    print("============ OLD HEADER ============")
                    print(old_header)
                    print("============ NEW HEADER ============")
                    print(new_header)

                    print("%i retries will remain" % (retry - 1))

                    self._run_job(
                        job_name,
                        molecule,
                        new_theory,
                        retry=retry - 1,
                        debug=debug
                    )
                else:
                    print("error could not be resolved")

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
            nrg = fr["energy"] + (
                JOULE_TO_HARTREE * ELECTRON_VOLT_TO_JOULE * \
                fr["uv_vis"].data[i_state - 1].excitation_energy
            )
        
            state_nrg.append(nrg, enums.EnergyUnits.HARTREE)

        energy.append(
            state_nrg.as_hartree()[program_state.current_electronic_state],
            enums.EnergyUnits.HARTREE
        )
        for v in fr["forces"]:
            forces.append(*v, enums.ForceUnits.HARTREE_PER_BOHR)
        program_state.number_of_basis_functions = fr["n_basis"]
        program_state.number_of_frozen_core = fr["n_frozen"]
        program_state.number_of_alpha_occupied = fr["n_occupied_alpha"]
        program_state.number_of_alpha_virtual = fr["n_virtual_alpha"]
        program_state.number_of_beta_occupied = fr["n_occupied_beta"]
        program_state.number_of_beta_virtual = fr["n_virtual_beta"]
        program_state.orb_final = (fr["n_occupied_alpha"] + fr["n_virtual_alpha"]) * np.ones(1, dtype=np.int32)

        return forces, energy, state_nrg



class ORCASurfaceHopHandler(NumericalNonAdiabaticSurfaceHopHandler):
    """handler for ORCA with Unix-MD surface hopping algorithms"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we calculate the AO overlap instead of reading it
        self.basis = None
        self.shells = []
        # the code I have for calculating the AO overlap is for
        # cartesian basis functions, so we need to map those to
        # basis functions with pure angular momentum (ORCA <= 5
        # only uses pure basis functions)
        # these arrays map cartesian to pure in the order ORCA uses
        # up to i orbitals
        # some common constants in the mapping:
        r2 = np.sqrt(2)
        r3 = np.sqrt(3)
        r5 = np.sqrt(5)
        r10 = r2 * r5
        r14 = np.sqrt(14)
        r15 = r3 * r5
        r21 = np.sqrt(21)
        r35 = np.sqrt(35)
        r63 = r3 * r21
        r70 = r2 * r35
        r105 = r3 * r35
        r154 = np.sqrt(154)
        r210 = r3 * r70
        r462 = r3 * r154

        # s
        s = np.ones((1, 1))
        # p: px, py, pz -> pz, px, py
        p = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ])
        # d
        # x2, xy, xz, y2, yz, z2 ->
        #   1.5 z2 - 0.5 x2 - 0.5 yz - 0.5 z2),
        #   xz
        #   yz
        #   sqrt(3) * x2 / 2 - sqrt(3) * y2 / 2
        #   xy
        d = np.array([
            [-1 / 2, 0, 0, -1 / 2, 0, 1],
            [0, 0, r3, 0, 0, 0],
            [0, 0, 0, 0, r3, 0],
            [np.sqrt(3) / 2, 0, 0, -np.sqrt(3) / 2, 0, 0],
            [0, r3, 0, 0, 0, 0],
        ])
        # f
        # x3, x2y, x2z, xy2, xyz, xz2, y3, y2z, yz2, z3 ->
        #   5 z3 / 2 - 3 (x2z + y2z + z3) / 2
        #   sqrt(3) * (5 xz2 - (x3 + xy2 + xz2)) / (2 * sqrt(2))
        #   sqrt(3) * (5 yz2 - (x2y + y3 + yz2)) / (2 * sqrt(2))
        #   sqrt(15) * (x2z - y2z) / 2
        #   sqrt(15) * xyz
        #   sqrt(5) * (3 xy2 -x3) / (2 * sqrt(2))
        #   sqrt(5) * (y3 -3 x2y) / (2 * np.sqrt(2))
        f = np.zeros((7, 10))
        f[0, 2] = -3. / 2
        f[0, 7] = -3. / 2
        f[0, 9] = 1
        
        f[1, 0] = -r3 / (2 * r2)
        f[1, 3] = -r3 / (2 * r2)
        f[1, 5] = 2 * r3 / r2
        
        f[2, 1] = -r3 / (2 * r2)
        f[2, 6] = -r3 / (2 * r2)
        f[2, 8] = 2 * r3 / r2
        
        f[3, 2] = r15 / 2
        f[3, 7] = -r15 / 2
        
        f[4, 4] = r15
        
        f[5, 0] = -r5 / (2 * r2)
        f[5, 3] = 3 * r5 / (2 * r2)
        
        f[6, 1] = -3 * r5 / (2 * r2)
        f[6, 6] = r5 / (2 * r2)
        # g
        # x4 x3y x3z x2y2 x2yz x2z2 xy3 xy2z xyz2 xz3 y4 y3z y2z2 yz3 z4
        #   (35 z4 - 30 (x2z2 + y2z2 + z4) + 3 (
        #       x4 + 2 x2y2 + 2 x2z2 + y4 + 2 y2z2 + z4
        #   )) / 8
        #   sqrt(10) * (7 xz3 - 3 (x3z + xy2z + xz3)) / 4
        #   sqrt(10) * (7 yz3 - 3 (x2yz + y3z + yz3)) / 4
        #   sqrt(5) * (7 x2z2 - (x4 + x2y2 + x2z2) - 7 y2z2 + x2y2 + y4 + y2z2) / 4
        #   sqrt(5) * (7 xyz2 - (x3y + xy3 + xyz2)) / 2
        #   -sqrt(70) * (x3z - 3 xy2z) / 4
        #   -sqrt(70) * (3 x2yz - y3z) / 4
        #   -sqrt(35) * (x4 - 3 x2y2 - 3 x2y2 + y4) / 8
        #   -sqrt(35) * (x3y - xy3) / 2        
        g = np.zeros((9, 15))
        g[0, 0] = 3 / 8
        g[0, 3] = 3 / 4
        g[0, 5] = -3
        g[0, 10] = 3 / 8
        g[0, 12] = -3
        g[0, 14] = 1
        
        g[1, 9] = r10
        g[1, 2] = -3 * r10 / 4
        g[1, 7] = -3 * r10 / 4
        
        g[2, 13] = r10
        g[2, 4] = -3 * r10 / 4
        g[2, 11] = -3 * r10 / 4
        
        g[3, 5] = 3 * r5 / 2
        g[3, 0] = -r5 / 4
        g[3, 12] = -3 * r5 / 2
        g[3, 10] = r5 / 4
        
        g[4, 8] = 3 * r5
        g[4, 1] = -r5 / 2
        g[4, 6] = -r5 / 2
        
        g[5, 2] = -r70 / 4
        g[5, 7] = 3 * r70 / 4
        
        g[6, 4] = -3 * r70 / 4
        g[6, 11] = r70 / 4
        
        g[7, 0] = -r35 / 8
        g[7, 3] = 3 * r35 / 4
        g[7, 10] = -r35 / 8
        
        g[8, 1] = -r35 / 2
        g[8, 6] = r35 / 2
        # h
        # x5 x4y x4z x3y2 x3yz x3z2 x2y3 x2y2z x2yz2 x2z3 xy4 xy3z xy2z2 xyz3 xz4 y5 y4z y3z2 y2z3 yz4 z5 ->
        #  0    (63 z5 - 70 (x2z3 + y2z3 + z5) + 15 (
        #           x4z + 2 x2y2z + 2 x2z3 + y4z + 2 y2z3 + z5
        #       )) / 8
        #  1    sqrt(15) (21 xz4 - 14 (x3z2 + xy2z2 + xz4) + (
        #           x5+ 2 x3y2 + 2 x3z2 + xy4 + 2 xy2z2 + xz4
        #       )) / 8
        #  2    sqrt(15) (21 yz4 - 14 (x2yz2 + y3z2 + yz4) + (
        #           x4y + 2 x2y3 + 2 x2yz2 + y5 + 2 y3z2 + yz4
        #       )) / 8
        #  3    sqrt(105) (3 x2z3 - (x4z + x2y2z + x2z3) - 3 y2z3 + (x2y2z + y4z + y2z3)) / 4
        #  4    sqrt(105) (3 xyz3 - (x3yz + xy3z + xyz3)) / 2
        #  5    -35 (9 x3z2 - (x5 + x3y2 + x3z2) - 27 xy2z2 + 3 (x3y2 + xy4 + xy2z2)) / (8 sqrt(70))
        #  6    -35 (27 x2yz2  - 3 (x4y + x2y3 + x2yz2) - 9 y3z2 + (x2y3 + y5 + y3z2)) / (8 sqrt(70))
        #  7    -105 (x4z - 6 x2y2z + y4z) / (8 * sqrt(35))
        #  8    -105 (x3yz - xy3z) / (2 sqrt(35))
        #  9    21 (x5 - 10 x3y2 + 5 xy4) / (8 sqrt(14))
        # 10    21 (5 x4y - 10 x2y3 + y5) / (8 sqrt(14))
        h = np.zeros((11, 21))
        h[0, 2] = 15 / 8
        h[0, 7] = 30 / 8
        h[0, 9] = -5
        h[0, 16] = 15 / 8
        h[0, 18] = -5
        h[0, 20] = 1
        
        h[1, 0] = r15 / 8
        h[1, 3] = r15 / 4
        h[1, 5] = -3 * r15 / 2
        h[1, 10] = r15 / 8
        h[1, 12] = -3 * r15 / 2
        h[1, 14] = r15
        
        h[2, 1] = r15 / 8
        h[2, 6] = r15 / 4
        h[2, 8] = -3 * r15 / 2
        h[2, 15] = r15 / 8
        h[2, 17] = -3 * r15 / 2
        h[2, 19] = r15
        
        h[3, 9] = r105 / 2
        h[3, 2] = -r105 / 4
        h[3, 18] = -r105 / 2
        h[3, 16] = r105 / 4
        
        h[4, 13] = r105
        h[4, 4] = -r105 / 2
        h[4, 11] = -r105 / 2
        
        h[5, 5] = -35 / r70
        h[5, 0] = 35 / (8 * r70)
        h[5, 3] = -35 / (4 * r70)
        h[5, 12] = 105 / r70
        h[5, 10] = -105 / (8 * r70)
        
        h[6, 8] = -105 / r70
        h[6, 1] = 105 / (8 * r70)
        h[6, 6] = 35 / (4 * r70)
        h[6, 17] = 35 / r70
        h[6, 15] = -35 / (8 * r70)
        
        h[7, 2] = -105 / (8 * r35)
        h[7, 7] = -6 * -105 / (8 * r35)
        h[7, 16] = -105 / (8 * r35)
        
        h[8, 4] = -105 / (2 * r35)
        h[8, 11] = 105 / (2 * r35)
        
        h[9, 0] = 21 / (8 * r14)
        h[9, 3] = -105 / (4 * r14)
        h[9, 10] = 105 / (8 * r14)
        
        h[10, 1] = 105 / (8 * r14)
        h[10, 6] = -105 / (4 * r14)
        h[10, 15] = 21 / (8 * r14)
        # i
        #  0   1   2    3    4    5    6     7     8    9   10    11     12
        # x6 x5y x5z x4y2 x4yz x4z2 x3y3 x3y2z x3yz2 x3z3 x2y4 x2y3z x2y2z2
        #    13   14  15   16    17    18   19  20 21  22   23   24   25
        # x2yz3 x2z4 xy5 xy4z xy3z2 xy2z3 xyz4 xz5 y6 y5z y4z2 y3z3 y2z4
        #  26 27
        # yz5 z6 ->
        #  0 (231 z6 - 315 (x2z4 + y2z4 + z6) + 105 (
        #       x4z2 + 2 x2y2z2 + 2 x2z4 + y4z2 + 2 y2z4 + z6
        #    ) - 5 (
        #       x6 + 3 x4y2 + 3 x4z2 + 3 x2y4 + 6 x2y2z2 + 3 x2z4 + y6 + 3 y4z2 + 3 y2z4 + z6
        #    )) / 16
        #  1 sqrt(21) (33 xz5 - 30 (x3z3 + xy2z3 + xz5) + 5 (
        #       x5z + 2 x3y2z + 2 x3z3 + xy4z + 2 xy2z3 + xz5
        #    )) / 8
        #  2 sqrt(21) (33 yz5 - 30 (x2yz3 + y3z3 + yz5) + 5 (
        #       x4yz + 2 x2y3z + 2 x2yz3 + y5z + 2 y3z3 + yz5
        #    )) / 8
        #  3 105 (33 x2z4 - 18 (x4z2 + x2y2z2 + x2z4) + (
        #       x6 + 2 x4y2 + 2 x4z2 + x2y4 + 2 x2y2z2 + x2z4
        #    ) - 33 y2z4 + 18 (x2y2z2 + y4z2 + y2z4) - (
        #       x4y2 + 2 x2y4 + 2 x2y2z2 + y6 + 2 y4z2 + y2z4
        #    ))/ (16 sqrt(210))
        #  4 105 (33 xyz4 - 18 (x3yz2 + xy3z2 + xyz4) + x5y + 2 x3y3 + 2 x3yz2 + xy5 + 2 xy3z2 + xyz4) / (8 * sqrt(210))
        #  5 -105 (11 x3z3 - 3 (x5z + x3y2z + x3z3) - 33 xy2z3 + 9 (x3y2z + xy4z + xy2z3)) / (8 sqrt(210))
        #  6 -105 (33 x2yz3 - 9 (x4yz + x2y3z + x2yz3) - 11 y3z3 + 3 (x2y3z + y5z + y3z3)) / (8 sqrt(210))
        #  7 -sqrt(63) (11 x4z2 - (x6 + x4y2 + x4z2) - 66 x2y2z2 + 6 (x4y2 + x2y4 + x2y2z2) + 11 y4z2 - (x2y4 + y6 + y4z2)) / 16 
        #  8 -sqrt(63) (11 x3yz2 - (x5y + x3y3 + x3yz2) - 11 xy3z2 + (x3y3 + xy5 + xy3z2)) / 4
        #  9 231 (x5z - 10 x3y2z + 5 xy4z) / (8 sqrt(154))
        # 10 231 (5 x4yz - 10 x2y3z + y5z) / (8 sqrt(154))
        # 11 231 (x6 - 15 x4y2 + 15 x2y4 - y6) / (16 sqrt(462))
        # 12 231 (6 x5y - 20 x3y3 + 6 xy5) / (16 sqrt(462))
        i = np.zeros((13, 28))
        i[0, 0] = -5 / 16
        i[0, 3] = -15 / 16
        i[0, 5] = 90 / 16
        i[0, 10] = -15 / 16
        i[0, 12] = 180 / 16
        i[0, 14] = -120 / 16
        i[0, 21] = -5 / 16
        i[0, 23] = 90 / 16
        i[0, 25] = -120 / 16
        i[0, 27] = 1
        
        i[1, 2] = 5 * r21 / 8
        i[1, 7] = 10 * r21 / 8
        i[1, 9] = -20 * r21 / 8
        i[1, 16] = 5 * r21 / 8
        i[1, 18] = -20 * r21 / 8
        i[1, 20] = r21
        
        i[2, 4] = 5 * r21 / 8
        i[2, 11] = 10 * r21 / 8
        i[2, 13] = -20 * r21 / 8
        i[2, 22] = 5 * r21 / 8
        i[2, 24] = -20 * r21 / 8
        i[2, 26] = r21
        
        i[3, 0] = 105 / (16 * r210)
        i[3, 3] = 105 / (16 * r210)
        i[3, 5] = -105 / r210
        i[3, 10] = -105 / (16 * r210)
        i[3, 14] = 105 / r210
        i[3, 21] = -105 / (16 * r210)
        i[3, 23] = 105 / r210
        i[3, 25] = -105 / r210
        
        i[4, 1] = 105 / (8 * r210)
        i[4, 6] = 2 * 105 / (8 * r210)
        i[4, 8] = -16 * 105 / (8 * r210)
        i[4, 15] = 105 / (8 * r210)
        i[4, 17] = -16 * 105 / (8 * r210)
        i[4, 19] = 16 * 105 / (8 * r210)
        
        i[5, 2] = -3 * -105 / (8 * r210)
        i[5, 7] = 6 * -105 / (8 * r210)
        i[5, 9] = 8 * -105 / (8 * r210)
        i[5, 16] = 9 * -105 / (8 * r210)
        i[5, 18] = -24 * -105 / (8 * r210)
        
        i[6, 4] = -9 * -105 / (8 * r210)
        i[6, 11] = -6 * -105 / (8 * r210)
        i[6, 13] = 24 * -105 / (8 * r210)
        i[6, 22] = 3 * -105 / (8 * r210)
        i[6, 24] = -8 * -105 / (8 * r210)
        
        i[7, 0] = -1 * -r63 / 16
        i[7, 3] = 5 * -r63 / 16
        i[7, 5] = 10 * -r63 / 16
        i[7, 10] = 5 * -r63 / 16
        i[7, 12] = -60 * -r63 / 16
        i[7, 21] = -1 * -r63 / 16
        i[7, 23] = 10 * -r63 / 16
        
        i[8, 8] = -5 * r63 / 2
        i[8, 1] = r63 / 4
        i[8, 17] = 5 * r63 / 2
        i[8, 15] = -r63 / 4
        
        i[9, 2] = 231 / (8 * r154)
        i[9, 7] = -1155 / (4 * r154)
        i[9, 16] = 1155 / (8 * r154)
        
        i[10, 4] = 5 * 231 / (8 * r154)
        i[10, 11] = -10 * 231 / (8 * r154)
        i[10, 22] = 231 / (8 * r154)
        
        i[11, 0] = 231 / (16 * r462)
        i[11, 3] = -3465 / (16 * r462)
        i[11, 10] = 3465 / (16 * r462)
        i[11, 21] = -231 / (16 * r462)
        
        i[12, 1] = 6 * 231 / (16 * r462)
        i[12, 6] = -20 * 231 / (16 * r462)
        i[12, 15] = 6 * 231 / (16 * r462)

        self.map_cart_2_pure = {
            "s": s, "p": p, "d": d, "f": f, "g": g, "h": h, "i": i,
        }

    def generate_forces(self, program_state):
        """Preform computation and append forces to list in program state."""
        """
        Preform computation and append forces to list in program state.
        also determines if the state should change using Unix-MD
        """

        print("current state:", program_state.current_electronic_state)


        print("set previous to current")
        program_state.previous_ci_coefficients = program_state.current_ci_coefficients
        program_state.previous_mo_coefficients = program_state.current_mo_coefficients

        if program_state.previous_ci_coefficients is not None:
            print("ci coeff")
            print(program_state.previous_ci_coefficients.shape)

        # print MO coefficients to output
        if not program_state.force_theory:
            force_theory = program_state.theory.copy()
            force_theory.add_kwargs(
                simple=[
                    "PrintMOs",
                ],
            )
        else:
            force_theory = program_state.force_theory

        force_theory.job_type = [
            ForceJob(), 
            TDDFTJob(
                program_state.number_of_electronic_states - 1,
                root_of_interest=program_state.current_electronic_state,
            ),
        ]
        # we need the basis set to calculate AO overlap
        if self.basis is None:
            force_theory = force_theory.copy()
            force_theory.add_kwargs(
                simple=["PrintBasis"],
            )
        # run force and excited state computation
        force_out_file = self._run_job(
            f"_{program_state.current_step}_force",
            program_state.molecule,
            force_theory,
            retry=3,
        )
        forces, energy, state_energy, mo_coefficients, ci_coefficients = self._grab_data(
            force_out_file, program_state
        )
        program_state.state_energies.append(state_energy)
        program_state.forces.append(forces)
        program_state.current_mo_coefficients = mo_coefficients
        program_state.current_ci_coefficients = ci_coefficients

        self._compute_overlaps(program_state)
      
        # need to have a dx/dt before we can calculate the NAC
        # so we don't do it the first iteration
        if program_state.current_step > 0:
            # surface hopping calculation
            # calculate overlap with the wavefunction from the previous
            # iteration
            # TODO: decoherence 
            program_state.nacmes.append(self._compute_nacme(program_state))
            print("NAC")
            print(program_state.nacmes[-1])
            print("rho")
            print(program_state.rho)
            print("state coeff")
            print(program_state.state_coefficients)

            self._propogate_electronic(program_state)
            hop = self._check_hop(program_state)
            if hop:
                self._decohere(program_state)
                self._scale_velocities(program_state)
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
                # state energies shouldn't change, but force and energy would
                forces, energy, _, _, _ = self._grab_data(force_log_file, program_state)
       
        else:
            program_state.nacmes.append(
                np.zeros(
                    (
                        program_state.number_of_electronic_states,
                        program_state.number_of_electronic_states
                    ), dtype=np.double)
            )

        program_state.forces[-1] = forces
        program_state.energies.append(energy)
        
        return True

    def _run_job(self, job_name, molecule, theory, retry=False, debug=False):
        """Call ORCA and return a string with the name of the log file."""
        job_inp_file = f"{job_name}.inp"
        job_out_file = f"{job_name}.out"
        self._prepare_inp_file(
            job_inp_file,
            molecule,
            theory,
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

        if retry:
            fr = FileReader(job_out_file, just_geom=False)
            if fr["error"]:
                print("encountered a gaussian error: %s\nattempting to resolve" % fr["error_msg"])
                new_theory = theory.copy()
                fix_attempted = False
                for job in theory.job_type:
                    try:
                        new_theory = job.resolve_error(
                            fr["error"], new_theory, "gaussian",
                        )
                        fix_attempted = True
                    except NotImplementedError:
                        pass
                
                if fix_attempted:
                    old_header = theory.make_header(
                        geom=molecule, style="orca"
                    )
                    new_header = new_theory.make_header(
                        geom=molecule, style="orca"
                    )
                    print("trying a potential fix")
                    print("============ OLD HEADER ============")
                    print(old_header)
                    print("============ NEW HEADER ============")
                    print(new_header)

                    print("%i retries will remain" % (retry - 1))

                    self._run_job(
                        job_name,
                        molecule,
                        new_theory,
                        retry=retry - 1,
                        debug=debug
                    )
                else:
                    print("error could not be resolved")

        return job_out_file

    @staticmethod
    def _prepare_inp_file(file_name, molecule, theory):
        """Prepare a .inp file for an ORCA run."""
        molecule.write(
            outfile=file_name,
            theory=theory,
            style="orca",
        )

    def _grab_data(self, out_file_name, program_state):
        """Parse forces into program_state from the given out file."""
        fr = FileReader(out_file_name, just_geom=False)
        if not fr["finished"]:
            raise exceptions.ElectronicStructureProgramError(
                "ORCA force calculation .out file was not valid. ORCA "
                "returned an error or could not be called correctly."
            )
        
        forces = containers.Forces()
        energy = containers.Energies()
        
        state_nrg = containers.Energies()
        state_nrg.append(fr["energy"], enums.EnergyUnits.HARTREE)
        # state energy is ground state energy + excitation energy
        for i_state in range(1, program_state.number_of_electronic_states):
            nrg = fr["energy"] + (
                JOULE_TO_HARTREE * ELECTRON_VOLT_TO_JOULE * \
                fr["uv_vis"].data[i_state - 1].excitation_energy
            )
        
            state_nrg.append(nrg, enums.EnergyUnits.HARTREE)

        energy.append(
            state_nrg.as_hartree()[program_state.current_electronic_state],
            enums.EnergyUnits.HARTREE
        )
        for v in fr["forces"]:
            forces.append(*v, enums.ForceUnits.HARTREE_PER_BOHR)

        if not self.basis:
            try:
                self.basis = fr["basis_set_by_ele"]
                self._setup_basis(program_state.molecule.elements)
            except KeyError:
                pass

        program_state.number_of_basis_functions = fr["n_basis"]
        ci_coefficients = ORCASurfaceHopHandler._read_ci_coefficients(
            program_state, out_file_name
        )
        mo_coefficients = np.array(fr["alpha_coefficients"])
        return forces, energy, state_nrg, mo_coefficients, ci_coefficients

    def _setup_basis(self, element_list):
        """sets up the basis set"""
        # number of pure and cartesian basis functions
        self.npao = 0
        self.ncao = 0
        self.cart_2_pure_map = []

        # shell type to number of functions and angular momentum
        type_to_nfunc_am = {
            "s": {"n_func":  1, "n_cart_func":  1, "l": 0},
            "p": {"n_func":  3, "n_cart_func":  3, "l": 1},
            "d": {"n_func":  5, "n_cart_func":  6, "l": 2},
            "f": {"n_func":  7, "n_cart_func": 10, "l": 3},
            "g": {"n_func":  9, "n_cart_func": 15, "l": 4},
            "h": {"n_func": 11, "n_cart_func": 21, "l": 5},
            "i": {"n_func": 13, "n_cart_func": 28, "l": 6},
        }
        self.shell_groups = []
        for i, element in enumerate(element_list):
            # basis information might be stored based on the index of the atom
            # if using a split basis where an element can have different
            # basis sets
            # otherwise, info is per-element
            try:
                shell = self.basis[i]
            except KeyError:
                shell = self.basis[element]
                
            for shell_type, n_prim, exponents, con_coef in shell:
                shell_type = shell_type.casefold()
                self.npao += type_to_nfunc_am[shell_type]["n_func"]
                self.ncao += type_to_nfunc_am[shell_type]["n_cart_func"]
                for n in range(0, type_to_nfunc_am[shell_type]["n_cart_func"]):
                    exponents = np.array(exponents)
                    l_total = type_to_nfunc_am[shell_type]["l"]
                    # angular momentum for x, y, and z (e.g., [2, 0, 0] for dx^2)
                    l_xyz = self._angular_momentum(l_total, n)
                    # basis set might not be normalized
                    # n is a normalization factor
                    n = np.sqrt((2 * exponents) ** (l_total + 3 / 2)) / (np.pi ** (3. / 4))
                    n *= np.sqrt(2 ** l_total / factorial2(2 * l_total - 1))
    
                    self.shells.append({
                        "type": shell_type,
                        "exp": exponents,
                        "coef": n * np.array(con_coef),
                        "nprim": n_prim,
                        "l": l_xyz,
                        "l_max": max(l_xyz),
                        "atom_ndx": i,
                    })
                
                self.shell_groups.append(shell_type)

    @staticmethod
    def _read_ci_coefficients(program_state, out_file_name):
        """read CI coefficients from the .cis file"""
        basename, ext = os.path.splitext(out_file_name)
        cis_file = basename + ".cis"
        
        # TODO: CI coefficients are printed to the output file in ORCA 5
        # use %tddft TPrint 0.0 
        """
        Read binary CI vector file from ORCA.
            Adapted from TheoDORE 1.7.1, Authors: S. Mai, F. Plasser
            https://sourceforge.net/p/theodore-qc
        """
        # actually adapted from https://github.com/eljost/pysisyphus b/c I
        # can't find where this is in the TheoDORE code
        cis_handle = open(cis_file, "rb")
    
        # the header consists of 9 4-byte integers, the first 5
        # of which give useful info.
        nvec  = struct.unpack('i', cis_handle.read(4))[0]
        # header array contains:
        # [0] index of first alpha occ,  is equal to number of frozen alphas
        # [1] index of last  alpha occ
        # [2] index of first alpha virt
        # [3] index of last  alpha virt, header[3]+1 is equal to number of bfs
        # [4] index of first beta  occ,  for restricted equal to -1
        # [5] index of last  beta  occ,  for restricted equal to -1
        # [6] index of first beta  virt, for restricted equal to -1
        # [7] index of last  beta  virt, for restricted equal to -1
        header = [struct.unpack('i', cis_handle.read(4))[0]
                for i in range(8)]
    
        if any([flag != -1 for flag in header[4:8]]):
            raise RuntimeError("_read_ci_coefficients: no support for unrestricted MOs")
    
        # TODO: beta?
        nfrzc = header[0]
        program_state.number_of_frozen_core = nfrzc
        nocc = header[1] + 1
        program_state.number_of_alpha_occupied = nocc
        nact = nocc - nfrzc
        nmo  = header[3] + 1
        nvir = nmo - header[2]
        program_state.number_of_alpha_virtual = nvir
        lenci = nact * nvir
    
        # Loop over states. For non-TDA order is: X+Y of 1, X-Y of 1,
        # X+Y of 2, X-Y of 2, ...
        prevroot = -1
        coeffs = [np.zeros((nocc, nvir))]
        for ivec in range(nvec):
            # header of each vector
            # contains 6 4-byte ints, then 1 8-byte double, then 8 byte unknown
            nele, d1, mult, d2, iroot, d3 = struct.unpack('iiiiii', cis_handle.read(24))
            ene,d3 = struct.unpack('dd', cis_handle.read(16))
            # then comes nact * nvirt 8-byte doubles with the coefficients
            coeff = struct.unpack(lenci * 'd', cis_handle.read(8 * lenci))
            coeff = np.array(coeff).reshape(-1, nvir)
            # create full array, i.e nocc x nvirt
            coeff_full = np.zeros((nocc, nvir))
            coeff_full[nfrzc:] = coeff
    
            # in this case, we have a non-TDA state!
            # and we need to compute (prevvector+currentvector)/2 = X vector
            if prevroot == iroot:
                x_plus_y = coeffs[-1]
                x_minus_y = coeff_full
                x = 0.5 * (x_plus_y + x_minus_y)
                coeffs[-1] = x
            else:
                coeffs.append(coeff_full)
    
            prevroot = iroot
        cis_handle.close()
        return np.array(coeffs)

    def _compute_overlaps(self, program_state):
        """
        compute atomic orbital overlap and overlap between wavefunctions
        from this iteration and the previous one
        """
        if not program_state.overlap_theory:
            overlap_theory = program_state.theory.copy()
            # XXX: do we need to account for center of mass motion/rotation?
            # print AO overlap matrix and don't do SCF
            # TODO: figure out how to ksip everything after the SCF
            overlap_theory.add_kwargs(
                blocks={
                    "output": ["print[P_Overlap] 1"],
                    "scf": ["MaxIter 0"],
                },
            )
            overlap_theory.job_type = "energy"
            # double the molecules, double the charge
            overlap_theory.charge *= 2
            program_state.overlap_theory = overlap_theory
        else:
            overlap_theory = program_state.overlap_theory

        if program_state.current_step > 0:
            # make a combined structure with the structure from this iteration and
            # the structure from the previous iteration
            prev_mol = program_state.molecule.copy()
            ao_overlap = self._calculate_ao_overlap(
                program_state.structures[-1].as_bohr(),
                program_state.structures[-2].as_bohr(),
            )
            program_state.atomic_orbital_overlap = ao_overlap

    def _calculate_ao_overlap(self, current_coordinates, previous_coordinates):
        """
        calculates atomic orbital overlap matrix between
        the two sets of coordinates
        """
        current_coordinates = np.array(current_coordinates)
        previous_coordinates = np.array(previous_coordinates)
        # calculating Sc is basically copy-pasted from CHEM 8950
        S = np.zeros((self.npao, self.npao))
        Sc = np.zeros((self.ncao, self.ncao))
        
        distances = distance_matrix(current_coordinates, previous_coordinates)
        
        # calculate cartesian overlap, Sc
        for i, shell_a in enumerate(self.shells):
            l_a = shell_a["l"]
            l_a_max = shell_a["l_max"]
            coords_a = current_coordinates[shell_a["atom_ndx"]]
            for j, shell_b in enumerate(self.shells[:i + 1]):
                l_b = shell_b["l"]
                l_b_max = shell_b["l_max"]
                r2 = distances[shell_a["atom_ndx"], shell_b["atom_ndx"]] ** 2
                coords_b = previous_coordinates[shell_b["atom_ndx"]]
                for k in range(0, shell_a["nprim"]):
                    for l in range(0, shell_b["nprim"]):
                        alpha = shell_a["exp"][k] + shell_b["exp"][l]
    
                        P_x = self._P(
                            coords_a, coords_b,
                            shell_a["exp"][k], shell_b["exp"][l],
                        )
                        Pa = P_x - coords_a
                        Pb = P_x - coords_b
    
                        x, y, z = self._os_recusion(
                            Pa, Pb,
                            alpha, 
                            max(l_a), max(l_b),
                        )
                        ex = shell_a["exp"][k] * shell_b["exp"][l] / alpha
                        ss = (np.pi / alpha) ** (3. / 2)
                        ss *= np.exp(-ex * r2)
                        ca = shell_a["coef"][k]
                        cb = shell_b["coef"][l]
                        norm = ss * ca * cb
                        sq = 1
                        for o, q in enumerate([x, y, z]):
                            sq *= q[l_a[o], l_b[o]]
                        Sc[i, j] += norm * sq
                        Sc[j, i] = Sc[i, j]
       
        # print("Sc", Sc.shape)
        # print(Sc)
        # map to pure
        ca = 0
        sa = 0
        for type_a in self.shell_groups:
            map_a = self.map_cart_2_pure[type_a]
            spherical_a, cartesian_a = map_a.shape
            cb = 0
            sb = 0
            for type_b in self.shell_groups:
                map_b = self.map_cart_2_pure[type_b]
                spherical_b, cartesian_b = map_b.shape
                # print("A", map_a.shape, shell_a["type"])
                # print("B", map_b.shape, shell_b["type"])
                # print("Sc", Sc[ca:ca + cartesian_a, cb:cb + cartesian_b])
                # print("ndx a", ca, ca + cartesian_a, list(range(ca, ca + cartesian_a)))
                # print("ndx b", cb, cb + cartesian_b, list(range(cb, cb + cartesian_b)))
                b_mapped = np.dot(
                    Sc[ca:ca + cartesian_a, cb:cb + cartesian_b], map_b.T
                )
                ab_mapped = np.dot(map_a, b_mapped)
                S[sa:sa + spherical_a, sb:sb + spherical_b] = ab_mapped
                cb += cartesian_b
                sb += spherical_b
            ca += cartesian_a
            sa += spherical_a
        
        return S
    
    @staticmethod
    def _P(A, B, exp_a, exp_b):
        """calculate P, a term used for OS recurrence"""
        return (exp_a * A + exp_b * B) / (exp_a + exp_b)
    
    @staticmethod
    def _angular_momentum(mam, relative_ndx):
        """determines x y and z angular momentum for the ith basis function"""    
        #determine x y and z am in the same order as psi4
        k = 0
        for x in range(mam, -1, -1):
            for y in range(mam-x, -1, -1):
                z = mam - x - y
                if k == relative_ndx:
                    return np.array([x, y, z])
                else:
                    k += 1

    @staticmethod
    def _os_recusion(PA, PB, alpha, AMa, AMb):
        """
        Obara-Saika recurrence relationship to calculate overlap
        of two AOs (a|b)
        results will need to be normalized
        """
        # basically copy-pasted from an assignment for Dr. Turney's CHEM 8950
        # the recurrence relationship is for cartesian GTOs (e.g., 6d)
        overlap = np.zeros((3, AMa + 1, AMb + 1))
        overlap[:, 0, 0] = 1
        
        for k in range(0, 3):
            for i in range(0, AMa + 1):
                for j in range(0, AMb + 1):
                    if j > 0:
                        overlap[k, i, j] = PB[k] * overlap[k, i, j - 1]
                        if i > 0:
                            overlap[k, i, j] += (i /(2 * alpha)) * overlap[k, i - 1, j - 1]
                        if j > 1:
                            overlap[k, i, j] += (j - 1)/(2 * alpha) * overlap[k, i, j - 2]
                    elif i > 0:
                        overlap[k, i, j] = PA[k] * overlap[k, i - 1, j]
                        if i > 1:
                            overlap[k, i, j] += (i - 1)/(2 * alpha) * overlap[k, i - 2, j]
            
        return overlap   

