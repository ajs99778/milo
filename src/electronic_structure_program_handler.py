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
            prev_mol.coords = program_state.structures[-1].as_angstrom()
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



class ORCASurfaceHopeHandler(NumericalNonAdiabaticSurfaceHopHandler):
    """handler for ORCA with Unix-MD surface hopping algorithms"""

    def generate_forces(self, program_state):
        """Preform computation and append forces to list in program state."""
        """
        Preform computation and append forces to list in program state.
        also determines if the state should change using Unix-MD
        """

        print("current state:", program_state.current_electronic_state)

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
        # run force and excited state computation
        force_out_file = self._run_job(
            f"_{program_state.current_step}_force",
            program_state.molecule,
            program_state.theory,
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
                np.zeros(
                    (
                        program_state.number_of_electronic_states,
                        program_state.number_of_electronic_states
                    ), dtype=np.double)
            )

        program_state.forces[-1] = forces
        program_state.energies.append(energy)
        
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

    def _grab_data(out_file_name, program_state):
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

        program_state.previous_ci_coefficients = program_state.current_ci_coefficients
        current_ci_coefficients = self._read_ci_coefficients(out_file_name)
        program_state.previous_mo_coefficients = program_state.current_mo_coefficients
        program_state.current_mo_coefficients = np.array(fr["alpha_coefficients"])
        program_state.number_of_basis_functions = fr["n_basis"]

    def _read_ci_coefficients(self, program_state, out_file_name):
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
        coeffs = [np.zeros(nocc, nvir)]
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
        program_state.current_ci_coefficients = np.array(coeff)

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
            prev_mol.coords = program_state.structures[-1].as_angstrom()
            overlap_mol = Molecule([*program_state.molecule.atoms, *prev_mol.atoms])
            job_name = f"_{program_state.current_step}_overlap"
            overlap_rwf = self._run_job(
                job_name,
                overlap_mol,
                overlap_theory,
            )
            ao_overlap = self._read_ao_overlap(job_name + ".out")
            # we only care about the overlap between the current AOs with the previous AOs
            # the upper left quadrant is the overlap of the current iteration overlap only
            # the lower right quadrant is the previous iteration overlap only
            dim = program_state.number_of_basis_functions
            program_state.atomic_orbital_overlap = ao_overlap[:dim, dim:]

    def _read_ao_overlap(self, out_file):
        """reads atomic orbital overlap matrix"""
        fr = FileReader(out_file, just_geom=False)
        ao_overlap = fr["ao_overlap"]
        return ao_overlap




