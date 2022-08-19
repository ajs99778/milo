#!/bin/env python

import argparse
import os

from AaronTools.const import UNIT
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.job_control import SubmitProcess
from AaronTools.theory import *
from AaronTools.utils.utils import combine_dicts

oscillator_types = ["classical", "quasiclassical"]
geometry_displacement = ["off", "egde_weighted", "gaussian", "uniform"]
integration_algorithm = ["verlet", "velocity_verlet"]
programs = ["gaussian09", "gaussian16"]

def abbreviated(arg, options, label, map_values=None):
    out = [x for x in options if x.startswith(arg.lower())]
    if len(out) == 1:
        if map_values:
            return map_values[out[0]]
        return out[0]
    elif not out:
        raise TypeError("%s must be one of %s" % (label, ", ".join(options)))
    raise TypeError("%s is ambiguous; could be %s" % (arg, ", ".join(out)))

setup_parser = argparse.ArgumentParser(
    description="setup Milo input files",
    formatter_class=argparse.RawTextHelpFormatter,
)

setup_parser.add_argument(
    "infile", metavar="frequency file",
    type=str,
    help="frequency computation with HPModes"
)

setup_parser.add_argument(
    "-n", "--trajectories",
    type=int,
    default=1,
    help="number of Milo jobs to set up (Default: 1)",
)

submit_options = setup_parser.add_argument_group("Submission options")
submit_options.add_argument(
    "-rj", "--run-jobs",
    action="store_true",
    default=False,
    dest="submit",
    help="submit jobs as they are set up",
)

submit_options.add_argument(
    "-jt", "--job-template",
    type=str,
    default="Milo_template.txt",
    dest="template",
    help="job submission template (Default: $AARONLIB/Milo_template.txt)",
)

job_options = setup_parser.add_argument_group("Job options")
job_options.add_argument(
    "-dt", "--time-step",
    type=float,
    dest="step_size",
    help="time step in femtoseconds",
)

job_options.add_argument(
    "-s", "--max-steps",
    type=int,
    dest="max_steps",
    help="number of MD iterations",
)

job_options.add_argument(
    "-t", "--temperature",
    type=float,
    dest="temperature",
    help="temperature in K for initial velocity sampling",
)

job_options.add_argument(
    "-ot", "--oscillator-type",
    type=lambda x, options=oscillator_types, label="oscillator type": abbreviated(x, options, label),
    dest="oscillator_type",
    choices=oscillator_types,
    help="oscillator type",
)

job_options.add_argument(
    "-dx", "--geometry-displacement",
    type=lambda x, options=geometry_displacement, label="geometry displacement": abbreviated(x, options, label),
    dest="geometry_displacement",
    choices=geometry_displacement,
    help="initial geometry displacement",
)

job_options.add_argument(
    "-i", "--integration-algorithm",
    type=lambda x, options=integration_algorithm, label="integration algorithm": abbreviated(x, options, label),
    choices=integration_algorithm,
    help="integration algorithm",
)

job_options.add_argument(
    "-re", "--rotational-energy",
    action="store_true",
    default=False,
    dest="rotational_energy",
    help="add rotational energy",
)

job_options.add_argument(
    "-eb", "--energy-boost",
    metavar=("min", "max"),
    nargs=2,
    type=float,
    dest="energy_boost",
    help="boost energy until the kinetic energy is between min and max (kcal/mol)"
)

job_options.add_argument(
    "-ff", "--fixed-forward",
    default=[],
    action="append",
    type=int,
    metavar="N",
    dest="fixed_forward",
    help="set the direction of mode to along the normal mode displacement",
)

job_options.add_argument(
    "-fr", "--fixed-reverse",
    default=[],
    action="append",
    type=int,
    metavar="N",
    dest="fixed_reverse",
    help="set the direction of mode N to opposite the normal mode displacement",
)

phase = job_options.add_mutually_exclusive_group(required=False)
phase.add_argument(
    "-bt", "--bring-together",
    metavar=("A", "B"),
    nargs=2,
    type=int,
    dest="bring_together",
    help="set direction of imaginary mode to bring atoms A and B together",
)

phase.add_argument(
    "-pa", "--push-apart",
    metavar=("A", "B"),
    nargs=2,
    type=int,
    dest="push_apart",
    help="set direction of imaginary mode to push atoms A and B apart",
)

job_options.add_argument(
    "-r", "--random-seed",
    type=int,
    dest="random_seed",
    help="seed used for classical and quasi-classical sampling",
)

job_options.add_argument(
    "-p", "--program",
    type=lambda x, options=programs, label="program": abbreviated(x, options, label),
    dest="program",
    choices=programs,
    default="gaussian16",
    help="program for computing forces",
)

job_options.add_argument(
    "-np", "--processors",
    type=int,
    dest="processors",
    help="number of processors to use when computing forces",
)

job_options.add_argument(
    "-mem", "--memory",
    type=int,
    dest="memory",
    help="memory in GB to use when computing forces",
)

job_options.add_argument(
    "-q", "--charge",
    type=int,
    dest="charge",
    help="net charge of the system",
)

job_options.add_argument(
    "-mult", "--multiplicity",
    type=int,
    dest="multiplicity",
    help="electronic multiplicity of the system",
)

theory_options = setup_parser.add_argument_group("Model chemistry options")
theory_options.add_argument(
    "-m", "--method",
    dest="method",
    help="method (i.e. DFT functional) - Defaults to last-used",
)

theory_options.add_argument(
    "-b", "--basis",
    action="append",
    nargs="+",
    dest="basis",
    help="basis set or list of elements and basis set (e.g. C O N aug-cc-pvtz)\n"
    "elements can be prefixed with ! to exclude them from the basis\n"
    "tm is a synonym for d-block elements\n"
    "auxilliary basis sets can be specified by putting aux X before the basis\n"
    "set name, where X is the auxilliary type (e.g. aux JK cc-pVDZ for cc-pVDZ/JK)\n"
    "a path to a file containing a basis set definition (like one\n"
    "downloaded from basissetexchange.org) can be placed after the\n"
    "basis set name",
)

theory_options.add_argument(
    "-ecp", "--pseudopotential",
    nargs="+",
    action="append",
    dest="ecp",
    help="ECP or list of elements and ECP (e.g. Pt LANL2DZ)\n"
    "elements can be prefixed with ! to exclude them from the ECP\n"
    "tm is a synonym for d-block elements\n"
    "a path to a file containing a basis set definition (like one\n"
    "downloaded from basissetexchange.org) can be placed after the\n"
    "basis set name",
)

theory_options.add_argument(
    "-ed", "--empirical-dispersion",
    dest="empirical_dispersion",
    help="empirical dispersion keyword",
)

theory_options.add_argument(
    "-sv", "--solvent",
    nargs=2,
    metavar=("model", "type"),
    dest="solvent",
    help="implicit solvent model and type (e.g., -sv SMD benzene)",
)

theory_options.add_argument(
    "-g", "--grid",
    dest="grid",
    help="DFT integration grid",
)

theory_options.add_argument(
    "--route",
    dest="route",
    action="append",
    nargs="+",
    metavar=("keyword", "option"),
    default=[],
    help="Gaussian route options (e.g. --route DensityFit)",
)


theory_options.add_argument(
    "--link0",
    dest="link0",
    default=[],
    action="append",
    nargs="+",
    metavar=("command", "value"),
    help="Link 0 commands (without %%)",
)

theory_options.add_argument(
    "--end-of-file",
    default=[],
    action="append",
    dest="end_of_file",
    metavar="input",
    help="lines to add at the end of the file (e.g. for NBORead)",
)




args = setup_parser.parse_args()

fr = FileReader(args.infile, just_geom=False)
geom = Geometry(fr)
try:
    theory = fr["theory"]
except KeyError:
    geom.LOG.warning("theory was not parsed from input file")
    theory = Theory()

# check for hpmodes
new_kwargs = {"route": dict()}
if GAUSSIAN_ROUTE in theory.kwargs:
    for key, value in theory.kwargs["route"].items():
        if not any(key.lower() == x for x in ["opt", "force"]):
            new_kwargs[key] = theory.kwargs["route"][key]

theory.kwargs = new_kwargs

if fr["file_type"] == "log" and not fr["hpmodes"]:
    raise TypeError("input file does not have high-precision normal modes (Freq=(HPModes))")

theory.geometry = geom

# remove job types we don't want
theory.job_type = None

kwargs = dict()
for position in ["route", "link0"]:
    options = getattr(args, position)
    if options:
        kwargs[position] = dict()

        for option in options:
            setting = option.pop(0)
            try:
                kwargs[position][setting].extend(option)
            except KeyError:
                kwargs[position][setting] = option

for position in ["end_of_file"]:
    option = getattr(args, position)
    if option:
        if option not in kwargs:
            kwargs[position] = []

        kwargs[position].extend([" ".join(word) for word in option])

theory.kwargs = combine_dicts(theory.kwargs, kwargs)

if args.method is not None:
    theory.method = args.method

if args.basis:
    basis_sets = []
    for basis in args.basis:
        basis_sets.append(
            BasisSet.parse_basis_str(" ".join(basis))[0]
        )

    theory.basis = BasisSet(basis_sets)

if args.ecp:
    ecps = []
    for ecp in args.ecp:
        ecps.append(
            BasisSet.parse_basis_str(" ".join(ecp))[0]
        )

    theory.basis.ecp = ecps

if theory.basis:
    theory.basis.refresh_elements(geom)

if args.charge is not None:
    theory.charge = args.charge

if args.multiplicity is not None:
    theory.multiplicity = args.multiplicity

# start making Milo input file
output = "$job\n"

# add route
header = theory.make_header(style="gaussian")
route = [x for x in header.splitlines() if x.startswith("#")][0]
route = route.lstrip("#n").strip()
output += "    %-25s    %-s\n" % ("gaussian_header", route)

# floating point options
for option in ["step_size", "temperature"]:
    value = getattr(args, option)
    if value is not None:
        output += "    %-25s    %-.2f\n" % (option, value)

# string options:
for option in ["program", "geometry_displacement", "integration_algorithm", "oscillator_type"]:
    value = getattr(args, option)
    if value is not None:
        output += "    %-25s    %-s\n" % (option, value)

#integer options
for option in ["max_steps", "memory", "processors", "random_seed"]:
    value = getattr(args, option)
    if value is not None:
        output += "    %-25s    %-s\n" % (option, value)

if args.bring_together:
    output += "    %-25s    bring_together %i %i\n" % ("phase", *args.bring_together)

if args.push_apart:
    output += "    %-25s    push_apart %i %i\n" % ("phase", *args.push_apart)

if args.fixed_forward:
    output += "    %-25s    %-3i  1\n" % ("fixed_mode_direction", *args.fixed_forward)

if args.fixed_reverse:
    output += "    %-25s    %-3i -1\n" % ("fixed_mode_direction", *args.fixed_reverse)


if args.energy_boost:
    output += "    %-25s    on %.2f %.2f\n" % ("energy_boost", *args.energy_boost)

if args.rotational_energy:
    output += "    %-25s    on\n" % "rotational_energy"

output += "$end\n\n"

footer = theory.get_gaussian_footer()
if footer.strip():
    output += "$gaussian_footer\n%s\n$end" % footer.lstrip()
    output += "\n\n"

output += "$molecule\n"
output += "%2i %i\n" % (theory.charge, theory.multiplicity)
for atom in geom:
    output += "%2s  %10.6f   %10.6f   %10.6f\n" % (atom.element, *atom.coords)
output += "$end\n\n"

output += "$isotope\n"
for i, atom in enumerate(geom.atoms):
    output += "    %3i   %10.6f\n" % (i + 1, atom.mass)
output += "$end\n\n"

output += "$frequency_data\n"
for mode in fr["frequency"].data:
    output += "    %.4f  " % mode.frequency
    if mode.frequency < 0:
        if not args.bring_together and not args.push_apart:
            geom.LOG.warning("there is an imaginary mode, but --bring-together and --push-apart were not used")
    output += "%.4f  " % mode.red_mass
    output += "%.4f" % (10000 * mode.forcek)
    for displacement in mode.vector:
        output += "  %.5f  %.5f  %.5f" % tuple(displacement)

    output += "\n"
output += "$end\n\n"


for i in range(1, args.trajectories + 1):
    os.mkdir("traj_%03i" % i)
    fname = "traj_%03i/traj_%03i.in" % (i, i)
    with open(fname, "w") as f:
        f.write(output)

        if args.submit:
            submit_process = SubmitProcess(
                fname, 72, args.processors, args.memory + 1, template=args.template
            )
            submit_process.submit()
