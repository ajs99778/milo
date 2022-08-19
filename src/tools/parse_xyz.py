#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creates a .xyz file for each .out file in the current directory."""

import os
from glob import glob
import argparse




def main(argv):
    """Serve as main."""

    parser = argparse.ArgumentParser(
        description="turn Milo output into XYZ trajectory files",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "infile", metavar="Milo output",
        type=str,
        nargs="+",
        help="a Milo output file",
    )
    
    args = parser.parse_args(argv)
    out_files = []
    for f in args.infile:
        out_files.extend(glob(f))

    for out_file in out_files:
        with open(out_file, mode="r") as out_reader:
            num_atoms = None
            final_xyz_lines = list()
            current_xyz = list()

            in_coordinates_section = False
            for line in out_reader:
                if "  Coordinates:" in line:
                    in_coordinates_section = True
                elif "  SCF Energy:" in line or "Normal termination." in line:
                    if num_atoms is None:
                        num_atoms = str(len(current_xyz))
                    final_xyz_lines.append(num_atoms)
                    final_xyz_lines.append("")
                    final_xyz_lines.extend(current_xyz)
                    current_xyz = list()
                    in_coordinates_section = False
                elif in_coordinates_section and line.strip() != "":
                    current_xyz.append(line.strip())

        with open(out_file[:-4] + ".xyz", mode="w") as xyz_writer:
            for line in final_xyz_lines:
                xyz_writer.write(line + "\n")


if __name__ == "__main__":
    main(None)
