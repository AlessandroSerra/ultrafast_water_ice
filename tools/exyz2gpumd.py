from argparse import ArgumentParser, Namespace

import numpy as np

TYPESD = {"O": 1, "H": 2}


def read_exyz(filename):
    cell_vectors = np.zeros((3, 3), dtype=np.float32)

    with open(filename) as f:
        lines = f.readlines()

        # first 2 lines are comments
        Natoms = int(lines[2])
        types = np.zeros(Natoms, dtype=np.int8)
        poss = np.zeros((Natoms, 3), dtype=np.float32)

        # 3rd line is a comment
        for i, line in enumerate(lines[4 : (Natoms + 4)]):
            sline = line.split()
            types[i] = TYPESD[sline[0]]
            poss[i] = float(sline[1]), float(sline[2]), float(sline[3])

        # last line isnt needed
        cell_vectors[0, :] = list(map(float, lines[-4].split()[1:4]))
        cell_vectors[1, :] = list(map(float, lines[-3].split()[1:4]))
        cell_vectors[2, :] = list(map(float, lines[-2].split()[1:4]))

    return poss, types, cell_vectors


# pbc="T T T" Lattice="31.0 0 0 0 31.0 0 0 0 31.0" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:unwrappe
# d_position:R:3


def write_gpumd(filename, poss, types, cell_vectors):
    Natoms = poss.shape[0]
    with open(filename, "w") as f:
        f.write(f"{Natoms}\n")
        f.write('pbc="T T T" ')
        f.write(
            f'Lattice="{cell_vectors[0, 0]} {cell_vectors[0, 1]} {cell_vectors[0, 2]} {cell_vectors[1, 0]} {cell_vectors[1, 1]} {cell_vectors[1, 2]} {cell_vectors[2, 0]} {cell_vectors[2, 1]} {cell_vectors[2, 2]}" '
        )
        f.write("Properties=species:S:1:pos:R:3:mass:R:1\n")

        for i in range(Natoms):
            species = "O" if types[i] == 1 else "H"
            mass = 15.999 if types[i] == 1 else 1.008
            f.write(f"{species} {poss[i, 0]} {poss[i, 1]} {poss[i, 2]} {mass}\n")


def main():
    parser = ArgumentParser(
        prog="wigner.py",
        description="Script to excite water molecules in a XYZ dump file using Wigner Sampling.",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input Extended XYZ file (e.g., 'dump.exyz').",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="converted.xyz",
        help="Output GPUMD file (default: 'converted.xyz').",
    )

    args: Namespace = parser.parse_args()

    poss, types, cell_vectors = read_exyz(args.input_file)
    write_gpumd(args.output_file, poss, types, cell_vectors)
    print(f"Converted {args.input_file} to {args.output_file}.")


if __name__ == "__main__":
    main()
