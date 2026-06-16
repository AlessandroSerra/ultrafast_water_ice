from ase.io import read, write

def convert_xyz(filepath, outfile = None, header = None):
    atoms = read(f'{filepath}')

    if header is None:
        comment = "pbc=\"T T T\" Lattice=\"31.0 0 0 0 31.0 0 0 0 31.0\" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\""

    else: 
        comment = header

    atomic_masses = {1: 1.008, 8: 15.999}

    # aggiungo la massa come proprietà per atomo
    masses = atoms.get_masses()
    atoms.set_masses(masses)

    # aggiungo velocità iniziali (esempio: zero)
    import numpy as np
    velocities = np.zeros((len(atoms), 3))
    atoms.set_velocities(velocities)

    if outfile is None:
        # riscrivo file con nuovo header
        write('water_init.xyz', atoms, comment=comment)
    else:
        write(f'{outfile}', atoms, comment=comment)
