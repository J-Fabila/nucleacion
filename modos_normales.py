import re
from ase import Atoms
from ase.io import read, write
import numpy as np
import os

import numpy as np

def read_xyz_with_modes(filename, n_conf):
    with open(filename, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    block_size = natoms + 2
    start = n_conf * block_size
    end = start + block_size
    header_line = lines[start + 1].strip()
    freq_match = re.search(r"frequency at\s+([-\d\.Ee\+]+)", header_line)
    freq = float(freq_match.group(1))
    block = lines[start + 2:end]  # saltar encabezado y comentario
    symbols = []
    data_matrix = []
    for line in block:
        parts = line.split()
        symbols.append(parts[0])
        data = np.array(parts[1:], dtype=float)
        data_matrix.append(data)
    data_matrix = np.array(data_matrix)
    positions = data_matrix[:, 0:3]
    nmodes = (data_matrix.shape[1] // 3) - 1 if data_matrix.shape[1] > 3 else 0
    modes = np.array([data_matrix[:, 3 + 3*i : 6 + 3*i] for i in range(nmodes)])
    return symbols, positions, modes, freq

symbols, positions, modes, freq = read_xyz_with_modes("TiC_32_4x4x4_-4C.xyz",)
generate_displaced_structures(symbols, positions, modes, delta=4)

def generate_displaced_structures(symbols, positions, modes, delta=0.1, outdir='displacements'):
    print("POSI", positions)
    print("MODES", modes)
    # 60,3 ; 1,60,3
    norms = np.linalg.norm(modes, axis=-1, keepdims=True)
    modes = modes / norms
    os.makedirs(outdir, exist_ok=True)
    displaced_positions = positions + modes[0,:,:] * delta
    atoms = Atoms(symbols=symbols, positions=displaced_positions)
    filename = os.path.join(outdir, f"mode_.xyz")
    write(filename, atoms)
    print(f"Guardado: {filename}")


if __name__ == "__main__":
    xyz_file = "molecule_modes.xyz"
    symbols, positions, modes, freq = read_xyz_with_modes(xyz_file)
    generate_displaced_structures(symbols, positions, modes, delta=0.1)
