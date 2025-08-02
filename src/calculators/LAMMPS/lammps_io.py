import os
import glob
from ase.data import atomic_numbers, atomic_masses


class Lammps_MD():
    """LAMMPS input generator."""

    def __init__(self, temp, symb_conv, shells, charges, ip_file):
        self.temp = temp
        self.shells = shells  # This will be a dictionary which assigns for element i shell number j and bondtpye k
        self.symb_conv = symb_conv
        self.symb_conv_reverse = {v[0]: k for k, v in symb_conv.items()}
        self.charges = charges  # Dictionary of symbol or number and charge
        self.run_command = "lmp_icc_openmpi -sf gpu -in in.RDX"  # "lmp_icc_openmpi -sf gpu -in in.RDX"
        f = open(ip_file, 'r')
        self.ip_def = f.readlines()

    def run_collision(self, particle, num_at_seed, max_dist):
        # Writing coordinate file
        coordinate_file = open('coordinates.lmpdata', 'w')
        # Check connectivity and convert atom symbols to

        s = "\n"
        atom_number = 0
        num_cs_atoms = 0
        cs_bonds = []
        molecule_num = 1
        seed_pos = []
        # Loop to write all atoms and bonds. It will be a bit problematic to incorporate Hydroxyls.
        for atom in particle:
            atom_number += 1
            if molecule_num == 2:
                seed_pos.append([atom_number, atom.position])

            atom_type = self.symb_conv[atom.symbol][0]
            s += "{0} {1} {2} {3:8.6f} {4:8.6f} {5:8.6f} {6:8.6f}\n".format(atom_number,
                                                                            molecule_num,
                                                                            atom_type,
                                                                            self.charges[atom_type],
                                                                            *atom.position
                                                                            )
            if atom.symbol in self.shells:
                atom_number += 1
                atom_type = self.symb_conv[atom.symbol][1]

                if molecule_num == 2:
                    seed_pos.append([atom_number, atom.position])
                num_cs_atoms += 1  # This might be replaced by a len(core_shell_bonds)?
                s += "{0} {1} {2} {3:8.6f} {4:8.6f} {5:8.6f} {6:8.6f}\n".format(atom_number,
                                                                                molecule_num,
                                                                                atom_type,
                                                                                self.charges[atom_type],
                                                                                *atom.position
                                                                                )
                cs_bonds.append([atom_number-1,
                                 atom_number,
                                 1])  # need to be changed so that other core-shells are available

            # Change molecule_num
            num_elements_seed = num_at_seed + num_cs_atoms
            if atom_number == num_at_seed + num_cs_atoms:
                molecule_num += 1

        coordinate_file.write("Nucleation running\n\n")
        coordinate_file.write("{} atoms\n".format(atom_number))
        coordinate_file.write("{} bonds\n".format(num_cs_atoms))
        coordinate_file.write("""0 angles
0 dihedrals
""")
        coordinate_file.write("{} atom types\n".format(
            len(set(particle.get_chemical_symbols()))+len(self.shells)))
        coordinate_file.write("{} bond types\n".format(1))
        coordinate_file.write("""0 angle types
0 dihedral types

-100.0 100.0 xlo xhi
-100.0 100.0 ylo yhi
-100.0 100.0 zlo zhi


Masses

""")
        for el in self.symb_conv.keys():  # This needs to be made sure who passes this info
            mass = atomic_masses[atomic_numbers[el]]
            if el in self.shells:
                mass_s = mass*0.1
                mass = mass*0.9
                coordinate_file.write("{} {} # {}\n".format(
                    self.symb_conv[el][0], mass, el))
                coordinate_file.write("{} {} # {}\n".format(
                    self.symb_conv[el][1], mass_s, "{} shell".format(el)))
            else:
                coordinate_file.write("{} {} # {}\n".format(
                    self.symb_conv[el][0], mass, el))

        coordinate_file.write("\nAtoms\n")
        coordinate_file.write(s)
        coordinate_file.write("""
Bonds

""")

        for i in range(num_cs_atoms):
            coordinate_file.write('{} {} {} {}\n'.format(i+1,
                                                         cs_bonds[i][2],
                                                         cs_bonds[i][0],
                                                         cs_bonds[i][1]))

        coordinate_file.write("""
Velocities

""")
        atom_num = 0
        s = ''
        for atom, vxyz in zip(particle.get_chemical_symbols(),
                              particle.get_velocities()):
            atom_num += 1
            s += "{0} {1:8.6f} {2:8.6f} {3:8.6f}\n".format(atom_num,
                                                           *vxyz)
            if atom in self.shells:
                atom_num += 1
                s += "{0} {1:8.6f} {2:8.6f} {3:8.6f}\n".format(atom_num,
                                                               *vxyz)
        coordinate_file.write(s)
        coordinate_file.close()

        configuration_file = open("in.RDX", 'w')
        configuration_file.write("""# ------------- INITIALIZATION
units           metal # m=grams/mo, E=eV, T=K, etc
atom_style      full # sistema atomic amb carregues

# -------------- ATOM DEFINITION
read_data       coordinates.lmpdata # coordenades dels atoms
group mainparticle molecule 1
group seedparticle molecule 2

kspace_style ewald 1.0e-6
\n""")
        for i in self.ip_def:
            configuration_file.write(i)

        cores = ''
        shells = ''
        for i in self.shells:
            cores += '{}'.format(self.symb_conv[i][0])
            shells += '{}'.format(self.symb_conv[i][1])

        configuration_file.write("group cores type {}\n".format(cores))
        configuration_file.write("group shells type {}\n\n".format(shells))
        configuration_file.write("""comm_modify vel yes
compute core_shells all temp/cs cores shells

# -------------- FF
neighbor        2 bin #skin (dist extra per les llistes) bin= tipo de construccio)
neigh_modify    one 5000
neigh_modify    every 5 delay 0 check yes #freq d'actualitzacio de les llistes
thermo_style custom step temp etotal ke pe
thermo 100
""")
        configuration_file.write(
            "velocity mainparticle create {} 123456789\n".format(self.temp))
        configuration_file.write("# -------------- MD RUN\n")
        configuration_file.write("fix 6 seedparticle recenter {a} {a} {a}\n".format(a=max_dist*2+5))

        configuration_file.write("""fix 1 all box/relax iso 0.0 vmax 0.001
unfix           1
fix             1 mainparticle nve
        """)
        configuration_file.write("""fix             3 mainparticle temp/berendsen {} {} 0.025
""".format(self.temp - 100, self.temp))
        configuration_file.write("""timestep 0.0005
dump            50 all xyz 10 preheat.xyz
restart         500 preheat.restart
run             5000
undump          50
unfix           1
unfix           3
unfix           6
""")
        for inc_at in seed_pos:
            configuration_file.write(
                "set atom {} x {} y {} z {}\n".format(inc_at[0],
                                                      *inc_at[1]))
        configuration_file.write("\nfix             1 all nve\n")
        configuration_file.write('timestep 0.0005\n')
        configuration_file.write("""dump            50 all xyz 10 nucleation.xyz
restart         500 nucleation.restart
run             3000
undump          50
unfix           1

dump  1 all xyz 1 final_raw.xyz
run      0
undump 1

        """)
        configuration_file.close()
        os.system(self.run_command)

    def read_final_structure(self):
        """Convert raw xyz file to readable file."""
        from ase.io import write
        from particle import Particle
        input_file = open('final_raw.xyz', 'r')
        lines = input_file.readlines()
        atom_symbols = []
        atom_positions = []
        num_atoms = int(lines[0].split()[0])
        for i in range(2, len(lines)):
            element_num = int(lines[i].split()[0])
            if element_num in self.symb_conv_reverse.keys():
                atom_symbols.append(self.symb_conv_reverse[element_num])
                atom_positions.append([lines[i].split()[1],
                                       lines[i].split()[2],
                                       lines[i].split()[3]])
        os.remove('final_raw.xyz')

        final_p = Particle(symbols=atom_symbols, positions=atom_positions)
        write('final.xyz', final_p)
        return final_p

    def clean_md(self, step, added):
        """Clean the results of the MD simulation."""

        if not os.path.exists('run{0}'.format(step)):
            os.makedirs('run{0}'.format(step))

            os.rename('log.lammps', 'run{0}/log.lammps'.format(step))
            os.rename('nucleation.xyz', 'run{0}/nucleation{0}.xyz'.format(step))
            os.rename('coordinates.lmpdata', 'run{0}/coordinates.lmpdata'.format(step))
            os.rename('in.RDX', 'run{0}/in.RDX'.format(step))
            os.rename('preheat.xyz', 'run{0}/preheat{0}.xyz'.format(step))
            os.rename('final.xyz', 'run{0}/final.xyz'.format(step))
            for i in glob.glob('*restart*'):
                os.remove(i)
