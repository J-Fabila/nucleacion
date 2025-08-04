import sys
import yaml
import time as tm

from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from src.utils import get_model_system
#from mace.calculators import MACECalculator
from src.calculators.LAMMPS.lammps_io import Lammps_MD
from src.calculators.MACE.MACE_MD import MACE_collision
from src.particle import Particle, Incoming_Part_Library

start_time = tm.time()
kb = 1.3806e-23

conf_file = 'conf.yaml'
if len(sys.argv) > 1:
    conf_file = sys.argv[1]

with open(conf_file, "r") as f:
    config = yaml.safe_load(f)

time = 0.0

seed = read(config["input_file"])
prev_charge = seed.info['charge']
particle = Particle(symbols=seed.get_chemical_symbols(),
                    positions=seed.get_positions())

MaxwellBoltzmannDistribution(particle, temperature_K=config["temperature"])

if "desired_stoichiometry" in config.keys():
    inc_part_libr = Incoming_Part_Library(config["condensation_species"], config["desired_stoichiometry"])

add_step = 1
temp = config["temperature"]
cols_per_second = 1
non_attached_steps = 0

if config["calculator"].upper() == "LAMMPS":
    calculator = Lammps_MD(temp,
                       symb_conv=config["elements"],
                       shells=config["shells"],
                       charges=config["charges"],
                       ip_file=config["ip_definition"])  # check

log_file = open('results.log', 'w')
log_file.write('Step Seed.Chem.Formula Temp den(ncm-3) P(Pa) cols.per.second position added atom_mov\n')

for i in range(1, config["cycles"]+1):
    # Returns max distance, although it has all of them
    particle_size = particle.get_maximum_distances()
    num_atoms_seed = len(particle)
    # incoming particle to be closer to the surface
    temp_ = 10
    if config["calculator"].upper() == "LAMMPS":
        temp_ = temp
    incoming_particle = inc_part_libr.generate_inc(temp_,
                                                   particle_size,
                                                   prev_charge,
                                                   current_stoichiometry=particle.get_chemical_formula())

    model_system, atom_movement = get_model_system(particle, incoming_particle)


# Check if the incoming particle is too close to the main particle and move it  
# FER UNA FUNCIÓ A PART PER A QUE QUEDI TOT MES RECOLLIT. HAURIA DE LLEGIR LES VARIABLES PARTICLE I INCOMING PARTICLE I RETORNAR >
    print(config["density"],type(config["density"]),type(temp),type(kb))
    add_step += 1
    print_string = "{} {} {:E} {:E}".format(i,
                                               incoming_particle.get_chemical_formula(),
                                               temp,
                                               float(config["density"])*temp*kb*1e6
                                               )

#    print_string += ' {} {}'.format(cols_per_second, run_config.position)
#   run_config.position == run_config.initial_position == initial condensation distance (¿?)
    print_string += ' {} {}'.format(cols_per_second, config["initial_condensation_distance"])


# ******** * * * * * * * *  *  *   *    *     *    *       *   *  *  *  *  * * * * * * * * * *******
    # Check where will be the lunch command stored. Result struct should be a MainParticle, type, which must have a check_result
#    write(sys.stdout, model_system, format='xyz')

    # Obtener datos
    symbols = model_system.get_chemical_symbols()
    positions = model_system.get_positions()
    vels = model_system.get_velocities()

    # Imprimir en formato extendido
    print(len(model_system))
    print("XYZ with velocities: symbol x y z vx vy vz")
    for s, pos, vel in zip(symbols, positions, vels):
        print(f"{s} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}")

    if config["calculator"].upper() == "MACE":
        result_struct = MACE_collision(model_system,config,i)  # atom_movement <- posible opción
    elif config["calculator"].upper() == "LAMMPS":
        calculator.run_collision(model_system, num_atoms_seed, particle_size)
        result_struct = calculator.read_final_structure()
# ******** * * * * * * * *  *  *   *    *     *    *       *   *  *  *  *  * * * * * * * * * *******

    added = result_struct.check_connectivity()
    if added is True:
        print_string += ' True '
        particle = result_struct.copy()
        prev_charge += incoming_particle.info['charge']
        print(print_string)

    else:
        print(print_string)
        print_string += ' False '
        non_attached_steps += 1

    print_string += str(atom_movement)+'\n'
    log_file.write(print_string)
    if config["calculator"].upper() == "LAMMPS":
        calculator.clean_md(i, added)
    print(particle.get_chemical_formula())
# Finishing lines of code
print(str(tm.time()-start_time)+" seconds ")
