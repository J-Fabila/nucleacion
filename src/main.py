#!/usr/bin/python3

import sys
import utilities
from particle import Particle, Incoming_Part_Library
from lammps_io import Lammps_MD
from ase.io import read
from runconfig import Run_Config
import time as tm
import numpy as np

from mace.calculators import MACECalculator

# Starting time
start_time = tm.time()

# Boltzmann constant
kb = 1.3806e-23


# Reads configuration file 'conf.data', otherwise reads it through command line
conf_file = 'conf.data'
if len(sys.argv) > 1:
    conf_file = sys.argv[1]


# Run will be a class holding all information on the run, at init it will read conf.data
run_config = Run_Config(fname=conf_file)
time = 0.0
# if no input has been specified run will read the conf.data file

# Initialize the main particle
seed = read(run_config.seed)
prev_charge = seed.info['charge']
particle = Particle(symbols=seed.get_chemical_symbols(),
                    positions=seed.get_positions())

# Initialize the incoming particle module. Remember that with ase a single xyz file can contain
# all incoming particles. We have to check that the comments actually contains the pressure
if hasattr(run_config, 'desired_stoichiometry'):
    inc_part_libr = Incoming_Part_Library(run_config.inc_library,
                                          desired_stoichiometry=run_config.desired_stoichiometry)
else:
    inc_part_libr = Incoming_Part_Library(run_config.inc_library)


add_step = 1
temp = run_config.temp
cols_per_second = 1
non_attached_steps = 0

# ******* * * * * * *  *  *   *   *    *   *   *   *   *  * * * * * * * * * * * * *******************

calculator = Lammps_MD(temp=run_config.temp,
                       symb_conv=run_config.element_conv,
                       shells=run_config.shells,
                       charges=run_config.charge,
                       ip_file=run_config.ip_def)  # check

#calculator = MACECalculator(model_paths=['MACE_dip/mace03_run-123.model'], device='cpu', default_dtype="float32")
#atoms.calc = calculator

# ******* * * * * * *  *  *   *   *    *   *   *   *   *  * * * * * * * * * * * * *******************

log_file = open('results.log', 'w')
log_file.write('Step Seed.Chem.Formula Temp den(ncm-3) P(Pa) cols.per.second position added atom_mov\n')

print(run_config.desired_stoichiometry)
for i in range(1, run_config.cycles+1):
    # Returns max distance, although it has all of them
    particle_size = particle.get_maximum_distances()
    num_atoms_seed = len(particle)
    # incoming particle to be closer to the surface
    incoming_particle = inc_part_libr.generate_inc(temp,
                                                   particle_size,
                                                   prev_charge,
                                                   current_stoichiometry=particle.get_chemical_formula())

    model_system, atom_movement = utilities.get_model_system(particle, incoming_particle)


# Check if the incoming particle is too close to the main particle and move it  
# FER UNA FUNCIÓ A PART PER A QUE QUEDI TOT MES RECOLLIT. HAURIA DE LLEGIR LES VARIABLES PARTICLE I INCOMING PARTICLE I RETORNAR EL MODEL_SYSTEM I LA VARIABLE ATOM_MOVEMENT
    
    add_step += 1
    print_string = "{} {} {} {:E} {:E}".format(i,
                                               incoming_particle.get_chemical_formula(),
                                               run_config.temp,
                                               run_config.pressure,
                                               run_config.pressure*run_config.temp*kb*1e6
                                               )
                                               
    print_string += ' {} {}'.format(cols_per_second, run_config.position)


    # Check where will be the lunch command stored. Result struct should be a MainParticle, type, which must have a check_result
    calculator.run_collision(model_system, num_atoms_seed, particle_size)
    result_struct = calculator.read_final_structure()

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
    calculator.clean_md(i, added)
    print(particle.get_chemical_formula())
# Finishing lines of code
print(str(tm.time()-start_time)+" seconds ")
