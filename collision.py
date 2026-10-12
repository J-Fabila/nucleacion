import os
import sys
import yaml
import torch
import time as tm
import numpy as np

from ase import Atoms
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mace.calculators import MACECalculator

from src.utils import masses
from src.utils import parse_formula
from src.utils import CheckConnectivity, GetParticleDiameter
from src.GenerateFragment import offset_radial_fragment, offset_angular_fragment

start_time = tm.time()
kb = 1.3806e-23

conf_file = 'conf.yaml'
if len(sys.argv) > 1:
    conf_file = sys.argv[1]

with open(conf_file, "r") as f:
    config = yaml.safe_load(f)

time = 0.0

seed_1 = read(config["input_file_1"])
seed_2 = read(config["input_file_2"])
output_file = config["output_file"]

theta = np.arccos(1 - 2 * np.random.rand())  # theta: [0, pi]
phi = 2 * np.pi * np.random.rand()           # phi: [0, 2pi]
direccion = np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
])

# Cosas por añadir:
# * Ángulo de rotación a ambas seeds
# * Offset radial o angular
# * Opción para relajación
# * Velocidad inicial de la seed 2
# * Añadir momento angular a ambas seeds
# * Velocidades para ambas seeds

rho_1 = seed_1.get_all_distances().max()
rho_2 = seed_2.get_all_distances().max()
posicion = -(rho_1/2 + rho_2/2 + config["min_dist"]) * direccion

seed_1.translate(-seed_1.get_center_of_mass())
seed_2.translate(posicion)

MaxwellBoltzmannDistribution(seed_1, temperature_K=config["initial_temperature_1"])
MaxwellBoltzmannDistribution(seed_2, temperature_K=config["initial_temperature_2"])

velocidades_1 = seed_1.get_velocities()
velocidades_1 = velocidades_1 + config["velocity_magnitude_1"] * direccion
velocidades_2 = seed_2.get_velocities()
velocidades_2 = velocidades_2 - config["velocity_magnitude_2"] * direccion

seed_1.set_velocities(velocidades_1)

if config["offset"] == "None":
    seed_2.set_velocities([velocidades_2])
elif config["offset"].lower() == "radial":
    posicions = seed_2.get_positions()
    posicions = offset_radial_fragment(posicions,config["max_radial_offset"])
    seed_2.set_positions([posicions])
    seed_2.set_velocities([velocidades_2])
elif config["offset"].lower() == "angular":
    velocidades_2 = offset_angular_fragment(velocidades_2,config["max_radial_offset"], config["min_dist"])
    seed_2.set_velocities([velocidades_2])

atoms = seed_1 + seed_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mace_calc = MACECalculator(model_type=config["mace_model_type"],model_paths=[config["mace_force_field_file"]], device='cuda', default_dtype="float32")
atoms.calc = mace_calc
dyn = VelocityVerlet(atoms, config["time_step_collision"] * units.fs)
dyn.step()
dyn.atoms.write(output_file, append=True) # archivo principal
contador_fallido = 0

for i in range(config["collision_steps"]):
    dyn.step()
    dyn.atoms.write("output_temp.xyz", append=True) # temporal

    if i % 100 == 0 or i == config["collision_steps"] - 1:
        max_dist = dyn.atoms.get_all_distances().max()
        threeshold = 10 * (rho_1 + rho_2)
        if max_dist > threeshold:
            print("Dinámica fallida")
            contador_fallido = contador_fallido + 1
            i = i - 100
            if contador_fallido > 10:
                print("Dinámica falló más de 10 veces")
                break
        else:
            with open("output_temp.xyz", "r") as src, open(output_file, "a") as dst:
                dst.write(src.read())
            contador_fallido = 0
        # borra el archivo temporal
        if os.path.exists("output_temp.xyz"):
            os.remove("output_temp.xyz")

if config["final_relaxation"]:
    print("Performing final relaxation...")

    timestep_ = config["time_step_relaxation"]
    if config["thermostat_relaxation"] == "Langevin":
        dyn = Langevin(atoms,timestep=timestep_ * units.fs, temperature_K=config["relaxation_temperature"],friction=0.1)
    elif config["thermostat_relaxation"] == "Verlet":
        dyn = VelocityVerlet(atoms,timestep=timestep_*units.fs)
    elif config["thermostat_relaxation"] == "NVTBerendsen":
        dyn = NVTBerendsen(atoms, timestep_ * units.fs, temperature_K=config["relaxation_temperature"],taut=0.5*100000*units.fs)

    i = 0
    while i < config["relaxation_steps"]:
        dyn.step()
        dyn.atoms.write(output_file, append=True)
        i = i + 1
    return atoms
