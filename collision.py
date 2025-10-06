
import os
import sys
import yaml
import time as tm
import numpy as np
from ase import Atoms
import torch

from mace.calculators import MACECalculator
from ase.io import read, write
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
#from ase.md.nosehoover import NoseHoover
from ase.md.langevin import Langevin
from ase.md.langevin_filter import LangevinFilter
from ase.md.nvtberendsen_filter import NVTBerendsenFilter
#from ase.md.berendsen_isokinetic import NVTBerendsenIsokinetic
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet

from src.RunCollision import RunCollision
from src.RelaxGeometry import RelaxGeometry
from src.GenerateFragment import GenerateFragment
from src.utils import CheckConnectivity, GetParticleDiameter
from src.utils import parse_formula
from src.utils import masses

temperatura = 1500
dist_min = 2
timesteps = 1000
seed_1 = read("/home/yuca/workdir/TiC/geometrias/Ti14C13.xyz")
seed_2 = read("/home/yuca/workdir/TiC/geometrias/Ti14C13.xyz")
output_file = "colision.xyz"

def offset_radial(posicion, rho):
    posicion_original = posicion
    posicion = posicion / np.linalg.norm(posicion)
    rand_vec = np.random.normal(size=3)
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    perp_vec = rand_vec - (np.dot(posicion, rand_vec)*posicion)
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    offset = perp_vec * np.random.uniform(0, rho)
    return posicion_original + offset

def offset_angular(velocidad_vector, rho, dist_min):
    velocidad_original = velocidad_vector
    norm_vel = np.linalg.norm(velocidad_vector)
    velocidad_vector = velocidad_vector / norm_vel
    rand_vec = np.random.normal(size=3)
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    perp_vec = rand_vec - (np.dot(rand_vec, velocidad_vector) * velocidad_vector)
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    vel = rho / (rho+dist_min) * norm_vel
    offset = perp_vec * np.random.uniform(0,vel)
    return velocidad_original + offset

MaxwellBoltzmannDistribution(seed_1, temperature_K=temperatura)
seed_1.translate(-seed_1.get_center_of_mass())

MaxwellBoltzmannDistribution(seed_2, temperature_K=temperatura)

theta = np.arccos(1 - 2 * np.random.rand())  # theta: [0, pi]
phi = 2 * np.pi * np.random.rand()           # phi: [0, 2pi]
direccion = np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
])
rho = 5
#rho = seed_1.get_all_distances().max()

posicion = -(rho/2+dist_min) * direccion
pos_seed_2 = seed_2.get_positions()
pos_seed_2_ = pos_seed_2 + posicion

seed_2.set_positions(pos_seed_2_)

particula = Atoms("C", positions=[posicion], masses=[masses["C"]])
MaxwellBoltzmannDistribution(particula, temperature_K=temperatura)

# Velocidad del offset
velocidad_magnitud = np.linalg.norm(particula.get_velocities())
velocidad_vector = direccion * velocidad_magnitud
# Velocidades de la NP seed_2
velocidades = seed_2.get_velocities()
velocidades = velocidades + velocidad_vector
print("VELOCIDADES", velocidades.shape, velocidad_vector.shape)
velocidades = velocidades[ :, :]
print("VELOCIDADES", velocidades.shape)
seed_2.set_velocities(velocidades)

# A partir de aqui pega la parte del dm hoover de yuca, la que funciona
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mace_calc = MACECalculator(model_type="EnergyDipoleMACE",model_paths=['MACE_dip/mace03_run-123.model'], device='cuda', default_dtype="float32")
atoms = seed_1 + seed_2
atoms.calc = mace_calc
dyn = NVTBerendsenFilter(atoms, 2 * units.fs, temperature_K=temperatura, taut=0.005*1000*units.fs)
dyn.step()
dyn.atoms.write(output_file, append=True) # archivo principal
contador_fallido = 0

for i in range(timesteps):
    dyn.step()
#    dyn.atoms.write(output_file, append=True)
    dyn.atoms.write("output_temp.xyz", append=True) # temporal

#    temperaturas.append(dyn.atoms.get_temperature())
    atoms.get_dipole_moment()
    #dipolos.append(dyn.atoms.get_dipole_moment())
#    fuerzas.append(dyn.atoms.get_forces())
#    energias.append(dyn.atoms.get_potential_energy())
#    print(i,temperaturas[i])
    if i % 100 == 0 or i == timesteps - 1:
        max_dist = dyn.atoms.get_all_distances().max()
        if max_dist > 3*rho: #float(config["dist_from_seed"]):
            print("Dinámica fallida")
            contador_fallido = contador_fallido + 1
            i = i - 100 # devuelve el contador 100 pasos antes
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
