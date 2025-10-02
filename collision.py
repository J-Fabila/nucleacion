import sys
import yaml
import time as tm
import numpy as np
from ase import Atoms

from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from src.RunCollision import RunCollision
from src.RelaxGeometry import RelaxGeometry
from src.GenerateFragment import GenerateFragment
from src.utils import CheckConnectivity, GetParticleDiameter
from src.utils import parse_formula
from src.utils import masses

temperatura = 1500
dist_min = 2
seed_1 = read("/home/jorge/work_dir/TiC/geometrias/Ti14C13.xyz")
seed_2 = read("/home/jorge/work_dir/TiC/geometrias/Ti14C13.xyz")

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

# Velocidad del offset
velocidad_magnitud = np.linalg.norm(particula.get_velocities())
velocidad_vector = direccion * velocidad_magnitud
# Velocidades de la NP seed_2
velocidades = seed_2.get_velocities()
velocidades = velocidades + velocidad_vector
particula.set_velocities([velocidades])
