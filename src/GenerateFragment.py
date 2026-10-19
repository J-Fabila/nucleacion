import numpy as np
from ase import Atoms
from ase.units import kB, _amu
from src.utils import masses, MaxwellBoltzmannAverageSpeed
from collections import Counter
import random
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
inicial = {'Ti': 1}
deseado = {'Ti': 4, 'C': 4}
disponibles = ['C', 'Ti']

def GenerateFragment(inicial, config, rho):
    """
    Genera elementos hasta alcanzar la estequiometría deseada.

    inicial : dict        Conteo actual de elementos, por ejemplo {'Ti': 1}.
    objetivo : dict        Conteo deseado de elementos, por ejemplo {'Ti': 1, 'C': 2}.
    disponibles : list        Lista de elementos que se pueden añadir, por ejemplo ['C', 'O'].
    orden_aleatorio : bool        Si es True, elige elementos aleatoriamente de entre los permitidos.
    lista_elementos : list        Lista de elementos generados para alcanzar la estequiometría.
    """
    objetivo = config["desired_stoichiometry"]
    disponibles = config["elements"]
    tempe = config["temperature"]

    conteo_actual = Counter(inicial)
    conteo_objetivo = Counter(objetivo)

    if conteo_actual != conteo_objetivo:
        candidatos = []
        for elem in disponibles:
            if conteo_actual[elem] < conteo_objetivo.get(elem, 0):
                candidatos.append(elem)

        if not candidatos:
            raise ValueError("Imposible to achieve stoichiometry with current list")

        nuevo = random.choice(candidatos)

        theta = np.arccos(1 - 2 * np.random.rand())  # theta: [0, pi]
        phi = 2 * np.pi * np.random.rand()           # phi: [0, 2pi]
        direccion = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        dist_min = 1

        if config["dist_from_seed"] is None:
            dist_min = 1
        else:
            dist_min = config["dist_from_seed"]

        posicion = -(rho/2+dist_min) * direccion

        particula = Atoms(nuevo, positions=[posicion], masses=[masses[nuevo]])
        if config["initialize_velocity"] == "MaxwellBoltzmann":
            MaxwellBoltzmannDistribution(particula, temperature_K=config["InitialTemperature"])
            velocidad_magnitud = np.linalg.norm(particula.get_velocities())
        elif config["initialize_velocity"] == "AverageTemperature":
#            velocidad_magnitud = 0.13610830385858072
            velocidad_magnitud = MaxwellBoltzmannAverageSpeed(config["InitialTemperature"], nuevo, config["n_samples"])
        elif config["initialize_velocity"] == "Manual":
            velocidad_magnitud = config["InitialVelocity"]

        velocidad_vector = direccion * velocidad_magnitud

        if config["offset"] == "None":
            particula.set_velocities([velocidad_vector])
        elif config["offset"].lower() == "radial":
            posicion = offset_radial(posicion,rho)
            particula.set_positions([posicion])
        elif config["offset"].lower() == "angular":
            velocidad_vector = offset_angular(velocidad_vector,rho, dist_min)
        particula.set_velocities([velocidad_vector])

    return particula

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


def offset_radial_fragment(posicion, rho):
    posicion_original = np.array(posicion, dtype=float)
    posiciones = posicion_original.copy()

    rand_vec = np.random.normal(size=3)
    rand_vec /= np.linalg.norm(rand_vec)
    offset_magnitude = np.random.uniform(0, rho)


    for i, pos in enumerate(posicion_original):
        norm_pos = np.linalg.norm(pos)

        if norm_pos == 0:
            continue

        pos_unit = pos / norm_pos

        perp_vec = rand_vec - np.dot(pos_unit, rand_vec) * pos_unit
        norm_perp = np.linalg.norm(perp_vec)

        if norm_perp == 0:
            continue

        perp_vec /= norm_perp
        offset = perp_vec * offset_magnitude
        posiciones[i] = pos + offset

    return posiciones

def offset_angular_fragment(velocidad_vector, rho, dist_min):
    velocidad_original = np.array(velocidad_vector, dtype=float)
    velocidades = velocidad_original.copy()

    rand_vec = np.random.normal(size=3)
    rand_vec /= np.linalg.norm(rand_vec)

    offset_magnitude = np.random.uniform(0, rho / (rho + dist_min))

    for i, vel_original in enumerate(velocidad_original):
        norm_vel = np.linalg.norm(vel_original)

        if norm_vel == 0:
            continue

        vel_unit = vel_original / norm_vel
 
        perp_vec = rand_vec - np.dot(rand_vec, vel_unit) * vel_unit
        norm_perp = np.linalg.norm(perp_vec)

        if norm_perp == 0:
            continue

        perp_vec /= norm_perp

        vel = rho / (rho + dist_min) * norm_vel
        offset = perp_vec * offset_magnitude * vel
        
        velocidades[i] = vel_original + offset

    return velocidades
