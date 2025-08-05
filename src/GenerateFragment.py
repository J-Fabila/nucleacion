import numpy as np
from ase import Atoms
from ase.units import kB, _amu
from src.utils import masses
from collections import Counter
import random
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
inicial = {'Ti': 1}
deseado = {'Ti': 4, 'C': 4}
disponibles = ['C', 'Ti']

def GenerateFragment(inicial, objetivo, disponibles,tempe, rho):
#def generar_hasta_estequiometria(inicial, objetivo, disponibles):
    """
    Genera elementos hasta alcanzar la estequiometría deseada.

    inicial : dict        Conteo actual de elementos, por ejemplo {'Ti': 1}.
    objetivo : dict        Conteo deseado de elementos, por ejemplo {'Ti': 1, 'C': 2}.
    disponibles : list        Lista de elementos que se pueden añadir, por ejemplo ['C', 'O'].
    orden_aleatorio : bool        Si es True, elige elementos aleatoriamente de entre los permitidos.
    lista_elementos : list        Lista de elementos generados para alcanzar la estequiometría.
    """
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
        posicion = -rho * direccion

        particula = Atoms(nuevo, positions=[posicion], masses=[masses[nuevo]])
        MaxwellBoltzmannDistribution(particula, temperature_K=tempe*10)
        ekin = particula.get_kinetic_energy()
        velocidad_magnitud = np.linalg.norm(particula.get_velocities())
        velocidad_vector = direccion * velocidad_magnitud
        particula.set_velocities([velocidad_vector])
        ekin = particula.get_kinetic_energy()
        return particula
