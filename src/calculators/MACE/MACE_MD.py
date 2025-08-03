import os
import sys
import numpy as np
from mace.calculators import MACECalculator
from ase.data import atomic_numbers, atomic_masses

from mace.calculators import MACECalculator
from ase.io import read
from ase import units
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen

def MACE_collision(atoms,config):
    #Stationary(atoms)
    #ZeroRotation(atoms)#atoms.set_calculator(mace_calc)
    mace_calc = MACECalculator(model_paths=[config["mace_force_field_file"]], device='cpu', default_dtype="float32")
    atoms.calc = mace_calc
    if config["thermostat"] == "Langevin":
        dyn = Langevin(atoms,timestep=1.0 * units.fs, temperature_K=config["temperature"],friction=0.1)
    elif config["thermostat"] == "Verlet":
        dyn = VelocityVerlet(atoms,timestep=1.0*units.fs)
    elif config["thermostat"] == "NVTBerendsen":
        dyn = NVTBerendsen(atoms, 2 * units.fs, temperature_K=config["temperature"], taut=0.5*100000*units.fs)
    timesteps = 1000 # config["timesteps"]
    for i in range(timesteps):
        dyn.step()
        dyn.atoms.write(output_file, append=True)

    return atoms
