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

def RelaxGeometry(atoms,config,index):
    if config["calculator"].upper() == "MACE":
        calc = MACECalculator(model_paths=[config["mace_force_field_file"]], device='cpu', default_dtype="float32")
    elif config["calculator"].upper() == "LAMMPS":
        #calc = LAMMPS
        pass

    atoms.calc = calc
    timestep_ = config["time_step_relaxation"]
    if config["thermostat"] == "Langevin":
        dyn = Langevin(atoms,timestep=timestep_ * units.fs, temperature_K=config["temperature"],friction=0.1)
    elif config["thermostat"] == "Verlet":
        dyn = VelocityVerlet(atoms,timestep=timestep_*units.fs)
    elif config["thermostat"] == "NVTBerendsen":
        dyn = NVTBerendsen(atoms, timestep_ * units.fs, temperature_K=config["temperature"], taut=0.5*100000*units.fs)

    if config["collisions_files"] is not None:
        output_file = config["collisions_files"]+"_relaxation_"+str(index)+".xyz"
    else:
        output_file = sys.stdout

    for i in range(config["relaxation_steps"]):
        dyn.step()
        dyn.atoms.write(output_file, append=True)

    return atoms
