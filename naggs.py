import sys
import yaml
import time as tm

from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from src.RunCollision import RunCollision
from src.RelaxGeometry import RelaxGeometry
from src.GenerateFragment import GenerateFragment
from src.utils import CheckConnectivity, GetParticleDiameter
from src.utils import parse_formula

start_time = tm.time()
kb = 1.3806e-23

conf_file = 'conf.yaml'
if len(sys.argv) > 1:
    conf_file = sys.argv[1]

with open(conf_file, "r") as f:
    config = yaml.safe_load(f)

time = 0.0
seed = read(config["input_file"])
seed.translate(-seed.get_center_of_mass())
MaxwellBoltzmannDistribution(seed, temperature_K=config["temperature"])

for i in range(1, config["cycles"]+1):
    initial_stoichiometry = parse_formula(str(seed.symbols))
    RelaxGeometry(seed,config,i)
    rho = GetParticleDiameter(seed)
    fragment = GenerateFragment(initial_stoichiometry,config["desired_stoichiometry"],config["elements"],config["temperature"], rho)
    if fragment is None: # No more atoms to add
        break
    else:
        collision = seed + fragment
        RunCollision(collision,config,i)
        accepted = CheckConnectivity(collision,3.5)
        if accepted:
            seed = collision

print(str(tm.time()-start_time)+" seconds ")
