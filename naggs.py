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

seed_ = read(config["input_file"])
MaxwellBoltzmannDistribution(seed_, temperature_K=config["temperature"])

print("i,current_stoichiometry,rho,accepted")

for i in range(1, config["cycles"]+1):
    seed = seed_.copy()
    current_stoichiometry = parse_formula(str(seed.symbols))
    seed.translate(-seed.get_center_of_mass())
    relaxed = RelaxGeometry(seed,config,i)
    accepted = CheckConnectivity(relaxed,3.5)
    # while accepted:
    if accepted:
        rho = GetParticleDiameter(relaxed) # antes estaba seed
        fragment = GenerateFragment(current_stoichiometry,config, rho)
        if fragment is None: # No more atoms to add
            print(i,current_stoichiometry,rho,accepted,"break")
            break
        else:
            # collision = relaxed + fragment
            collision = relaxed + fragment
            collision = RunCollision(collision,config,i)
            accepted = CheckConnectivity(collision,3.5)
            print(i,current_stoichiometry,rho,accepted)
            if accepted:
                seed_ = collision
            else:
                print("Collision not converged")
    else:
        print(i,"Relaxation not converged")

print(str(tm.time()-start_time)+" seconds ")

"""
stoichiometry = False # Inicializar el ciclo
cont = 0
while current_stoichiometry =! desired_stoichiometry  or cont < config["cycles"]:
    seed.translate(-seed.get_center_of_mass())
    accepted = False
    while accepted == False: # and current stiohetry =! desired, asi evitamos que haga la ultima, pero creo que no hará falta
        relaxed = RelaxGeometry(seed,config,i)
        accepted = CheckConnectivity(relaxed,3.5)

    accepted = False
    while accepted == False:
        rho = GetParticleDiameter(relaxed)
        fragment = GenerateFragment(current_stoichiometry,config, rho)
        if fragment is None: # No more atoms to add
            print(i,current_stoichiometry,rho,accepted,"break")
            # mas bien seria end
            # pero al devolver al while siguiente la estequiometria es igual y entonces true entonces sale, o sea qu etodo bien
            break
        else:
            collision = relaxed + fragment
            collision = RunCollision(collision,config,i)
            accepted = CheckConnectivity(collision,3.5)
            print(i,current_stoichiometry,rho,accepted)
            if accepted:
                seed = collision
                cont = cont +1
            else:
                print("Collision not converged, starting new collision")
    current_stoichiometry = parse_formula(str(seed.symbols))

"""
