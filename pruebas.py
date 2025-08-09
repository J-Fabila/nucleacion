import yaml
import sys
from  src.GenerateFragment import GenerateFragment

conf_file = 'conf.yaml'
if len(sys.argv) > 1:
    conf_file = sys.argv[1]

with open(conf_file, "r") as f:
    config = yaml.safe_load(f)

inicial = {'Ti': 1}
deseado = {'Ti': 4, 'C': 4}
disponibles = ['C', 'Ti']
rho = 8
for i in range(50):
#    a = GenerateFragment(inicial, deseado, disponibles, 100, 10)
    a = GenerateFragment(inicial,config, rho)

    pos = a.get_positions()[0,:]
    vel = a.get_velocities()[0,:]
    print("C ",pos[0],pos[1],pos[2], vel[0],vel[1],vel[2])
