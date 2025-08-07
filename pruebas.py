from  src.GenerateFragment import GenerateFragment

inicial = {'Ti': 1}
deseado = {'Ti': 4, 'C': 4}
disponibles = ['C', 'Ti']

for i in range(50):
    a = GenerateFragment(inicial, deseado, disponibles, 100, 10)
    pos = a.get_positions()[0,:]
    vel = a.get_velocities()[0,:]
    print("C ",pos[0],pos[1],pos[2], vel[0],vel[1],vel[2])
