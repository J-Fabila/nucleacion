import numpy as np


def get_model_system(particle,incoming_particle):

    atom_movement=False
    model_system=particle+incoming_particle

    if incoming_particle.get_chemical_formula() != 'SiO' and incoming_particle.get_chemical_formula() != 'OSi':

       distances=model_system.get_all_distances()[-1]
       distances=distances[:-1]
       minindex=np.argmin(distances)

       if distances[minindex] < 3 :

          atom_movement=True
          mindist=model_system.get_distance(minindex,len(model_system)-1)
          dist_increase=2

          ratio=(mindist+dist_increase)/mindist

          x,y,z=model_system.get_positions()[-1]
          x0,y0,z0=model_system.get_positions()[minindex]

          pos=model_system.get_positions()
          pos[-1]=np.asarray([x-x0,y-y0,z-z0])*ratio+np.asarray([x0,y0,z0])

          model_system.set_positions(pos)

    else :
         
         distances1=model_system.get_all_distances()[-1]
         distances1=distances1[:-2]
        
         distances2=model_system.get_all_distances()[-2]
         distances2=distances2[:-2]

         if min(distances2) < min(distances1):
            minindex=np.argmin(distances2)
            if distances2[minindex] < 3 :
   
               atom_movement=True
               mindist=model_system.get_distance(minindex,-2)
               dist_increase=2
               ratio=(mindist+dist_increase)/mindist
   
               x,y,z=model_system.get_positions()[-2]
               x0,y0,z0=model_system.get_positions()[minindex]
   
               vector=model_system.get_positions()[-1]-model_system.get_positions()[-2]
   
               pos=model_system.get_positions()
               pos[-2]=np.asarray([x-x0,y-y0,z-z0])*ratio+np.asarray([x0,y0,z0])
               pos[-1]=pos[-2]+vector
   
               model_system.set_positions(pos)

         else:
            minindex=np.argmin(distances1)
         
            if distances1[minindex] < 3 :
         
               atom_movement=True
               mindist=model_system.get_distance(minindex,-1)
               dist_increase=2
               ratio=(mindist+dist_increase)/mindist
         
               x,y,z=model_system.get_positions()[-1]
               x0,y0,z0=model_system.get_positions()[minindex]
         
               vector=model_system.get_positions()[-2]-model_system.get_positions()[-1]
         
               pos=model_system.get_positions()
               pos[-1]=np.asarray([x-x0,y-y0,z-z0])*ratio+np.asarray([x0,y0,z0])
               pos[-2]=pos[-1]+vector
         
               model_system.set_positions(pos)

     
    return model_system, atom_movement
