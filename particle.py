from ase import Atoms
from ase.formula import Formula
from ase.io import read
import numpy as np
import random as rnd
import math
import queue


class Particle(Atoms):
    """Main Particle class.

    We obtain all properties and methods from the ASE Atoms class, plus we add
    functions to check if the particle is connected. Also, it will generate
    shell positions from the info dictionary
    Velocities are set-up as A/ps
    """

    kb = 1.380649e-23

    def __init__(self, el_w_shells=None, **kwds):
        super().__init__(**kwds)  # This inherits the __init__ of ase Atoms and allows to add new properties
        if self.get_velocities() is None:
            self.set_velocities(self.get_positions()*0+0.001)
        if np.linalg.norm(self.get_center_of_mass()) > 1:
            cm = self.get_center_of_mass()
            self.set_positions(self.get_positions() - cm)
        # Aligning particle according to MOI if particle is not linear
        moi, vectors_moi = self.get_moments_of_inertia(vectors=True)
        if len(self.get_positions()) > 1:
            if moi[0] > 1:
                inv_vectors_moi = np.linalg.inv(vectors_moi)
                if np.linalg.det(inv_vectors_moi) < 0:
                    sup = inv_vectors_moi.copy()
                    inv_vectors_moi[0] = sup[1]
                    inv_vectors_moi[1] = sup[0]
                new_positions = np.dot(self.get_positions(), inv_vectors_moi)
                self.set_positions(new_positions)

    def check_connectivity(self):
        """Check connectivity of the particle."""
        from ase import data
        connected = True
        temp_cicle = queue.deque()
        list_atoms = self.get_atomic_numbers()
        connectivity_matrix = np.zeros((len(list_atoms),
                                        len(list_atoms)),
                                       dtype=bool
                                       )

        for i in range(len(list_atoms)-1):
            for j in range(i+1, len(list_atoms)):
                bond_dist = ((data.covalent_radii[list_atoms[i]] +
                              data.covalent_radii[list_atoms[j]])) * 1.5

                if self.get_distance(i, j) < bond_dist:
                    connectivity_matrix[i, j] = True
                    connectivity_matrix[j, i] = True

        # Tenim una matriu de conectivitat. Mirar si el primer atom esta
        # conectat amb tots els altres.
        # New try. Just check if 1 atom is connected to all the other ones
        visited = np.zeros(len(list_atoms), dtype=bool)
        visited[0] = True
        self.dfs(connectivity_matrix, 0, temp_cicle, visited)
        if np.sum(visited) != len(list_atoms):
            connected = False

        return connected

    def dfs(self, connectivity_matrix, node, frontera, visited):
        """Depth First Search Algorithm.

        This modified case keeps record of visited nodes even after probing
        different paths. Once all nodes have been visited, return.

        Uncomment print for debbugging.
        """

        # Afegim el primer node a la frontera
        frontera.append(node)
        # Mentre la frontera no estigui buida fer loop
        while frontera:
            node = frontera.popleft()

            # Marquem que el node ha estat visitat
            visited[node] = True

            # Mirem si els nodes adjacents han estat visitats i l'afegim a la frontera
            for child in range(len(connectivity_matrix[node, :])):
                if bool(connectivity_matrix[node, child]) is True:
                    if not visited[child]:  # == False:
                        frontera.appendleft(child)

    def get_maximum_distances(self, all=False):
        moi = self.get_moments_of_inertia()
        moi.reshape(3, 1)
        mass = np.sum(self.get_masses())
        i_eq = np.linalg.inv(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        radius_per_axis = np.dot(i_eq, moi*5/mass)
        radius_per_axis = np.sqrt(radius_per_axis)
        particle_size = max(radius_per_axis)
        if all:
            return particle_size, radius_per_axis
        else:
            return particle_size

    def sph2cart(self, vector_sph):
        vector_cart = np.zeros(3)
        vector_cart[0] = (vector_sph[0]*math.sin(vector_sph[1]) *
                          math.cos(vector_sph[2]))
        vector_cart[1] = (vector_sph[0]*math.sin(vector_sph[1]) *
                          math.sin(vector_sph[2]))
        vector_cart[2] = vector_sph[0]*math.cos(vector_sph[1])
        return vector_cart

    def rotation_matrix(self, angle, vector):
        """This funciton may be unnecesary, ASE probably incorporates it."""
        rotation = np.array([[math.cos(angle)+(vector[0]**2)*(1-math.cos(angle)),
                              vector[0]*vector[1]*(1-math.cos(angle))-vector[2] *
                              math.sin(angle),
                              vector[0]*vector[2]*(1-math.cos(angle)) +
                              vector[1]*math.sin(angle)],
                             [vector[1]*vector[0]*(1-math.cos(angle))+vector[2] *
                              math.sin(angle),
                              math.cos(angle)+(vector[1]**2)*(1-math.cos(angle)),
                              vector[1]*vector[2]*(1-math.cos(angle)) -
                              vector[0]*math.sin(angle)],
                             [vector[2]*vector[0]*(1-math.cos(angle))-vector[1] *
                              math.sin(angle),
                              vector[2]*vector[1]*(1-math.cos(angle)) +
                              vector[0]*math.sin(angle),
                              math.cos(angle)+(vector[2]**2)*(1-math.cos(angle))]])

        return rotation

    def approximate_sa(self):
        """Ellispoid SA approximation."""
        sizes = self.get_maximum_distances(all=True)[1]
        p = 1.6075
        root = sizes[0]**p*sizes[1]**p + sizes[0]**p*sizes[2]**p + sizes[1]**p*sizes[2]**p
        sa = 4*math.pi*(root/3)**(1/p)
        return sa


class Incoming_Particle(Particle):
    """
    Class that generates an incoming particle and its properties.

    The incoming particle is a Particle class, thus has self.shells and
    self.velocities. The properties that are computed once generation are:
    -Random vector to the CM of the particle
    -offset to allow for non-direct collisions
    -velocity (requires Temp)

    Every instance of Incoming particle, which is done by:
    >Incoming_Particle(Temp=,seed_r, symbols=[], positions=[[]])

    Will automatically reorient and rotate the particle
    """

    def __init__(self, temp=None, seed_r=None, **kwds):
        super().__init__(**kwds)  # This inherits the __init__ of ase Atoms and allows to add new properties
        self.set_positions(self.get_positions() - self.get_center_of_mass())  # Center to CM
        velocity_magnitude = self.calc_velocity(temp)
        # Check hoe does the velocity magnitude work in each function. There could be an inconsistency
        # Between what is used in vector_to_cm and when assigning the velocities to the incoming particle

        # Compute random position on sphere, with distance proportional to velocity and security treshold
        # Generating random polar position
        max_d = seed_r+1+velocity_magnitude*0.3
        polar_vector = np.array([max_d,
                                 rnd.uniform(0, 2*math.pi),
                                 rnd.uniform(0, math.pi)])

        vector_to_cm = self.sph2cart(polar_vector)

        # Get impact parameter
        impact_par = self.set_offset(vector_to_cm, seed_r*0.7)

        # Move particle position according to generated random point
        self.set_positions(np.dot(self.get_positions(), self.random_rotation()))  # Rotate particle
        self.set_positions(self.get_positions() + vector_to_cm + impact_par)  # displacement

        self.set_velocities(-vector_to_cm/np.linalg.norm(vector_to_cm)*velocity_magnitude)

    def calc_velocity(self, temp):
        """Calculate particle CM velocity according to Maxwell Boltzmann dist.

        We use 90% of the distribution.

        See:
        https://mathworld.wolfram.com/MaxwellDistribution.html
        and scipy stats maxwell
        """
        from scipy.stats import maxwell
        import math
        a_param = math.sqrt(self.kb*temp/(self.get_masses().sum()*1.66e-27))
        max_model = maxwell(scale=a_param)  # Velocity is in m/s
        min_v, max_v = max_model.interval(0.4)  # uses 40% of data around mean
        random_vel = max_model.rvs()
        while random_vel < min_v or random_vel > max_v:
            random_vel = max_model.rvs()
        random_vel = random_vel*1e-2
        return random_vel

    def set_offset(self, vector, max_distance):
        """Return a vector with a random offset.

        This function will generate a vector with a random offset with respect to
        the center. Therefore it requires a vector to translate and a radii for
        the max value to translate. Two possible aproaches:
            1. Obtain a point (P) in a circle using circle equation.
            2. Rescale it's vector randomly from 0 to max distance of the particle
            3. Convert vector to original coordinates
            3.1. Calculate the angle between Z axis and vector (alpha)
            3.2. Calculate perpendicular vector (R) between Z axis and vector
            3.3. Do rotation around (R) with angle alpha of the point (P)
        return a vector with
        """
        # 1. Point of a circle
        x_pos = np.random.uniform(-1, 1)
        y_pos = math.sqrt(1-x_pos**2)
        if np.random.uniform(0, 1) > 0.5:
            y_pos = -y_pos
        point = np.array([x_pos, y_pos, 0.0])
        # 2. Reescale
        point = point * np.random.uniform(0, max_distance)
        # 3. 1
        angle = math.acos(np.dot(vector, np.array([0.0, 0.0, 1.0])) /
                          np.linalg.norm(vector))
        # 3. 2
        perp_vect = (np.array([vector[1], -vector[0], 0.0]) /
                     np.linalg.norm(np.array([vector[1], -vector[0], 0.0])))
        # 3. 3
        rotation = self.rotation_matrix(angle, perp_vect)
        point = np.dot(point, rotation)
        return point

    def random_rotation(self):
        """Generate a random rotation for a molecule."""
        rnd.seed()
        vector = np.array([rnd.random(), rnd.random(), rnd.random()])
        vector = vector/np.linalg.norm(vector)
        angle = rnd.uniform(0, 2*math.pi)
        rotation = self.rotation_matrix(angle, vector)
        return rotation


class Incoming_Part_Library():
    def __init__(self,
                 fname,
                 desired_stoichiometry=None):
        """Init with variables.

        """
        library = read(fname, index=':')
        self.positive = {}
        self.negative = {}
        self.neutral = {}
        self.desired_stoichiometry = desired_stoichiometry

        for inc_mon in library:

            if inc_mon.info['charge'] < 0:
                self.negative[inc_mon.get_chemical_formula()] = [inc_mon, inc_mon.info['charge'], inc_mon.info['prob']]
            elif inc_mon.info['charge'] == 0:
                neutral[inc_mon.get_chemical_formula()] = [inc_mon, inc_mon.info['charge'], inc_mon.info['prob']]
            elif inc_mon.info['charge'] > 0:
                self.positive[inc_mon.get_chemical_formula()] = [inc_mon, inc_mon.info['charge'], inc_mon.info['prob']]

    def generate_inc(self, temp, seed_dist, prev_charge=None, current_stoichiometry=None):

        if not prev_charge:
            prev_charge = rnd.random() - 0.5
        if prev_charge < 0 and bool(self.positive):
            slctd_inc = self.positive
            slctd_inc.update(self.neutral)
        elif prev_charge > 0 and bool(self.negative):
            slctd_inc = self.negative
            slctd_inc.update(self.neutral)
        else:
            slctd_inc = {}
            slctd_inc.update(self.positive)
            slctd_inc.update(self.neutral)
            slctd_inc.update(self.negative)

        if self.desired_stoichiometry:
            current_formula = Formula(current_stoichiometry).count()
            desired_formula = Formula(self.desired_stoichiometry).count()

            nc_d = np.linalg.norm(list(desired_formula.values()))
            nc_c = np.linalg.norm(list(current_formula.values()))
            desired_formula = {k: desired_formula[k] / nc_d for k in desired_formula.keys()}
            current_formula = {k: current_formula[k] / nc_c for k in current_formula.keys()}

            element_keys = list(current_formula.keys()) + list(desired_formula.keys())
            for fragment in slctd_inc:
                element_keys += list(Formula(fragment).count().keys())
            element_keys = set(element_keys)

            formula_vectors = {}
            for fragment in slctd_inc:
                fragment_formula = Formula(fragment).count()
                fragment_formula_norm = {}
                formula_vector = []

                for element in element_keys:
                    formula_diff = current_formula.get(element, 0) + fragment_formula.get(element, 0)/nc_c
                    fragment_formula_norm[element] = formula_diff

                nc_f = np.linalg.norm(list(fragment_formula_norm.values()))
                fragment_formula_norm = {k: fragment_formula_norm[k] / nc_f for k in fragment_formula_norm.keys()}

                for element in element_keys:
                    formula_diff = desired_formula.get(element, 0) - fragment_formula_norm.get(element, 0)
                    formula_vector.append(formula_diff)

                formula_vectors[np.linalg.norm(formula_vector)] = fragment

            slctd_inc = slctd_inc[formula_vectors[min(list(formula_vectors.keys()))]][0]

        else:
            slctd_inc = self.random_choice(slctd_inc)

        inc_part = Incoming_Particle(positions=slctd_inc.get_positions(),
                                     symbols=slctd_inc.get_chemical_symbols(),
                                     temp=temp,
                                     seed_r=seed_dist,
                                     info=slctd_inc.info)
        return inc_part

    def random_choice(self, dict_molecules):

        weights = [dict_molecules[ele][2] for ele in dict_molecules]
        pop = list(dict_molecules.keys())
        chosen_particle = rnd.choices(pop, weights=weights)[0]

        return dict_molecules[chosen_particle][0]
