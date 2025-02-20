import math


class Run_Config():
    """Configuration of the run."""

    temp = None
    pressure = None

    def __init__(self, fname=None):
        if not fname:
            fname = 'conf.data'

        self.star = None
        self.element_conv = {}
        self.shells = []
        self.charge = {}
        self.inc_library = 'inc_lib.xyz'
        self.pressure = None
        self.temp = None

        with open(fname, 'r') as conf_file:
            for line in conf_file:
                # Particle data
                if "INPUT" in line.upper():
                    self.seed = next(conf_file).rstrip()
                # Incoming particles file
                elif "CONDENSATION SPECIES" in line.upper():
                    self.inc_library = next(conf_file).rstrip()
                    
                elif "TEMPERATURE" in line.upper():
                    self.temp = float(next(conf_file))
                    
                elif "INITIAL CONDENSATION DISTANCE" in line.upper():
                    self.initial_position = float(next(conf_file))
                    self.position = self.initial_position
                    
                elif "DESIRED STOICHIOMETRY" in line.upper():
                    self.desired_stoichiometry = next(conf_file).rstrip()

                elif "IP DEFINITION" in line.upper():
                    self.ip_def = next(conf_file).rstrip()

                elif "DENSITY" in line.upper():
                    self.pressure = float(next(conf_file))

                elif "CYCLES" in line.upper():
                    self.cycles = int(next(conf_file))
                
                elif "ELEMENTS" in line.upper():
                    num_elements = int(next(conf_file))
                    for i in range(num_elements):
                        line2 = next(conf_file).split()
                        symb = line2[0]
                        num_conv = [int(x.split(':')[0]) for x in line2[1:]]
                        self.element_conv[symb] = num_conv
                        for i in line2[1:]:
                            self.charge[int(i.split(':')[0])] = float(i.split(':')[1])

                elif "SHELLS" in line.upper():
                    number_shells = int(next(conf_file))
                    for i in range(number_shells):
                        line2 = next(conf_file)
                        self.shells.append(line2.split()[0])

        
