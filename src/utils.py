import numpy as np
from ase.geometry import geometry

import re
from collections import defaultdict

elements = [ "A",  "H", "He", "Li", "Be",  "B",  "C",  "N",  "O",  "F", "Ne",
      "Na", "Mg", "Al", "Si",  "P",  "S", "Cl", "Ar",  "K", "Ca", "Sc", "Ti",
       "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
      "Br", "Kr", "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
      "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe", "Cs", "Ba", "La", "Ce",
      "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
      "Lu", "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl"  "Pb",
      "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu",
      "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
      "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

radii  ={"H" :0.46, "He":1.22, "Li":1.57, "Be":1.12, "B" :0.81,
         "C" :0.77, "N" :0.74, "O" :0.74, "F" :0.72, "Ne":1.6,
         "Na":1.91, "Mg":1.6,  "Al":1.43, "Si":1.18,  "P":1.10,
          "S":1.04, "Cl":0.99, "Ar":1.92,  "K":2.35, "Ca":1.97,
         "Sc":1.64, "Ti":1.44, "Va":0.8,  "Cr":1.29, "Mn":1.37,
         "Fe":1.26, "Co":1.25, "Ni":1.25, "Cu":1.28, "Zn":1.37,
         "Ga":1.53, "Ge":1.22, "As":1.21, "Se":1.04, "Br":1.14,
         "Kr":1.98, "Rb":2.5,  "Sr":2.15,  "Y":1.82, "Zr":1.6,
         "Nb":1.47, "Mo":1.4,  "Tc":1.35, "Ru":1.34, "Rh":1.34,
         "Pd":1.37, "Ag":1.66, "Cd":1.52, "In":1.67, "Sn":1.58,
         "Sb":1.41, "Te":1.37, "I" :1.33, "Xe":2.18, "Cs":2.72,
         "Ba":2.24, "La":1.88, "Ce":1.82, "Pr":1.82, "Nd":1.82,
         "Pm":1.81, "Sm":1.81, "Eu":2.06, "Gd":1.79, "Tb":1.77,
         "Dy":1.77, "Ho":1.76, "Er":1.75, "Tm":1.00, "Yb":1.94,
         "Lu":1.72, "Hf":1.59, "Ta":1.47, "W" :1.41, "Re":1.37,
         "Os":1.35, "Ir":1.35, "Pt":1.39, "Au":1.44, "Hg":1.55,
         "Tl":1.71, "Pb":1.75, "Bi":1.82, "Po":1.77, "At":1.62,
         "Rn":1.8,  "Fr":1.00, "Ra":2.35, "Ac":2.03, "To":1.8,
         "Pa":1.63, "U" :1.56, "Np":1.56, "Pl":1.5,  "Am":1.73,
         "Cm":1.8,  "Bk":1.8,  "Cf":1.8,  "Es":1.8}

masses ={"H" :1.00784,   "He":4.002602, "Li":6.938,    "Be":9.0121831,   "B":10.806,
          "C":12.0096,    "N":14.00643,  "O":15.99903,  "F":18.9984031, "Ne":20.1797,
         "Na":22.989769, "Mg":24.304,   "Al":26.981538,"Si":28.084,      "P":30.973761998,
          "S":32.059,    "Cl":35.446,   "Ar":39.0983,   "K":39.0983,    "Ca":40.0788,
         "Sc":44.95590,  "Ti":47.867,   "Va":50.9415,  "Cr":51.9961,    "Mn":54.938044,
         "Fe":55.845,    "Co":58.933194,"Ni":58.6934,  "Cu":63.546,     "Zn":65.38,
         "Ga":69.723,    "Ge":72.630,   "As":74.921595,"Se":78.971,     "Br":79.901,
         "Kr":83.798,    "Rb":85.467,   "Sr":87.62,     "Y":88.90584,   "Zr":88.90584,
         "Nb":92.90637,  "Mo":95.95,    "Tc":98,       "Ru":101.07,     "Rh":102.90550,
         "Pd":106.42,    "Ag":107.8682, "Cd":112.414,  "In":114.818,    "Sn":118.710,
         "Sb":121.760,   "Te":127.60,    "I":126.90447,"Xe":131.293,    "Cs":132.905451,
         "Ba":137.327,   "La":138.90547,"Ce":140.116,  "Pr":140.90766,  "Nd":144.242,
         "Pm":145,       "Sm":150.36,   "Eu":151.964,  "Gd":157.25,     "Tb":158.92535,
         "Dy":162.500,   "Ho":164.93033,"Er":167.259,  "Tm":168.93422,  "Yb":173.054,
         "Lu":174.9668,  "Hf":178.49,   "Ta":180.94788, "W":183.84,     "Re":183.84,
         "Os":190.23,    "Ir":192.217,  "Pt":195.084,  "Au":196.966569, "Hg":200.592,
         "Tl":204.382,   "Pb":207.2,    "Bi":208.98040,"Po":209,        "At":210,
         "Rn":222,       "Fr":223,      "Ra":226,      "Ac":227,        "To":232.0377,
         "Pa":231.03588,  "U":238.02891,"Np":237,      "Pl":244,        "Am":241.0568293,
         "Cm":243.061389,"Bk":247.07030,"Cf":249.07485,"Es":252.082980}

charges={"H" : 1, "He": 2, "Li": 3, "Be": 4, "B" : 5,
         "C" : 6, "N" : 7, "O" : 8, "F" : 9, "Ne":10,
         "Na":11, "Mg":12, "Al":13, "Si":14, "P" :15,
         "S" :16, "Cl":17, "Ar":18, "K" :19, "Ca":20,
         "Sc":21, "Ti":22, "Va":23, "Cr":24, "Mn":25,
         "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30,
         "Ga":31, "Ge":32, "As":33, "Se":34, "Br":35,
         "Kr":36, "Rb":37, "Sr":38, "Y" :39, "Zr":40,
         "Nb":41, "Mo":42, "Tc":43, "Ru":44, "Rh":45,
         "Pd":46, "Ag":47, "Cd":48, "In":49, "Sn":50,
         "Sb":51, "Te":52, "I" :53, "Xe":54, "Cs":55,
         "Ba":56, "La":57, "Ce":58, "Pr":59, "Nd":60,
         "Pm":61, "Sm":62, "Eu":63, "Gd":64, "Tb":65,
         "Dy":66, "Ho":67, "Er":68, "Tm":69, "Yb":70,
         "Lu":71, "Hf":72, "Ta":73, "W" :74, "Re":75,
         "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80,
         "Tl":81, "Pb":82, "Bi":83, "Po":84, "At":85,
         "Rn":86, "Fr":87, "Ra":88, "Ac":89, "To":90,
         "Pa":91, "U" :92, "Np":93, "Pl":94, "Am":95,
         "Cm":96, "Bk":97, "Cf":98, "Es":99}

def CheckConnectivity(atoms, threshold=2.0):
    positions = atoms.get_positions()
    N = len(positions)
    dist_matrix = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    adjacency = (dist_matrix < threshold) & (dist_matrix > 0)
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in range(N):
            if adjacency[node, neighbor] and neighbor not in visited:
                dfs(neighbor)
    dfs(0)
    return len(visited) == N

#def CheckConnectivity(atoms, threshold=2.0):
#    if atoms.get_all_distances().max() < 3.5:
#        return False
#    else:
#        return True

def GetParticleDiameter(atoms):
    return atoms.get_all_distances().max()

def parse_formula(formula):
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    composition = defaultdict(int)
    for (element, count) in matches:
        count = int(count) if count else 1
        composition[element] += count

    return dict(composition)
