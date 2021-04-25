"""
Loss 4 attempts to find z values for points such that loss function, as specified in Vijais 2018 paper, 
is minimized. In particular, this loss function tries to minimize the sum of 
1) Deviation from symmetry 
2) - compactness of convex hull. 
3) Deviations from planarity, 
where planarity and compactness are scaled by experimentation.   
"""

from scipy import optimize
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sklearn.preprocessing import normalize
import numpy as np
import pickle
from numerical_summary import sos
from helpers import plot3Ds

with open('./data/everything2.pickle', 'rb') as handle:
    everything = pickle.load(handle)


#######################################################################
########      FUNCTION DEFINITIONS          ###########################
#######################################################################

def compactness(xyz):
    """
    Input: xyz 
    Output: compactness (V^2/SA^3) of convex hull of 3D points. 
    """
    xyz = np.array(xyz)
    ch = ConvexHull(xyz, qhull_options="QJ")
    return ch.volume**2/ch.area**3


def get_angle(p1, p2, p3):
  """
  Given 3 (x,y,z) 3x1 vectors p1, p2, p3, computes angle between p1 - p2 and p3 - p2
  using the dot product cosine rule. 
  """
  e1 = p1 - p2
  e2 = p3 - p2
  x = np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2))
  return np.arccos(x)



def loss4(z, xy, sym_triples, faces, w_plan=10, w_comp=.5, components = False):
    """
    Symmetry + Compactness + Planarity
    Implemented using a bunch of loops, so not the most efficient. 
    Inputs: xy: numpy ndarray 7 x 2
            sym_triples: list of lists of symmetric triples
            faces: list of faces for the planarity constraint. 

    Output: Loss 
    """
    xyz = np.vstack((xy.T, z)).T

    # Deviation from symmetry
    sym_dev = 0
    for triple_pair in sym_triples:
        t0 = triple_pair[0]
        t1 = triple_pair[1]
        angle0 = get_angle(xyz[t0[0], :], xyz[t0[1], :], xyz[t0[2], :])
        angle1 = get_angle(xyz[t1[0], :], xyz[t1[1], :], xyz[t1[2], :])
        sym_dev += np.abs(angle0 - angle1)

    sym_dev = sym_dev/(np.pi * len(sym_triples)) # scale such that symmetry deviation in [0,1]

    # Deviation from planarity
    plan_dev = 0
    for face in faces:
        n = len(face)
        M = np.pi*(n - 2)
        face_dev = M
        face_plus2 = face + face[:2]
        for i in range(n):
            angle = get_angle(xyz[face_plus2[i], :], xyz[face_plus2[i+1], :], xyz[face_plus2[i+2], :])
            face_dev -= angle
        plan_dev += face_dev/M

    plan_dev = plan_dev/len(faces) # scale so that planarity deviation in [0,1]

    # Compactness
    c = compactness(xyz)
    c = c/(1./216.) # scale so that compactness in [0,1] , cube has compactness 1/216 

    # Loss
    loss = sym_dev + w_plan*plan_dev - w_comp*c

    if components:
        return {"w_plan": w_plan, "w_comp":w_comp, "loss":loss, "sym_dev":sym_dev, "plan_dev":plan_dev, 
        "neg_comp":-1*c}
    else:
        return loss





#######################################################################
########      OPTIMIZATION          ###################################
#######################################################################

k = np.random.randint(0, len(everything['xyz']), 1).item()
xy = np.array(everything['xyz'][k])[:, :2]
faces = everything['faces'][k]
sym_triples = everything['sym_triples'][k]
triples = everything['triples'][k]

bounds = [[1,1.01]] + [[0, 10]]*(xy.shape[0]-1)
w_plan = 10
w_comp = 0
opt_result = optimize.shgo(lambda z: loss4(z, xy, sym_triples, faces, w_plan = w_plan, w_comp=w_comp), bounds)
print(opt_result)

components = loss4(opt_result['x'], xy, sym_triples, faces, components = True)
print(components)

xyz = np.vstack((xy.T, opt_result['x'])).T
sos(xyz, sym_triples, faces, triples)

print(compactness(xyz), 1/216)


plot3Ds(xyz, everything['pairs'][k])














