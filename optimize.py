from scipy import optimize
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from preprocess import get_face_mask, get_triple_mask, get_edge_matrix, get_triples,make_symmetry_mask,gmp
from preprocess import get_symmetry_values, get_planarity_values


def compactness(xyz):
    """
    Input: xyz 
    Output: compactness (V^2/SA^3) of convex hull of 3D points. 
    """
    xyz = np.array(xyz)
    ch = ConvexHull(xyz, qhull_options="QJ")
    return ch.volume**2/ch.area**3


def get_gram(xyz, M):
    """
    input: (xyz) n x 3 or n x 2 numpy ndarray, (M): matrix to compute edges from xyz
    output: gram matrix, square ndarray
    """
    edges = M @ xyz
    edges = normalize(edges, axis=1) # [42 x 3]
    gram = edges @ edges.T # [42 x 42]
    gram = np.clip(gram, -1., 1.) # [5 x 42 x 42]
    return gram

def get_angles_gram_mask(gram, mask):
    """
    Input: (gram) square numpy array, (mask) square numpy array where 
    1 = select, 0 = do not select
    Output: (angles) numpy array or angles in mask in degrees
    """
    angles = gram * mask
    angles = angles[angles != 0] 
    angles = np.degrees(np.arccos(angles))
    return angles


def slant_angle_error(theta, z, xyzR, mpl):
    L = 0
    for mp in mpl:
        x0 = xyzR[mp[0], 0]
        z0 = xyzR[mp[0], 2] + z
        x1 = xyzR[mp[1], 0]
        z1 = xyzR[mp[1], 2] + z
        est_z0 = (x1 - x0 * np.cos(2*theta))/np.sin(2*theta)
        est_z1 = (x0 - x1 * np.cos(2*theta))/np.sin(2*theta)
        L += (z0 - est_z0)**2 + (z1 - est_z1)**2
    return L

# ### Add true slant angle of object to everything dataset
# with open('./data/everything.pickle', 'rb') as f:
#     everything = pickle.load(f)
# eps = 1e-8
# bounds_theta = [[0 + eps, np.pi/2 - eps], [-20,20]]
# tsa = []
# for k in range(len(everything['uid'])):
#     xyzR = everything["xyz_rotated"][k]
#     mpl = everything["mpl"][k]
#     if len(mpl) == 3:
#         opt_results = minimize(lambda x: slant_angle_error(x[0], x[1], xyzR, mpl), 
#                                 bounds=bounds_theta, 
#                                 x0 = [1, 12])
#         tsa += [opt_results]
#     else:
#         tsa += [np.NaN]

# everything["true_slant_angle"] = tsa

# with open('./data/everything.pickle', 'wb') as handle:
#     pickle.dump(everything, handle, protocol=pickle.HIGHEST_PROTOCOL)



def loss1(z, xy, triple_mask, face_mask, M, ps, sym_mask = None, both=False): 
    """
    SDA + Planarity
    Optimize over z (depth) values
    """
    xyz = np.vstack((xy.T, z)).T

    # make gram matrix
    gram = get_gram(xyz, M) 

    # select angles for SDA
    angles_SDA = get_angles_gram_mask(gram, triple_mask)
    MSDA = np.std(angles_SDA)

    # Select angles for planarity
    face_mask = (face_mask > 0).astype(int)
    angles_planarity = get_angles_gram_mask(gram, face_mask)
    Planarity = np.abs(angles_planarity.sum() - ps)

    if both:  
        results = {'SDA':MSDA, "planarity":Planarity} 

        if sym_mask is not None:
            sym_mask = sym_mask.numpy()
            sym = 0
            sym_max = int(np.max(sym_mask))
            corresponding_angles = np.zeros((sym_max, 2))
            for sm in range(sym_max):
                A = gram * (sym_mask == sm+1).astype(int)
                cos_angles = A[A != 0]
                angles_S = np.degrees(np.arccos(cos_angles))
                #print("A = ", angles_S, "\n")
                corresponding_angles[sm, :] = angles_S
                sym += np.abs(angles_S[0] - angles_S[1])
            results['symmetry'] = sym/(sym_max + 1e-8)
            results["symmetry_angles"] = corresponding_angles

        return results
    else:
        return MSDA + Planarity


def loss2(z, xy, sym_mask, face_mask, ps, M, both=False):
    """
    Symmetry + Planarity 
    Optimize over z (depth) values
    """
    sym_mask = sym_mask.unsqueeze(0)
    face_mask = face_mask.unsqueeze(0)
    ps = ps.unsqueeze(0)

    xyz = np.vstack((xy.T, z)).T

    # make gram matrix
    gram = get_gram(xyz, M) 
    gram = torch.tensor(gram).unsqueeze(0)

    # get planarity and symmetry values
    symmetry = get_symmetry_values(1, sym_mask, gram)
    planarity = get_planarity_values(1, face_mask, gram, ps)

    loss = symmetry + planarity #+ z[0]**2

    if both:
        results = {}
        results["symmetry"] = symmetry.squeeze().tolist()
        results["planarity"] = planarity.squeeze().tolist()
        return results
    else:
        return loss.squeeze().numpy()




def loss3(z, xy, sym_mask, face_mask, ps, M, components = False):
    """
    Symmetry + Compactness + Planarity
    """
    sym_mask = sym_mask.unsqueeze(0)
    ps = ps.unsqueeze(0)
    face_mask = face_mask.unsqueeze(0)
    eps = 1e-16
    z = z + eps*np.random.randn(7)
    xyz = np.vstack((xy.T, z)).T

    # get compactness
    C = compactness(xyz)

    # make gram matrix
    gram = get_gram(xyz, M) 
    gram = torch.tensor(gram).unsqueeze(0)

    # get planarity and symmetry values
    symmetry = get_symmetry_values(1, sym_mask, gram)
    planarity = get_planarity_values(1, face_mask, gram, ps)
    #C = 1 * C
    loss = symmetry - 10* C + planarity.item() #+ (z[0] - 1)**2
    if components:
        return {"symmetry":symmetry.item(), "compactness":C, 
            "planarity":planarity.item(), "z0":z[0], "loss":loss.item()}
    else:
        return loss.item()





def loss4(theta, xy, faces, pairs, get_xyz=False):
    """
    Planarity & Symmetry assumed, optimize over slant angle theta, 
    with -1*compactness as the loss
    """
    n = xy.shape[0]
    z = np.zeros(n)
    idx = list(range(n))
    for i,j in pairs:
        x1 = xy[i, 0]
        x2 = xy[j, 0]
        idx.pop(idx.index(i))
        idx.pop(idx.index(j))
        #print(x1, x2)
        z1 = (x2 - x1 * np.cos(2*theta))/np.sin(2*theta)
        z2 = (x1 - x2 * np.cos(2*theta))/np.sin(2*theta)
        z[i] = z1
        z[j] = z2

    #print("faces = ", faces)
    xyz = np.vstack((xy.T,z)).T
    #print("idx = ", idx)
    # now use planarity to find z value of missing vertex
    for i in idx:
        for face in faces:
            if i in face:
                # make plane from three points in face. 
                triangle = [x for x in face if x != i] # list of three indexes
                #print("triangle = ", triangle)
                p1 = xyz[triangle[0], :]
                p2 = xyz[triangle[1], :]
                p3 = xyz[triangle[2], :]

                # These two vectors are in the plane
                v1 = p3 - p1
                v2 = p2 - p1

                # the cross product is a vector normal to the plane
                cp = np.cross(v1, v2)
                a, b, c = cp

                # This evaluates a * x3 + b * y3 + c * z3 which equals d
                d = np.dot(cp, p3)
                missingz = (d - a*xyz[i, 0] - b*xyz[i, 1])/c
                xyz[i, 2] = missingz


    C = compactness(xyz)
    if get_xyz:
        return {"xyz": xyz, "compactness": C}  
    else:
        return -1 * C


# #### UPDATE EVERYTHING DATASET to include loss4results & loss4xyz ##### 
# with open('./data/everything.pickle', 'rb') as handle:
#     everything = pickle.load(handle)

# loss4results = []
# loss4xyz = []
# n = len(everything["uid"])
# eps = 1e-8
# for k in range(n):
#   xy = np.array(everything["xyz_rotated"][k])
#   faces = everything["faces"][k]
#   mpl = everything["mpl"][k]
#   if xy.shape == (7,3) and len(faces) == 3 and len(mpl) == 3:
#     xy = xy[:, :2]
#     results = minimize(lambda alpha: loss4(alpha, xy, faces, mpl), x0 = 1, bounds = [[eps, np.pi/2 - eps]])
#     loss4results += [results]
#     temp = loss4(results['x'], xy, faces, mpl, get_xyz=True)
#     loss4xyz += [temp['xyz']]

#   else:
#     loss4results += [np.NaN]
#     loss4xyz += [np.NaN]

# everything["loss4results"] = loss4results
# everything['loss4xyz'] = loss4xyz

# with open('./data/everything.pickle', 'wb') as handle:
#     pickle.dump(everything, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loss5(theta, xy, pairs, get_xyz=False):
    """
    Symmetry assumed, compactness over single tetrahedron. 
    """
    n = xy.shape[0]
    z = np.zeros(n)
    idx = list(range(n))
    for i,j in pairs:
        x1 = xy[i, 0]
        x2 = xy[j, 0]
        idx.pop(idx.index(i))
        idx.pop(idx.index(j))
        #print(x1, x2)
        z1 = (x2 - x1 * np.cos(2*theta))/np.sin(2*theta)
        z2 = (x1 - x2 * np.cos(2*theta))/np.sin(2*theta)
        z[i] = z1
        z[j] = z2

    #print("faces = ", faces)
    xyz = np.vstack((xy.T,z)).T
    select4 = pairs[0] + [pairs[1][0]] + [pairs[2][0]]
    tetrahedron = xyz[select4, :]
    C = compactness(tetrahedron)

    if get_xyz:
        return {"xyz": xyz, "compactness": C}  
    else:
        return -1 * C

# #### UPDATE EVERYTHING DATASET to include loss5results  ##### 
# with open('./data/everything.pickle', 'rb') as handle:
#     everything = pickle.load(handle)

# loss5results = []
# n = len(everything["uid"])
# eps = 1e-8
# for k in range(n):
#   xy = np.array(everything["xyz_rotated"][k])
#   faces = everything["faces"][k]
#   mpl = everything["mpl"][k]
#   if xy.shape == (7,3) and len(faces) == 3 and len(mpl) == 3:
#     xy = xy[:, :2]
#     results = minimize(lambda alpha: loss5(alpha, xy, mpl), x0 = 1, bounds = [[eps, np.pi/2 - eps]])
#     loss5results += [results]
#   else:
#     loss5results += [np.NaN]

# everything["loss5results"] = loss5results

# with open('./data/everything.pickle', 'wb') as handle:
#     pickle.dump(everything, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loss5_torchM(theta, xy, X_mask_tetrahedron, Z_mask_tetrahedron, ns, M, get_xyz=False):
    """
    Symmetry assumed, compactness over single tetrahedron. Modified to be fully in pytorch. 
    Inputs: theta (radians)
            xy (7 x 2), we only care about x, as the assumption is that matched pairs have the same y-value
            pairs_mask ()
            M (42 x 7), matrix needed to compute z
            ns (network size, in this case, 7)
    
    """ 
    b = xy.shape[0]
    a = 1/torch.cos(2*theta)
    b = -torch.sin(2*theta)/torch.cos(2*theta)
    angle = torch.tensor([a, b]) # [b x 2 x 1] - this one's suspect
    x_col1 = torch.index_select(xy, 2, torch.tensor([0])).repeat_interleave(ns-1, dim = 1)
    x_col2 = torch.matmul(M, torch.index_select(xy, 2, torch.tensor([0])))
    xx = torch.cat([x_col1, x_col2], dim = 2)
    Z = torch.matmul(xx, angle) # [b x ns(ns - 1) x 2] @ [2 x 1] = [b x ns(ns - 1) x 1]
    print(Z.shape, Z)
    # xyz concatenate xy to pairs_mask @ z = [b  x ns x ns(ns-1)] @ [b x ns(ns-1) x 1] = [b x ns x 1]
    # concatenate to [b x ns x 2] = [b x ns x 3] = xyz
    # select 4 xyz using tetra_select, then compute Surface area and volume, then compactness, to return
    Z4 = torch.matmul(Z_mask_tetrahedron, Z)
    XY4 = torch.matmul(X_mask_tetrahedron, xy) # [b  x ns x ns(ns-1)] @ [b x ns(ns-1) x 1] = [b x ns x 1]
    # for z_mask_tetrahedron, always select first pair of symmetric pairs, then first element of subsequent 
    # 2 symmetric pairs. 
    print(Z_ns.shape, Z_ns)

    xyz4 = torch.cat([XY4, Z4], dim = 2)
    print(xyz4, xyz4.shape)

    # get surface area




    #print("faces = ", faces)
    # xyz = np.vstack((xy.T,z)).T
    # select4 = pairs[0] + [pairs[1][0]] + [pairs[2][0]]
    # tetrahedron = xyz[select4, :]
    # C = compactness(tetrahedron)

    # if get_xyz:
    #     return {"xyz": xyz, "compactness": C}  
    # else:
    #     return -1 * C
def loss5_torch(theta, xy, pairs, get_xyz=False):
    """
    Symmetry assumed, compactness over single tetrahedron. 
    """
    n = xy.shape[0]
    z = torch.zeros(n)
    idx = list(range(n))
    for i,j in pairs:
        x1 = xy[i, 0]
        x2 = xy[j, 0]
        idx.pop(idx.index(i))
        idx.pop(idx.index(j))
        #print(x1, x2)
        z1 = (x2 - x1 * torch.cos(2*theta))/torch.sin(2*theta)
        z2 = (x1 - x2 * torch.cos(2*theta))/torch.sin(2*theta)
        z[i] = z1
        z[j] = z2

    #print("faces = ", faces)
    xyz = torch.vstack((xy.T,z)).T
    select4 = pairs[0] + [pairs[1][0]] + [pairs[2][0]]
    tetrahedron = xyz[select4, :]

    #Surface area of tetrahedron
    SA = 0
    for combo in torch.combinations(torch.arange(4), 3):
        #print(combo)
        #xyz3 = tetrahedron[combo, :]
        v1 = tetrahedron[combo[0], :] - tetrahedron[combo[1], :]
        v2 = tetrahedron[combo[2], :] - tetrahedron[combo[1], :]
        A = .5 * torch.linalg.norm(torch.cross(v1, v2))
        SA += A

    # Volume of Tetrahedron
    v1 = tetrahedron[1, :] - tetrahedron[0, :]
    v2 = tetrahedron[2, :] - tetrahedron[0, :]
    v3 = tetrahedron[3, :] - tetrahedron[0, :]
    V = 1/6 * torch.linalg.norm(torch.dot(v1, torch.cross(v2, v3)))

    C = V**2/SA**3

    if get_xyz:
        return {"xyz": xyz, "compactness": C}
    else:
        return -1 * C



############################################################
################### TESTS ##################################
############################################################



# with open('./data/everything.pickle', 'rb') as handle:
#     everything = pickle.load(handle)
# #print(everything.keys())



# k = np.random.randint(0, len(everything["uid"]), 1).item()
# k = 4
# pairs = everything["pairs"][k]
# faces = everything["faces"][k]
# plane = everything["plane"][k]
# mpl = everything["mpl"][k]
# xyz = np.array(everything["xyz"][k])
# xyzR = np.array(everything["xyz_rotated"][k])
# xy = xyz[:, :2]
# xyR = xyzR[:, :2]
# z = xyz[:, 2]
# zR = xyz[:, 2]
# triples = get_triples(pairs, faces, face_triples = False)
# ps = sum(map(lambda x: 180 * (len(x) - 2), faces))
# M = get_edge_matrix(7)
# triple_mask = get_triple_mask(pairs, faces, 7, face_triples=False)
# face_mask = get_face_mask(faces, 7)
# mp1, mp2 = gmp(xyz, triples, plane)
# sym_mask = make_symmetry_mask(mp1, mp2, 7)


# print("Object {} selected".format(k))
# print("loss1(sample xyz) = ", loss1(z, xy, triple_mask.numpy(), face_mask.numpy(), 
#     M, ps, sym_mask = None, both=True), "\n")

# print("loss2(sample xyz) = ", loss2(z, xy, sym_mask, face_mask, torch.tensor(ps), M, both=True), "\n")

# print("loss3(sample xyz) = ", loss3(z, xy, sym_mask, face_mask, torch.tensor(ps), M, components = True), "\n")

# print("loss4(sample xyz) = ", loss4(np.radians(22), xyR, faces, mpl, get_xyz=False))

# print("loss5(sample xyz) = ", loss5(np.radians(22), xyR, mpl, get_xyz=False))

# print("loss5_torch(sample xyz) = ", loss5_torch(torch.deg2rad(torch.tensor(22.)), torch.tensor(xyR), mpl, get_xyz=False).item())

