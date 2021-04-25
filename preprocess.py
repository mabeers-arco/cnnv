import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
import collections
import math
import sys
from itertools import combinations
import time
import datetime
from tabulate import tabulate
import plotly.graph_objects as go



def visualize_gram(n, gram=None):
    """
    Inputs: n = gram matrix will have size n(n-1)/2 and will be filled with random values
            gram = gram matrix itself to visualize
    Output: Pandas dataframe visualization of the gram matrix 
    """
    if not gram:
        N = int(n*(n-1))
        gram = np.random.random((N,N))
        gram = np.round(gram.T @ gram,2)

    edges = []
    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            edges += ["V{}-V{}".format(i+dilation,i)]
            i += 1

    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            edges += ["V{}-V{}".format(i,i+dilation)]
            i += 1

    df = pd.DataFrame(gram, index = edges, columns = edges)
    return df



def in_pair_out_row_index(n, intersection, non_intersection):
    """
    Helper for in_triple_out_xy
    Inputs: n = number of vertices
            intersection = A triple has the form [a,b,c], where b is the vertex at which
            the line segments ab and ac make the angle abc. intersection is index b. 
            non_intersection = The index a or c

    Output: row index in a gram matrix. 

    Concerns: I think this function is going to break when n gets larger than 7. 
    """
    pair = np.array([non_intersection, intersection])
    row = 0
    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            if np.all((i+dilation, i) == pair):
                return row
            i+= 1
            row += 1

    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            if np.all((i, i+dilation) == pair):
                return row
            i+= 1
            row += 1

def in_triple_out_xy(n, triple):
    """
    Input: Triple e.g. [0,1,5] meaning vertex the angle formed with 1 at the intersection, 
    with 0 and 5 at the other points. 
    Output: (x, y) location of [0,1,5] in the gram matrix. 

    I need this function to make a mask
    """
    x = in_pair_out_row_index(n, intersection=triple[1], non_intersection=triple[0])
    y = in_pair_out_row_index(n, intersection=triple[1], non_intersection=triple[2])
    return x,y



def get_edge_matrix(n):
    """
    Input: integer n describing the size of the network or number of visible vertices
    Output: M, n(n-1)/2 x 3 matrix containing instructions on how to compute all 
    possible edges. 

    If U and V are xyz vectors, then M contains instruction to take U-V and V-U. 
    """
    M1 = torch.zeros((int(n*(n-1)/2), n))
    row = 0
    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            M1[row, i] = -1
            M1[row, i + dilation] = 1
            i += 1
            row += 1

    M2 = torch.zeros((int(n*(n-1)/2), n))
    row = 0
    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            M2[row, i] = 1
            M2[row, i + dilation] = -1
            i += 1
            row += 1

    return torch.vstack((M1, M2))

def unmask(mask, value=1):
    """
    Input: mask that is a square pytorch tensor.
    Output: list of triples
    """
    xy = torch.vstack(torch.where(mask == value)).numpy()
    N = mask.shape[0]
    n = np.int(np.ceil(np.sqrt(N))) # n * (n-1)
    triples = []

    edges = []
    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            edges += [[i+dilation,i]]
            i += 1

    for dilation in range(1,n):
        i = 0
        while i + dilation < n:
            edges += [[i,i+dilation]]
            i += 1

    for i in range(xy.shape[1]):
        x = xy[0,i]
        y = xy[1,i]
        edge1 = edges[x]
        edge2 = edges[y]
        triples += [edge1 + [edge2[0]]]

    return triples

def get_xy(xyz, network_size):
    """
    Input: xyz in list type (num visible vertices x 3)
    Output: Pytorch Tensor that is zero padded XY (2 x network_size) 
    """
    xyz = torch.tensor(xyz)
    pad = network_size - xyz.shape[0]
    M = nn.ZeroPad2d((0, 0, 0, pad))
    return M(xyz[:, :2])

def get_z(xyz, network_size):
    """
    Inputs: xyz in some type, will be converted to tensor & Network Size
    Output: padded tensor of the form z + [0]*n such that the total tensor has length network size
    """
    xyz = torch.tensor(xyz)
    pad = network_size - xyz.shape[0]
    z = torch.hstack((xyz[:, 2], torch.zeros(pad)))
    return z


def rotz(theta):
    """
    Input: Theta in radians
    Output: 3D rotation matrix about z axis 
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    M = np.array([[ct, -st, 0], 
                [st, ct, 0], 
                [0,0,1]])
    return M

def rotate_xyz(xyz, mp1, mp2):
    """
    Inputs: xyz, matched points 1 (mp1), matched points 2 (mp2)
    Output: rotated xyz
    Rotate xyz so that symmetrical points have the same y value. This is so that 
    neural network has to learn one fewer step in the reconstruction process. 
    i.e. Z = f(x, angle of symmetry plane) if y's for symmetric pairs are equal. 
    """

    xyz = np.array(xyz)
    xyzC = xyz - np.mean(xyz, axis = 0)
    i = mp1[0][1]
    j = mp2[0][1]
    xy1 = xyzC[i, :2]
    xy2 = xyzC[j, :2]
    h = np.array([1,0])
    v = xy2 - xy1
    theta = np.arccos(np.dot(h,v)/np.linalg.norm(v))
    if v[1] > 0:
        M = rotz(-theta)
    else:
        M = rotz(theta)
    xyzR = (M @ xyzC.T).T + np.mean(xyz, axis = 0)
    return xyzR

def conn_original_to_pairs(conn):
    """
    Input: conn of the form [0,0,0,1,0,... ] where 1 indicates presence of edge. 
    This format has been taken deprecated in favor of the adjacency matrix
    """
    N = math.ceil(math.sqrt(len(conn)*2))
    pairs = []
    count = 0
    for delta in range(1, N):
        start = 0
        while start + delta < N:
            index1 = start
            index2 = start + delta
            # print(index1, index2)
            # print("search for:", index1,index2)
            if conn[count] == 1:
            #print(index1, index2, "found")
                pairs.append((index1, index2))
            start += 1
            count += 1

    return pairs


def get_adj(pairs, network_size):
    """
    Input: original list format connection matrix
    Output: pytorch matrix (network_size x network_size), In D^(-1/2) @ A @ D^(-1/2)
    """
    n = np.max(np.vstack(pairs)) + 1
    pad = network_size - n
    A = torch.zeros(network_size,network_size)
    for i in range(network_size):
        for j in range(network_size):
            if (i,j) in pairs or (j,i) in pairs:
                A[i,j] = torch.tensor(1.)

    A += torch.eye(network_size)
    D = torch.diag(torch.sum(A, 1)**(-.5))
    DAD = D @ A @ D

    return DAD

    


def ff_get_triples(edge_list_one, edge_list_two):
    """
    Inputs: Edge lists one and two = list of all edges
    Output: list of triples constructed from all possible edges. 
    """
    triples = []
    for e1 in edge_list_one:
        for e2 in edge_list_two:
            if e1 == e2 or e1 == e2[::-1]:
                continue
            intersection = list(set(e1) & set(e2))
            if intersection:
                intersection = intersection[0]
                triple = [e1[abs(e1.index(intersection)-1)], 
                intersection, 
                e2[abs(e2.index(intersection)-1)]]
                if triple not in triples and triple[::-1] not in triples:
                    triples += [triple]
    return triples

def ff_get_edges(face):
    """
    Input: Face = list of vertices in a face 
    Output: decompose all possible edges in face into real edges and diagonals. 
    """
    edges = [(face[i], face[i+1]) for i in range(len(face)-1)]
    edges += [(face[-1], face[0])]
    diagonals = []
    for combo in combinations(face,2):
        if combo not in edges and combo[::-1] not in edges:
            diagonals += [combo]
    return edges, diagonals



def get_triples(pairs, faces, face_triples=False):
    """
    Inputs: pairs = list of edges in format [(1,2), (3,4), ...] meaning there's an edge between vertex 1&2
    and 3&4. faces = list of faces. face_triples = boolean to include angles formed only by real edges or 
    also to include angles formed by real edges and diagonals. 
    Output: list of triples 
    """
    triples = []
    #pairs = conn_original_to_pairs(conn)
    M = max([max(p) for p in pairs])

    shared = {}
    for shared_index in range(M+1):
        we_share_a_vertex = []
        for pair in pairs:
            if shared_index == pair[0]:
                we_share_a_vertex.append(pair[1])
            elif shared_index == pair[1]:
                we_share_a_vertex.append(pair[0])

        shared[shared_index] = we_share_a_vertex

    for key in shared.keys():
        for ends in combinations(shared[key], 2):
            triples.append([ends[0], key, ends[1]])
    #triples.append(expand([ends[0], key, ends[1]], network_size))
    #print(shared)

    if face_triples:
        face_diags = []
        for face in faces:
            real, diag = ff_get_edges(face)
            face_diags += ff_get_triples(real, diag)
        triples += face_diags

    #pad = max_angles - len(triples)
    #triples2 = torch.cat([torch.tensor(triples), torch.zeros(pad,3)], dim = 0)
    return triples
  
# test_triples = get_triples(data_dct["conn"][k], data_dct["vis_faces"][k], face_triples=True)
# print(test_triples)


def get_triple_mask(conn, faces, network_size, face_triples):
    n = int(network_size*(network_size-1))
    mask = torch.zeros((n, n))
    triples = get_triples(conn, faces, face_triples)
    for triple in triples:
        x,y = in_triple_out_xy(network_size, triple)
        mask[x,y] = 1. 
    return mask

def get_tmask2(triples, network_size):
    n = int(network_size*(network_size-1))
    mask = torch.zeros((n, n))
    for triple in triples:
        x,y = in_triple_out_xy(network_size, triple)
        mask[x,y] = 1. 
    return mask

from collections import Counter

def format_faces(vis_faces, max_face_size, max_faces):
    faces = -1 * torch.ones((max_faces, max_face_size), dtype = torch.int8)
    for i in range(len(vis_faces)):
        N = len(vis_faces[i])
        face = vis_faces[i] + vis_faces[i][:2]
        faces[i, :N+2] = torch.tensor(face, dtype=torch.int8)
    return faces

#format_faces(data_dct["vis_faces"][k], 8, 8)


def get_face_mask(vis_faces, network_size):
    """
    Inputs: vis_faces = list of faces, network_size
    Output: Mask indicating which angles correspond to a face. Face mask has size n(n-1)/2 square and
    is equal to zero everywhere but at the angles corresponding to a face. If three quadtrilateral 
    faces visible, mask all zeros except 4 ones, 4 twos and 4 threes. 
    """
    n = int(network_size * (network_size -1))
    mask = torch.zeros((n,n))
    for i in range(len(vis_faces)):
        face = vis_faces[i] + vis_faces[i][:2]
        for j in range(len(vis_faces[i])):
            triple = face[j:j+3]
            x,y = in_triple_out_xy(network_size, triple)
            mask[x,y] = i+1.

    return mask

def get_planarity_sum(vis_faces):
    """
    input: vis_faces = list of visible faces
    output: number indicating sum of internal angles of the faces. pi*(n-2)
    """

    p = 0
    for face in vis_faces:
        p += (len(face) - 2)*180
    return p

def get_sda(xyz, M, triple_mask):
    """
    Input: (xyz) in list type, (M) = get_edge_matrix, (triple_mask) = mask of 0's and 1's
    Output: SDA of angles specified in triple_mask in degrees
    """
    xyz = torch.tensor(xyz)
    edges = torch.matmul(M, xyz) 
    edges = F.normalize(edges, dim=1) # [42 x 3]
    gram = torch.matmul(edges, edges.T) # [42 x 42]
    gram = torch.clamp(gram, -1., 1.) # [42 x 42]
    angles = torch.masked_select(gram, triple_mask > 0.)
    angles = torch.rad2deg(torch.arccos(angles))
    return torch.std(angles)


def gmp(xyz, triples, plane):
    """
    get matched points
    This function desperately needs a rewrite. 

    Input: xyz, triples, plane
    Output: 2 lists, one = subset of input triples, two = corresponding matches
    Output example: 
    out1 = [[3, 1, 2], [1, 3, 5], [1, 3, 4], [5, 3, 4], [3, 5, 6]] 
    out2 = [[4, 2, 1], [2, 4, 6], [2, 4, 3], [6, 4, 3], [4, 6, 5]]
    """
    expanded_triples = []
    for t in triples:
        expanded_triples += [t] + [[t[2], t[1], t[0]]]
    triples = expanded_triples

    out1 = []
    out2 = []
    S = np.sqrt(np.sum(plane[:3]**2))
    for t in triples:
        P = np.vstack((xyz[t[0]], xyz[t[1]], xyz[t[2]]))
        P = np.hstack((P, np.array([[1,1,1]]).T))
        dists = P @ plane/S
        flip_vector = (-2 * np.tile(plane[:3]/np.linalg.norm(plane[:3]), (3,1)).T @ np.diag(dists)).T
        newP = P[:, :3] + flip_vector
        for t1 in triples:
            test_points = np.vstack((xyz[t1[0]], xyz[t1[1]], xyz[t1[2]]))
            if np.all(np.isclose(newP, test_points)):
                reversedT = [t[2], t[1], t[0]]
                if t not in out1 and reversedT not in out1 and t not in out2 and reversedT not in out2:
                    out1 += [t]
                    out2 += [t1]

    return out1, out2

def get_symmetry_values(b, sym_mask, gram):
    """
    Inputs: b = number of batches, sym_mask = symmetry_mask, gram = gram matrix
    Output: list of deviations from symmetry, angles selected in sym_mask 
    """
    symmetry = torch.zeros(b)
    for i in range(b):
        sym_max = torch.max(sym_mask[i, :, :]).int()
        for sm in range(sym_max):
            cos_angles_S = torch.masked_select(gram[i, :, :], sym_mask == sm+1)
            angles_S = torch.rad2deg(torch.arccos(cos_angles_S))
            #print(angles_S[0] , angles_S[1])
            symmetry[i] += torch.abs(angles_S[0] - angles_S[1])
    return symmetry


def get_planarity_values(b, face_mask, gram, ps):
    """
    Inputs: b = number of batches, face_mask = face mask, gram = gram matrix, ps = total internal angles
    of object, usually 1080
    Output: List of deviations from planarity for each object 
    """
    planarity = torch.zeros(b)
    P = torch.sum(face_mask > 0, dim=(1,2))
    P = torch.cumsum(P, dim = 0)
    P = torch.hstack((torch.tensor(0), P)).int()
    cos_angles_P = torch.masked_select(gram, face_mask > 0.)
    angles_P = torch.rad2deg(torch.arccos(cos_angles_P))
    for i in range(b):
        planarity[i] = torch.abs(ps[i] - torch.sum(angles_P[P[i]:P[i+1]]))
    return planarity



def make_symmetry_mask(triples1, triples2, network_size):
    """
    Input: triples 1, triples 2 = lists of triples of equal length where triples1[k] and triples2[k]
    should correspond to angles that should be the same. Network size also an input so mask size can
    be determined. 

    Output: Symmetry mask in pytorch tensor
    """
    n = int(network_size * (network_size -1)) 
    mask = torch.zeros((n,n))
    assert len(triples1) == len(triples2)
    i = 0
    for t1, t2 in zip(triples1, triples2):
        x,y = in_triple_out_xy(network_size, t1)
        mask[x,y] = i+1.
        x,y = in_triple_out_xy(network_size, t2)
        mask[x,y] = i+1.
        i += 1
    return mask

def get_M_xcol2(ns):
    """
    Input: ns (network_size, usually 7)
    Output: ns(ns-1) x ns tensor indicating which x to select
    """
    M = torch.zeros((ns * (ns - 1), ns))
    for i in range(ns):
        idx = list(range(ns))
        idx.pop(i)
        row = i*(ns-1)
        M[row:(row + ns - 1), :] = torch.eye(ns)[idx]
    return M



def get_X_and_Z_mask_tetrahedron(ns, mpl):
    """
    Input: Network size, mpl = matched points
    Output: Z-mask for tetrahedron = [4 x ns(ns-1)] = [4 x 42]
            X-mask for tetrahedron = 4 x 7
    
    suppose pairs are [[0,1], [3,4], [5,6]]
    X has structure 
    [[x0, x1],       = z1
    [x0, x2],        = Nonsense  
    ...
    [x0, x6],        = Nonsense
    [x2, x1],        = Nonsense
    ...
    [x3, x4],        = z4
    [x6, x4],        = Nonsense
    [x6, x5]]        = z5
    
    """
    get_pairs = []
    for i in range(ns):
        for j in range(ns):
            if i != j:
                get_pairs += [(i,j)]
    
    
    X_mask = torch.zeros((4,ns))
    Z_mask = torch.zeros((4, ns*(ns - 1)))
    
    # Pair 1
    i = mpl[0][0]
    j = mpl[0][1]
    X_mask[0, i] = 1.
    Z_mask[0, get_pairs.index((j,i))] = 1. 
    X_mask[1, j] = 1.
    Z_mask[1, get_pairs.index((i,j))] = 1. 
    
    # Pair 2
    i = mpl[1][0]
    j = mpl[1][1]
    X_mask[2, i] = 1.
    Z_mask[2, get_pairs.index((j,i))] = 1. 
    
    # Pair 3
    i = mpl[2][0]
    j = mpl[2][1]
    X_mask[3, i] = 1.
    Z_mask[3, get_pairs.index((j,i))] = 1. 

    return X_mask, Z_mask    


############################################################
################### TESTS ##################################
############################################################
def run_tests_c7():
    # Alternate dataset: 
    with open('./data/cuboid7.pickle', 'rb') as f:
        c7 = pickle.load(f)
    print(c7.keys())
    print(len(c7["uid"]))



    k = np.random.randint(0, len(c7["uid"]), 1).item()
    pairs = c7["pairs"][k]
    faces = c7["faces"][k]
    xyz = np.array(c7["xyz"][k])
    plane = c7["plane"][k]
    triples = get_triples(pairs, faces, face_triples = False)

    ## visualize gram
    print("-------------------------------")
    print("Visualize Gram")
    print("Triple [0,1,2] corresponds V2-V1 and V0-V1 (Final-Initial)")
    print(visualize_gram(3))

    ## gram matrix helpers 
    print("-------------------------------")
    print("In triple, out position of triple in gram matrix")
    print(in_triple_out_xy(3, [0,2,1]))

    ## Edge matrix
    print("-------------------------------")
    print("Matrix to compute edges from verts")
    print(get_edge_matrix(4))

    ## get_xy & get_z


    ## Rotate xyz
    print("-------------------------------")
    print("Rotate XYZ so that y-values constant")
    mp1, mp2 = gmp(xyz, triples, plane)
    xyzR = rotate_xyz(xyz, mp1, mp2)
    plt.scatter(xyz[:, 0], xyz[:, 1], label = 'xyz')
    plt.scatter(xyzR[:, 0], xyzR[:, 1], label = "xyzR")
    for i in range(7):
        plt.text(xyz[i, 0], xyz[i, 1], str(i))
        plt.text(xyzR[i, 0], xyzR[i, 1], str(i))
    plt.axis('equal');
    plt.legend()
    plt.show()

    ## Adjacency matrix
    print("-------------------------------")
    print("Adjacency matrix")
    print(get_adj(pairs, 8))


    ## Triples & Unmask
    print("-------------------------------")
    print("Compute triples mask, then unmask to confirm same triples")
    triple_mask=get_triple_mask(pairs, faces, 7, face_triples=False)
    print(np.vstack(unmask(triple_mask)))

    ## Faces
    print("-------------------------------")
    print("Faces")
    print("planarity sum = ", get_planarity_sum(faces))
    face_mask = get_face_mask(faces, 7)
    triple_mask_locs = np.vstack(np.where(triple_mask > 0)).T
    face_mask_locs = np.vstack(np.where(face_mask > 0)).T
    print("face_mask_locs = \n", face_mask_locs, "\n triple_mask_locs = \n",triple_mask_locs)
    print(Counter(face_mask.numpy().ravel()))


    ## Symmetry
    print("-------------------------------")
    print("Symmetry")
    sym_mask = make_symmetry_mask(mp1, mp2, 7)
    print(torch.vstack(torch.where(sym_mask > 0)))




#run_tests_c7()

def run_tests_everything():
    with open('./data/everything.pickle', 'rb') as f:
        everything = pickle.load(f)

