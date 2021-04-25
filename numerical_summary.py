import numpy as np
import pickle
import pandas as pd
from tabulate import tabulate


def dfs_to_markdown(dfs):
	for key in dfs.keys():
		if key == "FACES":
			for face_string in dfs["FACES"].keys():
				blob = tabulate(dfs["FACES"][face_string], headers='keys', tablefmt='pipe')
				blob = blob.split("\n")
				print("\n", face_string)
				for line in blob:
					print(line)
		else:
			blob = tabulate(dfs[key], headers='keys', tablefmt='pipe')
			blob = blob.split("\n")
			print("\n", key)
			for line in blob:
				print(line)

def get_angle(p1, p2, p3):
	"""
	Given 3 (x,y,z) 3x1 tensors p1, p2, p3, computes angle between p1 - p2 and p3 - p2
	using the dot product cosine rule. 
	"""
	e1 = p1 - p2
	e2 = p3 - p2
	x = np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2))
	return np.degrees(np.arccos(x))

def get_symmetry_df(xyz, sym):
	"""
	inputs: xyz = numpy n x 3 , sym from matched_points_less, list of lists of triples. 
	output: symmetry dataframe
	"""
	angle = []
	angle_match = []
	n = len(sym)
	for i in range(n):
		t0 = sym[i][0]
		t1 = sym[i][1]
		angle += [get_angle(xyz[t0[0], :], xyz[t0[1], :], xyz[t0[2], :])]
		angle_match += [get_angle(xyz[t1[0], :], xyz[t1[1], :], xyz[t1[2], :])]
	diffs = [angle[i] - angle_match[i] for i in range(n)]
	sym_df = pd.DataFrame([[sym[k][0] for k in range(n)], [sym[k][1] for k in range(n)], angle, angle_match, diffs],
		index = ["triple", "match", "triple.angle", "match.angle", "diff"]).T
	return sym_df

def get_angle_df(xyz, triples):
	angle = []
	for t in triples:
		angle += [get_angle(xyz[t[0], :], xyz[t[1], :], xyz[t[2], :])]
	angle_df = pd.DataFrame([triples, angle],
		index = ["triple", "angle"]).T
	return angle_df

def get_face_dfs(xyz, faces):
	face_dfs = {}
	for face in faces:
		n = len(face)
		face_angles = []
		triples = []
		for i in range(n):
			i1 = face[i % n]
			i2 = face[(i+1) % n]
			i3 = face[(i + 2) % n]
			face_angles += [get_angle(xyz[i1, :], xyz[i2, :], xyz[i3, :])]
			triples += [str([i1,i2, i3])]
		face_df = pd.DataFrame(face_angles, index = triples, columns = ["angle"])
		new_index = list(face_df.index)+['Total']
		colsums = pd.DataFrame(face_df.sum()).T
		face_df = face_df.append(colsums)
		face_df.index = new_index
		face_dfs[str(face).replace(" ", "")] = face_df

	return face_dfs

def sos(xyz, sym, faces, triples, print_all = True, return_dfs = False):
	"""
	Inputs: xyz = xyz (numpy n x 3), sym = sample from matched_points_less or matched_points_more, 
	faces = faces (list)
	Outputs: symmetry df, faces dfs, overall df
	"""

	dfs = {}
	dfs["XYZ"] = pd.DataFrame(xyz, columns = ["X", "Y", "Z"])
	dfs["ANGLES"] = get_angle_df(xyz, triples)
	dfs["FACES"] = get_face_dfs(xyz, faces)
	dfs["SYMMETRY"] = get_symmetry_df(xyz, sym)

	if print_all:
		dfs_to_markdown(dfs)
	if return_dfs:
		return dfs


############################################################
################### TESTS ##################################
############################################################


with open('./data/everything.pickle', 'rb') as handle:
    everything = pickle.load(handle)


k = np.random.randint(0, len(everything["uid"]), 1).item()
def testsos(k, xyz_key = "xyz"):
	xyz = np.array(everything[xyz_key][k])
	sym = everything["matched_points_less"][k]
	triples = everything["triples_less"][k]
	faces = everything['faces'][k]

	sos(xyz, sym, faces, triples)

#testsos(k)


