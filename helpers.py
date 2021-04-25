"""
helpers.py contains a number of functions that are helpful for all the different loss functions 
"""

import pickle
import numpy as np
import pandas as pd
import torch
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


def triple2mask_loc(triple, ns):
	"""
	Inputs: Triple, e.g. [1,2,3], network size, which dictates the size of the mask
	Output: Location of angle corresponding to triple in an angle mask. e.g. (1,2)
	"""

def plot3Ds(xyz, pairs=None):
    """
    Inputs: xyz, pairs 
    Output: 3D plotly plot of points & optionally, pairs
    """    
    # convert to numpy array
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().numpy()
    if isinstance(xyz, list):
        xyz = np.array(xyz)
    
    plot_data = []
    
    # plot points
    plot_data += [go.Scatter3d(
    x=xyz[:, 0],
    y=xyz[:, 1],
    z=xyz[:, 2],
    mode='markers+text',
    text = [str(i) for i in range(xyz.shape[0])],
    name="xyz",
    marker=dict(color='#1f77b4'))]

    # plot edges if pairs argument provided
    if pairs: 
        x_lines = []
        y_lines = []
        z_lines = []

        for p in pairs:
            for i in range(2):
                x_lines.append(xyz[p[i], 0])
                y_lines.append(xyz[p[i], 1])
                z_lines.append(xyz[p[i], 2])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        plot_data += [go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            name="xyz",
            marker=dict(color='#1f77b4')
        )]
    
    
    fig = go.Figure(data = plot_data)
    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()







##############################################################################################

# Copy cuboid7.pickle into everything2.pickle and make sure NOT to overwrite everything.pickle.
# Shouldn't have to run this again.  
# with open('./data/cuboid7.pickle', 'rb') as handle:
#     c7 = pickle.load(handle)

# with open('./data/everything2.pickle', 'wb') as handle:
#     pickle.dump(c7, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################################




