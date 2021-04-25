import bpy, bmesh
import numpy as np
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
import random
import pickle
#import plotly.graph_objects as go
# import os
# os.chdir("users/mark/documents/uci/research/files")
# exec(open('Stimuli_v8.py').read())

#https://blender.stackexchange.com/questions/61879/create-mesh-then-add-vertices-to-it-in-python

################################################################
########## GENERATE OBJECTS    #################################
################################################################

class cuboid:
    """
    Input: depth range = range of depths at which inital plane can be sampled from. 
           obj_label = numeric label  indicating which object shape
           view_label = multiple views of the same object, so this indicates the view 
           verts = 3 x 8 numpy array of vertex locations. (optional)
    Output: cuboid object with Vertices, Pairs, Faces, Adj fields

    Makes a cuboid. Steps:
    1. An arbitrary convex quadrilateral is generated on a plane 
    parallel to XY plane,with random depth. 
    2. Four more points are added where, (x,y,z) -> (x,y,-z), to 
    give mirror symmetrical object about plane Z = 0. 
    3. Faces and Adjacency matrix is constant out of this function. 
    """

    def __init__(self, obj_label, view_label, verts=None, plane = None, m=1, M=2):
        # Identification
        self.uid = "Object {}, View {}".format(obj_label, view_label)
        self.label = obj_label

        # Vertices
        if verts is not None:
            formatted_verts = []
            for j in range(8):
                formatted_verts.append(tuple(verts[:, j]))
            self.verts = formatted_verts
        else:
            self.verts = []
            z = (m + (M-m) * np.random.rand())/2
            r = (m + (M-m) * np.random.rand())
            for i in range(4):
                theta = np.pi/2*i + np.pi/2 * (.15 + .7 * np.random.rand())
                self.verts.append((r*np.cos(theta), r*np.sin(theta), z))
                self.verts.append((r*np.cos(theta), r*np.sin(theta), -z))

        # Pairs
        self.pairs = [(0,1), (2,3), (4,5), (6,7), (0,2), (2,4), (4,6), 
                      (6, 0), (1,3), (3,5), (5,7), (7,1) ]

        # Faces
        self.faces = [[0,2,4,6], [1,3,5,7], [2,4,5,3], 
                      [0,2,3,1], [0,6,7,1], [6,4,5,7]]

        # Triples, 3 triples per vertex on a cube, therefore 24 total
        self.triples = [(1,0,2), (1,0,6), (2,0,6),
                        (0,1,7), (0,1,3), (3,1,7), 
                        (0,2,3), (0,2,4), (3,2,4),  
                        (2,3,5), (1,3,5), (2,3,1),  
                        (2,4,5), (2,4,6), (5,4,6),  
                        (3,5,7), (3,5,4), (4,5,7), 
                        (4,6,7), (4,6,0), (0,6,7), 
                        (1,7,5), (1,7,6), (5,7,6)]

        # symmetric pairs
        self.sym_pairs = [(0,1), (2,3), (4,5), (6,7)]


        # Symmetric triples - every angle has it's symmetric match. So 12 matches! Oof. 
        self.sym_triples = [[[1,0,2], [0,1,3]], #1
                            [[1,0,6], [0,1,7]], #2
                            [[2,0,6], [3,1,7]], #3
                            [[0,2,3], [1,3,2]], #4
                            [[0,2,4], [1,3,5]], #5
                            [[3,2,4], [2,3,5]], #6
                            [[2,4,5], [3,5,4]],
                            [[2,4,6], [3,5,7]],
                            [[5,4,6], [4,5,7]],
                            [[0,6,7], [1,7,6]],
                            [[1,7,5], [0,6,4]],
                            [[5,7,6], [4,6,7]]]


        # Plane
        if plane is not None:
            self.plane = plane
        else:
            self.plane = np.array([0,0,1])

        #visible quantities
        self.vvid = None
        self.vis_faces = None
        self.vis_pairs = None
        self.vis_verts = None
        self.vis_sym_pairs = None
        self.visible_triples = None
        self.vis_symmetric_triples = None



    def show(self):
        xyz = np.vstack(self.verts)
        X = xyz[:, 0]
        Y = xyz[:, 1]
        Z = xyz[:, 2]
            
        trace1 = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            text = [str(i) for i in range(len(X))],
            mode='markers+text',
            name='Vertices'
        )

        x_lines = list()
        y_lines = list()
        z_lines = list()

        #create the coordinate list for the lines
        for p in self.pairs:
            for i in range(2):
                x_lines.append(X[p[i]])
                y_lines.append(Y[p[i]])
                z_lines.append(Z[p[i]])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        trace2 = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            name='Edges'
        )

        fig = go.Figure(data=[trace1, trace2])
        fig.show()


def rotx(theta):
    theta = np.radians(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([[1,0,0], [0,c,-s],[0,s,c]])
    return M

def roty(theta):
    theta = np.radians(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([[c,0,s], [0,1,0],[-s,0,c]])
    return M

def rotz(theta):
    theta = np.radians(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([[c,-s,0], [s,c,0],[0,0,1]])
    return M


def make_cuboids(num_objects, num_views):
    """
    Makes list of cuboid objects
    Input: num_objects = number of unique objects, 
           num_views = Number of views of each of the num_objects objects
    Output: list of cuboid objects of length num_objects * num_views
    """
    objects = []
    for i in range(num_objects):
        c = cuboid(obj_label = i, view_label = 0)
        objects.append(c)
        verts = np.vstack(c.verts).T
        for j in range(1, num_views):
            R = rotx(180*np.random.rand()) @ roty(180*np.random.rand()) @ rotz(180*np.random.rand())
            new_cuboid = cuboid(obj_label = i, view_label = j, verts = R @ verts, plane = R @ c.plane)
            objects.append(new_cuboid)
    return objects




################################################################
####### GET VISIBLE VERTICES AND SURFACES IN BLENDER   #########
################################################################


def view(cuboid):
    """
    Input: Cuboid object 
    Result: Cuboid object shown in blender 
    """
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    mesh = bpy.data.meshes.new("me") 
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(cuboid.verts, [], cuboid.faces)


def BVHTreeAndVerticesInWorldFromObj(obj):
    """
    Input: Object of Blender type Object
    Output: BVH Tree necessary for ray tracing and vertsInWorld = verts in global coordinate system. 
    """
    mWorld = obj.matrix_world
    vertsInWorld = [mWorld @ v.co for v in obj.data.vertices]
    bvh = BVHTree.FromPolygons( vertsInWorld, [p.vertices for p in obj.data.polygons] )
    return bvh, vertsInWorld

def vvid_to_inds(vvid):
    translation_dict = {}
    for i, vertid in enumerate(vvid):
        translation_dict[vertid] = i
    return translation_dict

def get_visible_faces(bm, vvid):
    new_faces = []
    translation_dict = vvid_to_inds(vvid)
    for face in bm.faces:
        face_verts = []
        for v in face.verts:
            try:
                face_verts += [translation_dict[v.index]]
            except KeyError:
                break
        else:
            new_faces += [face_verts]
    return new_faces

def get_vis_triples(cube):
    vis_triples = []
    translation_dict = vvid_to_inds(cube.vvid)
    for triple in cube.triples:
        try:
            vis_triple = [translation_dict[t] for t in triple]
            vis_triples += [vis_triple]
        except KeyError:
            continue
    return vis_triples


def get_vis_sym_pairs(cube):
    vis_sym_pairs = []
    translation_dict = vvid_to_inds(cube.vvid)
    for pair in cube.sym_pairs:
        try:
            p = tuple([translation_dict[t] for t in pair])
            vis_sym_pairs += [p]
        except KeyError:
            continue
    return vis_sym_pairs


def get_vis_sym_triples(cube):
    vst = []
    translation_dict = vvid_to_inds(cube.vvid)
    for triple_pair in cube.sym_triples:
        t0 = triple_pair[0]
        t1 = triple_pair[1]
        try:
            t0v = [translation_dict[t] for t in t0]
            t1v = [translation_dict[t] for t in t1]
            vst += [[t0v, t1v]]
        except KeyError:
            continue
    return vst



def get_visible_edges(bm, vvid):
    new_edges = []
    translation_dict = vvid_to_inds(vvid)
    for edge in bm.edges:
        index0 = edge.verts[0].index
        index1 = edge.verts[1].index
        try:
            new_edges += [(translation_dict[index0], translation_dict[index1])]
        except KeyError:
            continue

    return new_edges


def get_pv(cam, vverts):
    """
    Input: Camera and visible vertices
    Output: List of visible vertices in camera coordinate system
    """
    camI = cam.rotation_euler.to_matrix().copy()
    camI.invert()
    from_cam_to_vert = [Vector(v) - cam.location for v in vverts]
    new_verts = [camI @ v for v in from_cam_to_vert]
    return [[v.x, v.y, v.z] for v in new_verts]

def edit_plane(cam, p):
    """
    Input: camera and current plane 
    Output: Plane in camera coordinate system 
    """
    p = np.hstack((p, 0))
    R = cam.rotation_euler.to_matrix().copy()
    T = np.expand_dims(cam.location, axis = 0).T
    C = np.block([[R, T],
                  [np.zeros(3), 1]])

    return p @ C



def getVisibleVertices(obj, cam, scene, limit = .1):    
    # In world coordinates, get a bvh tree and vertices
    bvh, vertices = BVHTreeAndVerticesInWorldFromObj( obj )
    visible_vertices = []
    visible_vertices_id = []
    #projected_verts = []
    for i, v in enumerate( vertices ):
        # Get the 2D projection of the vertex
        co2D = world_to_camera_view( scene, cam, v )
        obj.data.vertices[i].select = False
        # If inside the camera view
        if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0: 
            #print(i,"this vertex is inside camera view")
            # Try a ray cast, in order to test the vertex visibility from the camera
            location, normal, index, distance = bvh.ray_cast( cam.location, (v - cam.location).normalized() )
            mindists = []
            loc = None
            for _ in range(1000):
                direction_vector = ((v - cam.location).normalized() + Vector((random.gauss(0,.01),random.gauss(0,.01),random.gauss(0,.01)))).normalized()
                location, normal, index, distance = bvh.ray_cast( cam.location, direction_vector )
                if location:
                    #break
                    mindists.append((v - location).length)
            # print("cam.location, (v - cam.location).normalized() , location, normal, index, distance")
            # print(cam.location, (v - cam.location).normalized() , location, normal, index, distance)
            # # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex

            #if location and (v - location).length < limit:
            if mindists and min(mindists) < limit:# and (v - location).length < limit:
                obj.data.vertices[i].select = True
                visible_vertices.append([v.x,v.y,v.z])
                #projected_verts.append([co2D.x, co2D.y, co2D.z])
                visible_vertices_id.append(i)

        #print("\n\n")        
    #print("#verts:", NUMVERTS, " #visible vertices:", len(visible_vertices))
    del bvh
    #print("visible vertices:",visible_vertices)
    return visible_vertices, visible_vertices_id




def get_visible_quantities(cuboids):
    """
    Updates cuboids list to include data on which vertices are visible. 

    Input: list of cuboids
    Output: cuboids, instead updates cuboids objects in input cuboids list. 
    """

    for cube in cuboids:

        # delete existing objects
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()

        # load cuboid to scene
        mesh = bpy.data.meshes.new("me") 
        obj = bpy.data.objects.new(mesh.name, mesh)
        col = bpy.data.collections.get("Collection")
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        mesh.from_pydata(cube.verts, [], cube.faces)
        scene = bpy.context.scene
        cam = bpy.context.scene.camera

        # get visible vertices 
        visible_vertices, vvid = getVisibleVertices(obj, cam, scene)
        cube.vis_verts = get_pv(cam, visible_vertices)
        cube.vvid = vvid

        # load object data into bmesh type on which some computations are easier
        bm = bmesh.new()
        bm.from_mesh(obj.data)

        # get visible faces
        cube.vis_faces = get_visible_faces(bm, vvid)

        # get visible pairs or adjacency matrix 
        cube.vis_pairs = get_visible_edges(bm, vvid)

        #modify plane
        cube.plane = edit_plane(cam, cube.plane)

        # get visible triples
        cube.visible_triples = get_vis_triples(cube)
        cube.vis_sym_pairs = get_vis_sym_pairs(cube)
        cube.vis_symmetric_triples = get_vis_sym_triples(cube)

       
    return cuboids



################################################################
##########                WRITE TO FILE           ##############
################################################################

def save(cuboids, fname):
    """
    Save the important quantities to pickle
    (uid, label, plane, vis_faces, vis_pairs, vis_verts)
    """
    uids = []
    labels = []
    planes = []
    vis_faces_lst = []
    vis_pairs_lst = []
    vis_verts_lst = []
    visible_triples = []
    visible_symmetric_triples = []
    visible_symmetric_pairs = []
    dct = {}
    for cube in cuboids:
        uids.append(cube.uid)
        labels.append(cube.label)
        planes.append(cube.plane)
        vis_faces_lst.append(cube.vis_faces)
        vis_pairs_lst.append(cube.vis_pairs)
        vis_verts_lst.append(cube.vis_verts)
        visible_triples.append(cube.visible_triples)
        visible_symmetric_pairs.append(cube.vis_sym_pairs)
        visible_symmetric_triples.append(cube.vis_symmetric_triples)

    dct["uid"] = uids
    dct["label"] = labels
    dct["plane"] = planes
    dct["faces"] = vis_faces_lst
    dct["pairs"] = vis_pairs_lst
    dct["xyz"] = vis_verts_lst
    dct["sym_pairs"] = visible_symmetric_pairs
    dct["sym_triples"] = visible_symmetric_triples
    dct["triples"] = visible_triples

    with open(fname, 'wb') as f:
        pickle.dump(dct, f, protocol=pickle.HIGHEST_PROTOCOL)



################################################################
##########                RUN           ########################
################################################################
cuboids = make_cuboids(num_objects = 500, num_views = 7)

#view(cuboids[0])
cuboids = get_visible_quantities(cuboids)
cuboids7 = [c for c in cuboids if len(c.vis_verts) == 7]
save(cuboids7, "./data/cuboid7.pickle")







