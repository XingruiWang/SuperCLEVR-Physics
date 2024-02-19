import pymeshlab
import numpy as np
import os


meshs_path = '/home/shuo/kubric_exp/output/convert/objs/'
removed_mesh_path = '/home/shuo/kubric_exp/output/convert/objs_remove/'

if not os.path.exists(removed_mesh_path):
    os.makedirs(removed_mesh_path)

for cate in os.listdir(meshs_path):

    source_path = meshs_path + cate
    # source_path = 'CAD/%s/'
    save_path = removed_mesh_path + cate


    configs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                [0.7, 0.7, 0.1], [0.7, -0.7, 0.1], [-0.7, 0.7, 0.1], [-0.7, -0.7, 0.1]]

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(source_path)

    for config in configs:
        ms.compute_scalar_ambient_occlusion(conedir=np.array(config), coneangle=45, usegpu=True)
    ms.select_by_vertex_quality(minq=0, maxq=0.1)
    # ms.invert_selection(invfaces=True, invverts=True)
    ms.delete_selected_faces_and_vertices()

    ms.save_current_mesh(save_path)