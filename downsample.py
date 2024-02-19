# need bpy python=3.7 with "pip install bpy==2.91a0 && bpy_post_install"
import bpy
import os
import bmesh
import numpy as np

meshs_path = '/home/shuo/kubric_exp/output/convert/obj_collisions/'
remesh_path = '/home/shuo/kubric_exp/output/convert/objs_downsample'

if not os.path.exists(remesh_path):
    os.makedirs(remesh_path)


idx = 0
for item in os.listdir(meshs_path):
    # import pdb; pdb.set_trace()
    instance_path = os.path.join(meshs_path, item)
    instance_new_path = os.path.join(remesh_path, item)

    bpy.ops.import_scene.obj(filepath=instance_path, filter_glob="*.obj")
    print('import done')

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
    print('using GPU done')

    # Grab original mesh.
    orig_mesh = bpy.context.selected_objects[0]
    print('orig', orig_mesh)
    print('get mesh done')

    bpy.context.scene.objects.active = orig_mesh
    print('active done')
    

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    cube = bpy.data.objects['Cube']
    cube.select = False

    print('select done')

    bpy.ops.object.join()

    bpy.ops.object.select_all(action='DESELECT')
    orig_mesh.select = True
    #import pdb; pdb.set_trace()

    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply remesh modifier.
    bpy.ops.object.modifier_add(type='REMESH')
    print('add remesh done')

    # adjust the remesh settings as desired
    remesh = orig_mesh.modifiers["Remesh"]

    remesh.mode = 'SMOOTH'  # can also be 'SMOOTH' or 'SHARP'
    remesh.octree_depth = 6  # resolution of the mesh, increase for more detail
    remesh.sharpness = 1  # how much to preserve sharp corners, 1 is max
    remesh.use_remove_disconnected = False

    # apply the remesh modifier
    bpy.ops.object.modifier_apply(modifier="Remesh")

    # # add decimate modifier
    # bpy.ops.object.modifier_add(type='DECIMATE')

    # # adjust decimate settings as desired
    # decimate = orig_mesh.modifiers["Decimate"]

    # decimate.ratio = 0.01  # reduce the vertex count to 10%

    # # apply decimate modifier
    # bpy.ops.object.modifier_apply(modifier="Decimate")

    new_mesh = bpy.context.selected_objects[0]
    print('new', new_mesh)

    bpy.ops.export_scene.obj(filepath=instance_new_path, use_selection=True)
    print('export remesh done')

    bpy.ops.object.delete()

