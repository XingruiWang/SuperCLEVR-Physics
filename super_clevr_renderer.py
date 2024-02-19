# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=logging-fstring-interpolation
# see: https://docs.python.org/3/library/subprocess.html

# TODO:
# 1. Watertight change(stage 1)
# 2. Super-clevr render & json output(for every frame)
# 3. camera to pytorch3d
# 4. dataloader 
import os
print(os.getcwd())
import argparse
import json
import sys
import logging
from pathlib import Path
import shutil
import subprocess
import tarfile
import bmesh
from typing import Tuple

# from trimesh_utils import get_object_properties
# import trimesh_utils
from urdf_template import URDF_TEMPLATE
import os, random, math

# kubrics

_DEFAULT_LOGGER = logging.getLogger(__name__)
INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.") 
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()
# super clevr
parser.add_argument('--properties_json', default='data/properties.json',
      help="JSON file defining objects, materials, sizes, and colors. " +
                "The \"colors\" field maps from CLEVR color names to RGB values; " +
                "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                "rescale object models; the \"materials\" and \"shapes\" fields map " +
                "from CLEVR material and shape names to .blend files in the " +
                "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
      help="Directory where .obj files for object models are stored")
parser.add_argument('--model_dir', default='data/save_models_1/',
        help="Directory where .blend files for object models are stored")
parser.add_argument('--filename_prefix', default='superCLEVR',
      help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
      help="Name of the split for which we are rendering. This will be added to " +
                "the names of rendered images, and will also be stored in the JSON " +
                "scene structure for each image.")
parser.add_argument('--output_dir', default='../output',
      help="The directory where output images, jsons and blender files will be stored." + 
      "It will be created if it does not exist.")

parser.add_argument('--save_blendfiles', type=int, default=0,
      help="Setting --save_blendfiles 1 will cause the blender scene file for " +
                "each generated image to be stored in the directory specified by " +
                "the --output_blend_dir flag. These files are not saved by default " +
                "because they take up ~5-10MB each.")  

# super-clevr-kubric dataset
parser.add_argument('--convert', default=0, type=int,
        help="Make the .blend file to kubric object and save it")
parser.add_argument('--convert_dir', default='../output/convert',
        help="Make the .blend file to kubric object and save it")
parser.add_argument('--fix_watertight', default=0, type=int,
        help="Make the .blend file to watertight and save it")
parser.add_argument('--make_obj', default=0, type=int,
        help="Make the .blend file to .obj and save it")
parser.add_argument('--make_glb', default=0, type=int,
        help="Make the .obj file to .glb and save it")
parser.add_argument('--make_kubrics', default=0, type=int,
        help="Make the kubric files and save it")

# kubric render
parser.add_argument('--num_obj_min', default=0, type=int,
        help="Make the kubric files and save it")
parser.add_argument('--num_obj_max', default=0, type=int,
        help="Make the kubric files and save it")
parser.add_argument('--iteration', type=int)


# others
parser.add_argument('--use_gpu', default=0, type=int,
        help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                 "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                 "to work.")


def main(args):
    # TODO: read data; fix watertight; conert to kubric; add objects; sim; render; output image & anno
    global color_name_to_rgba, size_mapping, material_mapping, obj_info
    # Load the property file
    color_name_to_rgba, size_mapping, material_mapping, obj_info = utils.load_properties_json(args.properties_json, os.path.join(args.shape_dir, 'labels'))
    
    images_dir = args.output_dir + '/images/'
    scenes_dir = args.output_dir + '/scenes/'
    blendfile_dir = args.output_dir + '/blendfiles/'

    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    img_template = os.path.join(images_dir, img_template)
    scene_template = os.path.join(scenes_dir, scene_template)
    blend_template = os.path.join(blendfile_dir, blend_template)

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    if not os.path.isdir(scenes_dir):
        os.makedirs(scenes_dir)
    if args.save_blendfiles == 1 and not os.path.isdir(blendfile_dir):
        os.makedirs(blendfile_dir)

    # convert .blend file to kubric
    if args.convert:
        convert(args)
    
    # run sim to render
    for i in range(args.iteration):
        cmd = 'docker run --runtime=nvidia --rm --interactive \
                --user $(id -u):$(id -g)        \
                --gpus device=0                     \
                --env KUBRIC_USE_GPU=1          \
                --volume "$(pwd):/kubric"       \
                kubricdockerhub/kubruntu        \
                python3 sim_render.py            \
                --data_dir=output/convert       \
                --output_dir=output/convert       \
                --iteration={}                  \
                --camera=fixed'.format(i)
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # run_sim()
  
    return

def fix_watertight(item, blender_path, output_dir, logger=_DEFAULT_LOGGER):
    # TODO: If failed, then jump
    output_path = output_dir + item
    
    if os.path.isdir(output_path):
        logger.debug('{} have exsisted'.format(item))
        return
    bpy.ops.wm.open_mainfile(filepath=blender_path)

    # Check if there are any non-manifold edges left
    remaining_non_manifold_edges = bpy.context.active_object.data.total_edge_sel
    if remaining_non_manifold_edges == 0 & os.path.isdir(output_path):
        logger.debug("Mesh is watertight.")
    else:
        logger.debug("Mesh has {} remaining non-manifold edges.".format(remaining_non_manifold_edges))

    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()

    mesh = bpy.context.active_object.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    for edge in boundary_edges:
        edge.select = True

    bpy.ops.mesh.edge_face_add()
    bpy.ops.object.mode_set(mode='OBJECT')
    bm.to_mesh(mesh)
    bm.free()

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()

    bpy.ops.wm.save_as_mainfile(filepath=output_path)

def blend2obj(item, watertight_path, output_dir, logger=_DEFAULT_LOGGER):
    output_path = output_dir + os.path.splitext(item)[0] + '.obj'
    bpy.ops.wm.open_mainfile(filepath=watertight_path)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        obj.select = True

    # Export to .obj
    bpy.ops.export_scene.obj(
    filepath=output_path,
    use_selection=True,
    use_materials=False,  
    use_triangles=True    
    )
    return

def cleanup_mesh(asset_id: str, source_path: str, target_path: str):
    '''
    For Blender < 2.79b, you need to install the glTF-Blender-IO: https://github.com/KhronosGroup/glTF-Blender-Exporter
    You can follow this file to install:https://github.com/KhronosGroup/glTF-Blender-Exporter/tree/master/scripts.

    '''
    # import pdb; pdb.set_trace()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")

    # import pdb; pdb.set_trace()
    bpy.ops.wm.addon_enable(module='io_scene_gltf2') 
    bpy.ops.import_scene.gltf(filepath=source_path, loglevel=50)
    bpy.ops.object.select_all(action='DESELECT')

    for obj in bpy.data.objects:
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=1e-06)
        bpy.ops.object.mode_set(mode='OBJECT')

        obj.data.use_auto_smooth = False
        # split edges with an angle above 70 degrees (1.22 radians)
        m = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
        m.split_angle = 1.22173
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")
        m = obj.modifiers.new("Displace", "DISPLACE")
        m.strength = 0.00001
        bpy.ops.object.modifier_apply(modifier="Displace")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()

    bpy.context.active_object.name = asset_id

    bpy.ops.export_scene.gltf(filepath=str(target_path), check_existing=True)


def convert(args, logger=_DEFAULT_LOGGER):
    # TODO: to convert the .blend to the urdf_tem
    # get objects watertight
    watertight_dir = args.convert_dir + '/watertight/'
    glb_dir = args.convert_dir + '/glbs/'
    obj_dir = args.convert_dir + '/objs/'
    obj_collision_dir = args.convert_dir + '/obj_collisions/'
    logs_dir = args.convert_dir + '/logs/'

    if args.fix_watertight:
        if not os.path.isdir(watertight_dir):
            os.makedirs(watertight_dir)
        for item in os.listdir(args.model_dir):
            blender_path = os.path.join(args.model_dir, item)
            if os.path.splitext(blender_path)[1] != '.blend':
                continue
            fix_watertight(item, blender_path, output_dir=watertight_dir)

    if args.make_obj:
        if not os.path.isdir(obj_dir):
            os.makedirs(obj_dir)
        for item in os.listdir(watertight_dir):
            watertight_path = os.path.join(watertight_dir, item)
            blend2obj(item, watertight_path, output_dir=obj_dir)
        logger.debug('making obj files done')

    if args.make_glb:
        # import pdb; pdb.set_trace()
        if not os.path.isdir(glb_dir):
            os.makedirs(glb_dir)
        for item in os.listdir(obj_dir):
            obj_path = os.path.join(obj_dir, item)
            glb_path = glb_dir + os.path.splitext(item)[0] + '.glb'
            cmd = 'obj2gltf -i {} -o {}'.format(obj_path, glb_path)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.debug('making glb files done')
    
    if args.make_kubrics:
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        if not os.path.isdir(obj_collision_dir):
            os.makedirs(obj_collision_dir)
        for item in os.listdir(obj_dir):
            asset_id = os.path.splitext(item)[0]
            # make collision obj
            obj_path = os.path.join(obj_dir, item)
            glb_path = glb_dir + asset_id + '.glb'
            glb_new_path = glb_dir + 'visual_' + asset_id + '.glb'
            collision_path = obj_collision_dir + 'collision_' + item
            watertight_path = watertight_dir + item
            log_path_0 = logs_dir + 'collision_stdout_' + asset_id + '.txt'
            
            cmd = 'python pybullet_vhacd.py --source_path={} --target_path={} --stdout_path={}'.format(obj_path, collision_path, log_path_0)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # make visual_geometry
            asset_id = os.path.splitext(item)[0]
            cleanup_mesh(asset_id, glb_path, glb_new_path)

            # make urdf file
            cmd = 'python urdf_make.py --asset_id={} --collision_obj={} --watertight_obj={} --visual_glb={} --output={}'.format(asset_id, collision_path, obj_path, glb_new_path, args.convert_dir)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.debug('make kubric file done')

    # sim and render
    cmd = 'docker run --rm --interactive \
            --user $(id -u):$(id -g)    \
            --volume "$(pwd):/kubric"   \
            kubricdockerhub/kubruntu    \
            python sim_render.py        \
            --asset_id={}          \
            --data_dir=output/convert   \
            --camera=clevr'.format()
    return


def render_scene(args,
        num_objects=5,
        output_index=0,
        output_split='none',
        output_image='render.png',
        output_scene='render_json',
        output_blendfile=None,
    ):
    # TODO: something about the render

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES" #BLENDER_RENDER, CYCLES
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
            'split': output_split,
            'image_index': output_index,
            'image_filename': os.path.basename(output_image),
            'objects': [],
            'directions': {},
    }
    

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    # scene_struct['relationships'] = compute_all_relationships(scene_struct) #relationship
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)



def add_random_objects(scene_struct, num_objects, args, camera):
    print('adding', num_objects, 'objects.')
    # num_objects = 5
    """
    Add random objects to the current blender scene
    """
    import pdb; pdb.set_trace()
    positions = []
    objects = []
    blender_objects = []
    obj_pointer = []
    for i in range(num_objects):
        # Choose a random size
        size_name, r = random.choice(size_mapping)

        # Choose random color and shape
        
        obj_name, obj_pth = random.choice(list(obj_info['info_pth'].items()))
        # obj_name, obj_pth = "suv", "car/473dd606c5ef340638805e546aa28d99"
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
        
        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args, camera)
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Choose random orientation for the object.
            theta = 360.0 * random.random()
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            
            def dist_map(x,y,t):
                theta = t / 180. * math.pi
                dx1 = x * math.cos(theta) - y * math.sin(theta)
                dy1 = x * math.sin(theta) + y * math.cos(theta)
                dx2 = x * math.cos(theta) + y * math.sin(theta)
                dy2 = x * math.sin(theta) - y * math.cos(theta)
                return dx1, dy1, dx2, dy2
            
            def ccw(A,B,C):
                # return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

            # Return true if line segments AB and CD intersect
            def intersect(A,B,C,D):
                return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
            
            
            def check(xx,yy,box_xx,box_yy,rr,tt,x,y,box_x,box_y,r,theta):
                xx1, yy1, xx2, yy2 = dist_map(box_xx/2*rr, box_yy/2*rr, tt)
                AA = (xx+xx1, yy+yy1)
                BB = (xx+xx2, yy+yy2)
                CC = (xx-xx1, yy-yy1)
                DD = (xx-xx2, yy-yy2)
                x1, y1, x2, y2 = dist_map(box_x/2*r, box_y/2*r, theta)
                A = (x+x1, y+y1)
                B = (x+x2, y+y2)
                C = (x-x1, y-y1)
                D = (x-x2, y-y2)
                for (p1, p2) in [(AA, BB), (BB, CC), (CC, DD), (DD, AA), (AA, CC), (BB, DD)]:
                    for (p3, p4) in [(A, B), (B, C), (C, D), (D, A), (A, C), (B, D)]:
                        if intersect(p1, p2, p3, p4):
                            return True
                return False
                
                
            for (objobj, xx, yy, rr, tt) in positions:
                box_x, box_y, _ = obj_info['info_box'][obj_name]
                box_xx, box_yy, _ = obj_info['info_box'][objobj]
                if check(xx,yy,box_xx,box_yy,rr*1.1,tt,x,y,box_x,box_y,r*1.1,theta):
                    margins_good = False
                    break

            if dists_good and margins_good:
                break


        # Actually add the object to the scene
        loc = (x, y, -r*obj_info['info_z'][obj_name])
        current_obj = utils.add_object(args.model_dir, obj_name, obj_pth, r, loc, theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((obj_name, x, y, r, theta))

        # Attach a random color
        # rgba=(1,0,0,1)
        mat_name, mat_name_out = random.choice(material_mapping)
        utils.modify_color(current_obj, material_name=mat_name, mat_list=obj_info['info_material'][obj_name], color=rgba)
        

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name,
            'size': size_name,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
            'material': mat_name_out
        })
        
        obj_pointer.append(current_obj)


    # Check that all objects are at least partially visible in the rendered image
    # TODO: rewrite this part
    all_visible, visible_parts = check_visibility(blender_objects, args.min_pixels_per_object, args.min_pixels_per_part, is_part=True, obj_info=obj_info)

    print('check vis done')
    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)

    for i in range(num_objects):
        # randomize part material
        
        current_obj = obj_pointer[i]
        obj_name = current_obj.name.split('_')[0]
        color_name = objects[i]['color']
        part_list = visible_parts[current_obj.name]
        part_names = random.sample(part_list, min(3, len(part_list)))
        # part_name = random.choice(obj_info['info_part'][obj_name])
        part_record = {}
        for part_name in part_names:
            while True:
                part_color_name, part_rgba = random.choice(list(color_name_to_rgba.items()))
                if part_color_name != color_name:
                    break
            part_name = part_name.split('.')[0]
            if part_name not in obj_info['info_part_labels'][obj_name]:
                print(part_name, obj_name, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                continue
            part_verts_idxs = obj_info['info_part_labels'][obj_name][part_name]
            mat_name, mat_name_out = random.choice(material_mapping)
            utils.modify_part_color(current_obj, part_name, part_verts_idxs, mat_list=obj_info['info_material'][obj_name], 
                                    material_name=mat_name, color_name=part_color_name, color=part_rgba)
            part_record[part_name] = {
                    "color": part_color_name,
                    "material": mat_name_out,
                    "size": objects[i]['size']
                    }
                
        objects[i]['parts'] = part_record

    return objects, blender_objects

def check_visibility():
    # TODO: check the visivility
    return

if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()


