import sys
sys.path.insert(0, '/kubric')
import logging
# logging.basicConfig(level=logging.ERROR)
import argparse
import os
import json
import math
import pyquaternion
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np

from pathlib import Path
import bpy

from time import time

os.environ["KUBRIC_USE_GPU"] = "1"

with open("assets/all_objects_name.json") as f:
    All_objects_name = json.load(f)

no_smooth_cate = ['double']
def config():
    parser = kb.ArgumentParser()

    parser.set_defaults(
        frame_end = 10,
        resolution=(1280, 960),
        # frame_rate=60
        frame_rate=6
    )


    parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")

    parser.add_argument("--hdri_assets", type=str,
        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    
    parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
    
    parser.add_argument("--properties_cgpart", type=str,
                    default="assets/properties_cgpart.json")

    # scene parameters
    parser.add_argument("--background", choices=["colored", "hdri"], default="hdri")
    parser.add_argument("--num_obj_min", type=int, default=4)
    parser.add_argument("--num_obj_max", type=int, default=6)
    
    parser.add_argument("--scene_size", type=int, default=4)
    # parser.add_argument("--floor_friction", type=float, default=0.0)
    parser.add_argument("--obj_friction", type=float, default=0.2)
    parser.add_argument("--floor_friction", type=float, default=0.4)
    parser.add_argument("--floor_restitution", type=float, default=0.5)    
    parser.add_argument('--camera', type=str)

    # frames parameters
    # io
    parser.add_argument('--iteration', type=int, default=100)
    parser.add_argument('--data_dir', type=str)
    # parser.add_argument('--job_dir', type=str, default="output")
    # parser.add_argument('--scratch_dir', type=str, default="tmp")

    # debug
    parser.add_argument('--height', type=str, choices=["realistic", "random"], default='realistic')
    parser.add_argument('--skip_segmentation', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_scene', default="")


    args = parser.parse_args()

    if args.properties_cgpart:
        with open(args.properties_cgpart, "r") as f:
            args.properties_cgpart = json.load(f)

    return args

def setup_camera(scene, rng, args, load_camera_from_scene=None):

    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)

    if args.camera == "fixed":  # Specific position + jitter
         
        look_at = [0, 0, 1.3] + rng.rand(3)
        location = [8.0, -7.0, 3.2]
        
        scene.camera.position = location
        scene.camera.look_at(look_at)

    elif args.camera == "random":  # Random position in half-sphere-shell
        scene.camera.position = kb.sample_point_in_half_sphere_shell(
            inner_radius=7., outer_radius=9., offset=0.1
            )
        
        look_at = [0, 0, 1.3] + rng.rand(3)
        scene.camera.look_at(look_at)
    
    elif args.camera == "god":  # Random position + jitter
        # scene.camera.position = [0, 0, 80] + rng.rand(3)
        scene.camera.position = [7.48113, -6.50764,70.34367] 
        scene.camera.look_at((-50, 50, 0))
    
    if load_camera_from_scene:
        # replace with the loaded scene
        camera_info = load_camera_from_scene["camera"]
        location = camera_info['location']
        scene.camera.position = location
        
        if 'look_at' in camera_info:
            look_at = camera_info['look_at']
        else:
            look_at = [0, 0, 1.3]
        scene.camera.look_at(look_at)

    camera_info = kb.get_camera_info(scene.camera)
    camera_info['location'] = scene.camera.position
    camera_info['look_at'] = look_at
     
    return scene, camera_info

def add_walls(scene, renderer, background_hdri):

    # Define dimensions of the wall
    wall_width = 30
    wall_height = 10
    wall_depth = 0.1

    # Create a wall object
    # wall = kb.Cube(scale=(wall_width, wall_height, wall_depth), \
    #                position = (0, 0, -wall_depth/2))
    wall_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)
    wall_material.color = kb.Color.from_name("gray")

    left_wall = kb.Cube(scale=(wall_depth, wall_width, wall_height), \
                   position = (-wall_width // 2, 0, wall_height/2), static=True, background=True,
                   material=wall_material)
    right_wall = kb.Cube(scale=(wall_width, wall_depth, wall_height), \
                   position = (0, wall_width // 2, wall_height/2), static=True, background=True,
                    material=wall_material)
    
    
    # Add the wall to the scene
    scene += left_wall
    scene += right_wall

    return scene


def realistic_sizes(obj_name, sub_name):

    if obj_name == 'aeroplane':
        size = 3.2 if sub_name == 'fighter' else 3.4
    elif obj_name == 'bicycle':
        size = 1.5
    elif obj_name == 'car':
        size = 2.2
    elif obj_name == 'motorbike':
        size = 1.5
    elif obj_name == 'bus':
        size = 2.8 if sub_name == 'regular' else 3.2
    else:
        raise ValueError(f'Object name {obj_name} not found')
    return "realistic", size

def add_light(scene, rng, args, illumination=(0.05, 0.05, 0.05)):
    scene += kb.assets.utils.get_clevr_lights(rng=rng)  
    r, g, b = illumination
    scene.ambient_illumination = kb.Color(r, g, b)
    return scene


def add_floor(scene, kubasic, renderer, background_hdri, hdri_id, args, load_scene=None):
    floor_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)

    if args.background == "colored":
        floor_material.color = kb.random_hue_color()
        scene.metadata["background"] = floor_material.color.hexstr
    
    floor = kubasic.create("dome", name="floor", material=floor_material,
                        # scale=2.0,
                        scale=0.3,
                        friction=args.floor_friction,
                        restitution=args.floor_restitution,
                        static=True, background=True, position = (0, 0, 0))
    scene += floor

    if args.background == "hdri":
        scene.metadata["background"] = "hdri"
        scene.metadata["background_hdri_id"] = hdri_id
        
        set_texture(renderer, floor, background_hdri.filename)
    
    return scene

def set_texture(renderer, my_object, texture_path):

    my_object_blender = my_object.linked_objects[renderer]
    
    texture_node = my_object_blender.data.materials[0].node_tree.nodes
    links = my_object_blender.data.materials[0].node_tree.links

    image_texture_node = texture_node.new(type='ShaderNodeTexImage')
    image_texture_node.image = bpy.data.images.load(texture_path)
    
    principled_bsdf_node = texture_node.get("Principled BSDF")
    link = links.new(image_texture_node.outputs['Color'], principled_bsdf_node.inputs['Base Color'])

def add_obstacles(scene, renderer, rng, gso, args, N = 3):
    for i in range(N):
        # obstracle = gso.create(asset_id=rng.choice(active_split)) # 2_of_Jenga_Classic_Game
        obstracle = gso.create(asset_id="2_of_Jenga_Classic_Game") # 

        assert isinstance(obstracle, kb.FileBasedObject)
        
        scale = 3.0
        obstracle.scale = scale / np.max(obstracle.bounds[1] - obstracle.bounds[0])
        obstracle.metadata["scale"] = scale
        
        obstracle.static = True

        position = args.scene_size * rng.uniform(-0.8, 0.8, 3)
        position[2] = obstracle.position[2]
        
        obstracle.position = position

        scene += obstracle
    return scene

def add_objects(args, rng, scene, source_path, simulator, load_scene=None):
    if load_scene: 
        return load_objects(args, rng, scene, source_path, simulator, load_scene)
    else: 
        return add_random_objects(args, rng, scene, source_path, simulator)
    
def add_random_objects(args, rng, scene, source_path, simulator):

    objects = []

    # obj_lib = [f for f in os.listdir(args.data_dir + '/urdf') if f.endswith('.obj')]
    # obj_lib = [f for f in os.listdir(args.data_dir + '/urdf') if f.endswith('.obj')]
    obj_lib = [f + '.obj' for f in All_objects_name]

    num_objs = rng.randint(args.num_obj_min, args.num_obj_max)

    print('adding', num_objs, 'objects.')

    for i in range(num_objs):
        # Assign object's type (or name):
        static = bool(rng.choice([True, False, False]))
        if i == 0:
            obj_name = os.path.splitext(rng.choice([p for p in obj_lib if 'aeroplane' in p]))[0]
            static = False
        elif i == 1:
            obj_name = os.path.splitext(rng.choice([p for p in obj_lib if 'aeroplane' not in p]))[0]
            static = False
        elif i == 2:
            obj_name = os.path.splitext(rng.choice([p for p in obj_lib if 'utility' not in p]))[0]
            static = True
        else:
            obj_name = os.path.splitext(rng.choice(obj_lib))[0]


        cls_name, instance_name = obj_name.split('_')
        
        # Assign object's size:
        # The default way is following the realistic size: aeroplane > bus > car > motorbike = bicycle
        if args.height == 'realistic':
            size_label, size = realistic_sizes(obj_name.split('_')[0], obj_name.split('_')[1])
        else:
            size_label, size = kb.randomness.sample_sizes("super_clevr", rng)
        
        color_label, random_color = kb.randomness.sample_color("super_clevr", rng)
        # print(f"Create object: {cls_name} / {instance_name} / {color_label}")
        
        obj = kb.FileBasedObject(asset_id=obj_name,
                        render_filename= os.path.join(args.data_dir, "CGParts_colored", cls_name, instance_name, color_label, "object.obj"),
                        simulation_filename=args.data_dir+'/urdf/'+obj_name+'.urdf',
                        scale=size,
                        need_auto_smooth=instance_name not in no_smooth_cate
                        )

        obj.friction = args.obj_friction
        obj.restitution = 0.5
        # obj.mass *= 2.7 * size ** 3
        obj.mass *= 2.7

        # Random position
        
        if i == 0:
            static = False

        position = np.zeros(3, dtype=float)

        if not static:
            beta_samples = rng.beta(0.5, 0.5, 2)
            beta_samples = 2 * beta_samples - 1
            position[0], position[1] = args.scene_size * beta_samples
            
            if args.debug:
                position = np.array([-4, 0, 0], dtype=float)
        else:
            position[0], position[1] = args.scene_size * rng.uniform(-0.6, 0.6, 2)

        # Realistic height: only plane can be in the sky for initial position
        if args.height == 'fixed':
            position[2] = 1.0
        elif args.height == 'random':
            position[2] = args.scene_size * rng.uniform(-1, 1, 1)[0]
        elif args.height == 'realistic':
            if 'plane' in obj_name and not static:
                position[2] = args.scene_size * rng.uniform(0.4, 0.9, 1)[0]
            else:
                position[2] = 0
                # adjust the height of the object
                sub_obj_name = obj_name.split('_')[1]
                info_z = args.properties_cgpart['info_z'][sub_obj_name]
                position[2] -= (info_z * size)
                # import pdb; pdb.set_trace()

        
        obj.position = position

        # Random poses:
        # Only randomize the rotation around the z-axis, 
        # return `rand_degree` 
        # obj_quaternion, rand_degree = random_rotate_quaternion(args, rng, obj_name)
        obj_quaternion, rand_degree = facing_center_quaternion(args, rng, obj_name, position)
        obj.quaternion = obj_quaternion

        # Random velocity:
        # In the setting, the velocity is will be affected by: 1. fricion, 2. collision, 3. gravity. 
        #                                                      4. Add the engine force
        # speed = rng.choice([2, 5, 10]) # slow, medium, fast

        # Random speed: 0, 1, 3; static, slow, fast
        # For slow car, it won't accelerate above 5 m /s
        # For fast car, it won't accelerate to 10 m/s
        # speed = rng.choice([1, 3]) if not static else 0
        speed = rng.choice([3, 6]) if not static else 0


        rand_radians = np.radians(rand_degree)
        obj.velocity = np.array([math.cos(rand_radians), math.sin(rand_radians), 0.0]) * speed 
        
        if args.debug:
            obj.velocity = np.array([0, 0, 0], dtype=float)
        
        floated = False
        if 'plane' in obj_name:
            floated = bool(rng.choice([True, False]))
            if speed == 0:
                floated = False

        # engine_on = True if floated else bool(rng.choice([True, False]))
        engine_on = bool(rng.choice([True, False]))

        scene += obj
        move_until_no_overlap(scene, obj, simulator, rng)

        obj.metadata = {
            # appearance properties
            "id": i,
            "name": obj_name.lower(),
            "color": color_label,
            "size": size,

            # physics properties
            "mass": obj.mass,
            "engine_on": engine_on,
            "floated": floated,

            # first frame properties
            "init_position": obj.position,
            "init_speed": speed,
            "rand_degree": rand_degree,
            "init_quaternion": obj.quaternion,
        }

    return objects


def load_objects(args, rng, scene, source_path, simulator, load_scene=None):

    objects = []
    load_objects = load_scene['instances']
    load_objects = sorted(load_objects, key=lambda x: x['id'])

    for i, load_obj in enumerate(load_objects):
        # Assign object's type (or name):
        obj_name = load_obj['asset_id']
        
        cls_name, instance_name = obj_name.split('_')
        
        size = load_obj['size']
        
        color_label = load_obj['color']
        print(f"Create object: {cls_name} / {instance_name} / {color_label}")

        
        obj = kb.FileBasedObject(asset_id=obj_name,
                        render_filename= os.path.join(args.data_dir, "CGParts_colored", cls_name, instance_name, color_label, "object.obj"),
                        simulation_filename=args.data_dir+'/urdf/'+obj_name+'.urdf',
                        scale=size,
                        need_auto_smooth=instance_name not in no_smooth_cate
                        )

        obj.friction = load_obj['friction']
        obj.restitution = load_obj['restitution']
        obj.mass = load_obj['mass']

        # Random position
        obj.position = load_obj['init_position']
        rand_degree = load_obj['rand_degree']
        obj.quaternion = load_obj['init_quaternion']

        # Random velocity:     
        speed = load_obj['init_speed']

        rand_radians = np.radians(rand_degree)
        obj.velocity = np.array([math.cos(rand_radians), math.sin(rand_radians), 0.0]) * speed 
        
        floated = load_obj['floated']
        engine_on = load_obj['engine_on']

        scene += obj

        obj.metadata = {
            # appearance properties
            "id": i,
            "name": obj_name.lower(),
            "color": color_label,
            "size": size,

            # physics properties
            "mass": obj.mass,
            "friction": obj.friction,
            "restitution": obj.restitution,
            "engine_on": engine_on,
            "floated": floated,

            # first frame properties
            "init_position": obj.position,
            "init_speed": speed,
            "rand_degree": rand_degree,
            "init_quaternion": obj.quaternion,
        }

    return objects

def move_until_no_overlap(scene, obj, simulator, rng):
    for i in range(10):
        has_overlap = simulator.check_background_overlap(obj)
        if not has_overlap:
            break

        XYZ_jitter = np.array([0, 0, 0.001])

        obj.position = obj.position + XYZ_jitter


    for i in range(100):
        has_overlap = simulator.check_foreground_overlap(obj)
        if not has_overlap:
            break

        XYZ_jitter = rng.uniform(-0.5, 0.5, 3)
        XYZ_jitter[2] = 0

        obj.position = obj.position + XYZ_jitter
    return

def random_quaternion(rng):
    random_values = rng.rand(4)
    normalized_vector = random_values / np.linalg.norm(random_values)

    return normalized_vector

def random_rotate_quaternion(args, rng, obj_name):

    initial_quaternion = pyquaternion.Quaternion(axis=[1, 0, 0], degrees=90)

    rand_degree = rng.randint(0, 360)
    if args.debug:
        rand_degree = 0

    # if 'bicycle' not in obj_name:
    #     add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), degrees=(rand_degree-90)%360)
    # else:
    #     add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), degrees=rand_degree)
    add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), degrees=(rand_degree-90)%360)

    return add_random_rotation * initial_quaternion, rand_degree

def facing_center_quaternion(args, rng, obj_name, position):

    initial_quaternion = pyquaternion.Quaternion(axis=[1, 0, 0], degrees=90)

    # rand_degree = rng.randint(0, 360)
    dx, dy = position[0], position[1]
    if dx == 0 and dy == 0:
        rand_degree = rng.randint(-20, 20)
    else:
        rand_degree = np.degrees(np.arctan2(dy, dx)) + 180 + rng.randint(-20, 20)
        if rand_degree < 0:
            rand_degree += 360
    # rand_degree = 0

    if args.debug:
        rand_degree = 0

    # if 'bicycle' not in obj_name:
        # add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), degrees=(rand_degree-90)%360)
    # else:
        # add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), degrees=rand_degree)
    add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), degrees=(rand_degree-90)%360)

    return add_random_rotation * initial_quaternion, rand_degree


def sim_run(args):
    
    start_time = time()
    load_scene = None
    if args.load_scene:
        load_path = f'{args.load_scene}/super_clever_{args.iteration}/metadata.json'
        with open(load_path, 'r') as f:
            load_scene = json.load(f)

    scene, rng, output_dir, scratch_dir = kb.setup(args)
    
    output_dir = Path(str(output_dir) + '/super_clever_{}'.format(args.iteration))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)
    
    kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
    hdri_source = kb.AssetSource.from_manifest(args.hdri_assets)
    gso = kb.AssetSource.from_manifest(args.gso_assets)
    
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    hdri_id = rng.choice(train_backgrounds)
    hdri_id = rng.choice(['kloetzle_blei', 'lenong_3'])
    
    if load_scene:
        hdri_id = load_scene["metadata"]["background_hdri_id"]

    background_hdri = hdri_source.create(asset_id=hdri_id)
    logging.info("Using background %s", hdri_id)
    # simulator
    simulator = PyBullet(scene, scratch_dir)


    #add floor and background
    scene = add_floor(scene, kubasic, renderer, background_hdri, hdri_id, args, load_scene=load_scene)
    # scene = add_walls(scene, renderer, background_hdri)


    #add light
    # if load_scene and "light" in load_scene:
    #     illumination = load_scene["light"]["illumination"]
    # else:
    #     illumination=[0.25, 0.25, 0.25]
        
    # scene = add_light(scene, rng, args, illumination=illumination)

    renderer._set_ambient_light_hdri(background_hdri.filename)


    # scene = add_obstacles(scene, renderer, rng, gso, args)

    # Add objects
    objects = add_objects(args, rng, scene, args.data_dir, simulator, load_scene=load_scene)

    # Camera
    logging.info("Setting up the Camera...")
    scene, camera_info = setup_camera(scene, rng, args, load_camera_from_scene=load_scene)
    

    animation, collisions = simulator.run(frame_start=0,
                                      frame_end=scene.frame_end)

    data_stack = renderer.render(return_layers=['rgba','segmentation'])
    logging.info("Post processing...")

    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
        
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    logging.info("Write image files.")
    kb.write_image_dict(data_stack, output_dir)
    # kb.post_processing.compute_bboxes(data_stack["segmentation"],
    #                                 visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    # import pdb; pdb.set_trace()

    kb.write_json(filename=output_dir / "metadata.json", data={
        "args": vars(args),
        "metadata": kb.get_scene_metadata(scene),
        "camera": camera_info,
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions":  kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })
    end_time = time()
    logging.info("Scene index %d | Time elapsed: %s", args.iteration, end_time - start_time)
    return

if __name__ == '__main__':
    args = config()
    sim_run(args)
