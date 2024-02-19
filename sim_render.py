import sys
sys.path.insert(0, '/kubric')
import logging
import argparse
import os
import json
import pyquaternion
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np

from pathlib import Path
import bpy

def config():
    parser = kb.ArgumentParser()

    parser.set_defaults(
        frame_end = 60,
        # frame_end = 2,
        resolution=(1280, 960),
    )


    parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--properties_cgpart", type=str,
                    default="data/properties_cgpart.json")

    # scene parameters
    parser.add_argument("--background", choices=["clevr", "colored"], default="clevr")
    parser.add_argument("--num_obj_min", type=int, default=2)
    parser.add_argument("--num_obj_max", type=int, default=10)
    
    parser.add_argument("--scene_size", type=int, default=5)
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)    
    parser.add_argument('--camera', type=str)

    # frames parameters

    
    # io
    parser.add_argument('--iteration', type=int, default=100)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    # debug
    parser.add_argument('--height', type=str, default='realistic')
    parser.add_argument('--skip_segmentation', action='store_true')


    args = parser.parse_args()

    if args.properties_cgpart:
        with open(args.properties_cgpart, "r") as f:
            args.properties_cgpart = json.load(f)

    return args

def setup_camera(scene, rng, args):
    
    
    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
    if args.camera == "fixed":  # Specific position + jitter
        # scene.camera.position = [7.48113, -6.50764, 5.34367] + rng.rand(3)
        scene.camera.position = [7.48113, -6.50764, 3.34367] + rng.rand(3)
    if args.camera == "random":  # Random position in half-sphere-shell
        scene.camera.position = kb.sample_point_in_half_sphere_shell(
        inner_radius=7., outer_radius=9., offset=0.1)
    if args.camera == "god":  # Random position + jitter
        scene.camera.position = [70, -60, 30] + rng.rand(3)
    scene.camera.look_at((0, 0, 1.3))
    return scene


def add_walls(scene):

    # Define dimensions of the wall
    wall_width = 10
    wall_height = 10
    wall_depth = 0.1

    # Create a wall object
    # wall = kb.Cube(scale=(wall_width, wall_height, wall_depth), \
    #                position = (0, 0, -wall_depth/2))
    left_wall = kb.Cube(scale=(wall_depth, wall_width, wall_height), \
                   position = (-wall_width // 2, 0, wall_height/2))
    right_wall = kb.Cube(scale=(wall_width, wall_depth, wall_height), \
                   position = (0, wall_width // 2, wall_height/2))
    
    left_wall.color = kb.Color.from_name("gray")
    right_wall.color = kb.Color.from_name("gray")
    
    # Add the wall to the scene
    scene.add(left_wall)
    scene.add(right_wall)
    return scene

def realistic_sizes(obj_name):
    if obj_name == 'aeroplane':
        size = 3.2
    elif obj_name == 'bicycle':
        size = 1.5
    elif obj_name == 'car':
        size = 2.2
    elif obj_name == 'motorbike':
        size = 1.5
    elif obj_name == 'bus':
        size = 3.5
    else:
        raise ValueError(f'Object name {obj_name} not found')
    return "realistic", size

def add_light(scene, rng, args):
    scene += kb.assets.utils.get_clevr_lights(rng=rng)  
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
    return scene

def add_floor(scene, kubasic, args):
    floor_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)

    if args.background == "clevr":
        # use default color instead
        # floor_material.color = kb.Color.from_name("gray")
        scene.metadata["background"] = "clevr"

    elif args.background == "colored":
        floor_material.color = kb.random_hue_color()
        scene.metadata["background"] = floor_material.color.hexstr

    scene += kubasic.create("dome", name="floor", material=floor_material,
                        scale=2.0,
                        friction=args.floor_friction,
                        restitution=args.floor_restitution,
                        static=True, background=True, position = (0, 0, 0))
    return scene

def add_random_objects(args, rng, scene, source_path):
    objects = []
    obj_lib = os.listdir(args.data_dir + '/objs')

    num_objs = rng.randint(args.num_obj_min, args.num_obj_max)
    print('adding', num_objs, 'objects.')

    for i in range(num_objs):
        obj_name = os.path.splitext(rng.choice(obj_lib))[0]
        print(obj_name)

        color_label, random_color = kb.randomness.sample_color("super_clevr", rng)
        
        if args.height == 'realistic':
            size_label, size = realistic_sizes(obj_name.split('_')[0])
        else:
            size_label, size = kb.randomness.sample_sizes("super_clevr", rng)
        
        obj = kb.FileBasedObject(asset_id=obj_name,
                                render_filename=args.data_dir+'/objs/'+obj_name+'.obj',
                                simulation_filename=args.data_dir+'/urdf/'+obj_name+'.urdf',
                                scale=size
                                )
        # import pdb; pdb.set_trace()
        # obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
        # obj.position = obj.position - (0, 0, obj.aabbox[0][2])  
        # obj.quaternion = np.array([0., 1., 0., 0.], dtype=float)
        position = args.scene_size * rng.uniform(-1, 1, 3)
        
        if args.height == 'fixed':
            position[2] = 1.0
        elif args.height == 'random':
            position[2] = args.scene_size * rng.uniform(-1, 1, 1)
        elif args.height == 'realistic':
            if 'plane' in obj_name:
                position[2] = args.scene_size * rng.uniform(0.5, 0.8, 1)
            else:
                position[2] = 0
                # adjust the height of the object
                sub_obj_name = obj_name.split('_')[1]
                info_z = args.properties_cgpart['info_z'][sub_obj_name]
                position[2] -= info_z * size
                # import pdb; pdb.set_trace()
        else:
            raise ValueError(f'Height option {args.height} not found')
        ### move the object upwords
        # if abs(position[2]) < 1:
        #     position[2] = abs(position[2]) + 2
        # else: 
        #     position[2] = abs(position[2])
        obj.position = position
        # random angle or fix
        # obj.quaternion = random_quaternion(rng)
        # add_rotation = pyquaternion.Quaternion(axis=(0.0, 1.0, 0.0), degrees=90)
        # obj.quaternion = pyquaternion.Quaternion(axis=[1, 0, 0], degrees=90) * add_rotation
        obj.quaternion = random_rotate_quaternion(rng)

        obj.velocity = (rng.uniform(*[(-4., -4., 0.), (4., 4., 0.)]) - [obj.position[0], obj.position[1], 0])

        scene += obj
        rng.uniform((-4., -4., 0.), (4., 4., 0.))

        # material and color
        material_name = rng.choice(["metal", "rubber"])
        if material_name == "metal":
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0,
                                                    roughness=0.2, ior=2.5)
            obj.friction = 0.4
            obj.restitution = 0.3
            obj.mass *= 2.7 * size ** 3
        else:  # material_name == "rubber"
            obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0.,
                                                    ior=1.25, roughness=0.7,
                                                    specular=0.33)
            obj.friction = 0.8
            obj.restitution = 0.7
            obj.mass *= 1.1 * size ** 3

        obj.metadata = {
            #"shape": shape_name.lower(),
            "size": size,
            "size_label": size_label,
            "material": material_name.lower(),
            "color": random_color.rgb,
            "color_label": color_label,
        }

    return objects

def random_quaternion(rng):
    random_values = rng.rand(4)
    normalized_vector = random_values / np.linalg.norm(random_values)

    return normalized_vector

def random_rotate_quaternion(rng):

    initial_quaternion = pyquaternion.Quaternion(axis=[1, 0, 0], degrees=90)
    rand_degree = rng.randint(0, 360)
    add_random_rotation = pyquaternion.Quaternion(axis=(0.0, 1.0, 0.0), degrees=rand_degree)
    return initial_quaternion * add_random_rotation


def sim_run(args):
    scene, rng, output_dir, scratch_dir = kb.setup(args)
    
    output_dir = Path(str(output_dir) + '/super_clever_{}'.format(args.iteration))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)
    
    kubasic = kb.AssetSource.from_manifest(args.kubasic_assets)
    
    # simulator
    simulator = PyBullet(scene, scratch_dir)

    #add light
    scene = add_light(scene, rng, args)

    #add floor and background
    scene = add_floor(scene, kubasic, args)
    scene = add_walls(scene)

    # add objects
    objects = add_random_objects(args, rng, scene, args.data_dir)

    # Camera
    logging.info("Setting up the Camera...")
    scene = setup_camera(scene, rng, args)
    

    animation, collisions = simulator.run(frame_start=0,
                                      frame_end=scene.frame_end+1)

    data_stack = renderer.render(return_layers=['rgba','segmentation'])

    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    if not args.skip_segmentation:
        data_stack["segmentation"] = kb.adjust_segmentation_idxs(
            data_stack["segmentation"],
            scene.assets,
            visible_foreground_assets)
        
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    # kb.post_processing.compute_bboxes(data_stack["segmentation"],
    #                                 visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(filename=output_dir / "metadata.json", data={
        "args": vars(args),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions":  kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })
    return



if __name__ == '__main__':
    args = config()
    sim_run(args)
