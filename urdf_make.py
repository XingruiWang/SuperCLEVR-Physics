import os
import argparse
import logging
import json
import trimesh
import trimesh_utils
from trimesh_utils import get_object_properties
_DEFAULT_LOGGER = logging.getLogger(__name__)
from pathlib import Path
from typing import Tuple
logger=_DEFAULT_LOGGER
from urdf_template import URDF_TEMPLATE

def get_object_volume(obj_path:Path, logger=_DEFAULT_LOGGER, density=1.0):
    # --- override the trimesh logger
    trimesh.util.log = logger
    # import pdb; pdb.set_trace()
        
    tmesh = trimesh_utils.get_tmesh(str(obj_path))

    properties = {
        "volume": tmesh.volume,
        "surface_area": tmesh.area,
        "mass": tmesh.volume * density,
    }
    return properties

def get_visual_properties(obj_path:Path, logger=_DEFAULT_LOGGER):
    # --- override the trimesh logger
    trimesh.util.log = logger

    tmesh = trimesh_utils.get_tmesh(str(obj_path))

    properties = {
        "nr_vertices": len(tmesh.vertices),
        "nr_faces": len(tmesh.faces),
    }
    return properties


def make_urdf(asset_id, collision_obj, watertight_obj, visual_glb, output):
    urdf_dir = output + '/urdf/'
    json_dir = output + '/json/'
    if not os.path.isdir(urdf_dir):
            os.makedirs(urdf_dir)
    if not os.path.isdir(json_dir):
            os.makedirs(json_dir)

    urdf_path = urdf_dir + asset_id + '.urdf'
    json_path = json_dir + asset_id + '.json'

    properties = get_object_properties(collision_obj, logger)
    properties.update(get_object_volume(watertight_obj))
    properties.update(get_visual_properties(visual_glb))
    properties['asset_id'] = asset_id
    
    properties["id"] = asset_id
    urdf_str = URDF_TEMPLATE.format(**properties)

    with open(urdf_path, 'w') as fd:
        fd.write(urdf_str)

    asset_entry = {
        "assets": asset_id,
        "id": asset_id,
        "asset_type": "FileBasedObject",
        "kwargs": {
            "bounds": properties["bounds"],
            "mass": properties["mass"],
            "render_filename": asset_id + '.glb',
            "simulation_filename": asset_id + '.urdf',
        },
        # "license": "https://shapenet.org/terms",
        "metadata": {
            "watertight_mesh_filename": asset_id + '.obj',
            "nr_faces": properties["nr_faces"],
            "nr_vertices": properties["nr_vertices"],
            "surface_area": properties["surface_area"],
            "volume": properties["volume"],
        }
    }

    with open(json_path, "w") as fd:
        json.dump(asset_entry, fd, indent=4, sort_keys=True)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_id', type=str)
    parser.add_argument('--collision_obj', type=str)
    parser.add_argument('--watertight_obj', type=str)
    parser.add_argument('--visual_glb', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    make_urdf(asset_id=args.asset_id,
                collision_obj=args.collision_obj,
                watertight_obj=args.watertight_obj,
                visual_glb=args.visual_glb,
                output=args.output)