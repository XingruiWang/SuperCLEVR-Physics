import os
import subprocess
import sys
import argparse

import pybullet as pb

parser = argparse.ArgumentParser()
parser.add_argument('--convert_dir', default='../output/convert',
        help="Make the .blend file to kubric object and save it")

def main(args):
    watertight_dir = args.convert_dir + '/watertight/'
    glb_dir = args.convert_dir + '/glbs/'
    obj_dir = args.convert_dir + '/objs/'
    logs_dir = args.convert_dir + '/logs/'

    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    for item in os.listdir(obj_dir):
        # make collision obj
        obj_path = os.path.join(obj_dir, item)
        glb_path = glb_dir + os.path.splitext(item)[0] + '.glb'
        target_path = obj_dir + 'collision_' + item
        log_path_0 = logs_dir + 'collision_stdout_' + os.path.splitext(item)[0] + '.txt'
        log_path_1 = logs_dir + 'collision_' + os.path.splitext(item)[0] + '.txt'
        pb.vhacd(obj_dir, target_path, log_path_0)

if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()

