import os
import sys

import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes

from roca.engine import Predictor

def main(args):
    predictor = Predictor(
        data_dir=args.data_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        wild=args.wild,
    )
    data_root = '/home/aston/Desktop/Datasets/pose_data/ScanNOCS'
    scene_list = os.listdir(data_root)
    for scene in scene_list:
        image_list = [os.path.join(data_root, scene, file) for file in os.listdir(os.path.join(data_root,scene)) if '_color' in file]





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False,default='/home/aston/Desktop/python/ROCA/Data/Dataset')
    parser.add_argument('--model_path', required=False,default='/home/aston/Desktop/python/ROCA/Models/model_best.pth')
    parser.add_argument('--config_path', required=False,default='/home/aston/Desktop/python/ROCA/Models/config.yaml')
    parser.add_argument('--wild', action='store_true')
    parser.add_argument('--output_dir', default='demo_output')
    args = parser.parse_args(sys.argv[1:])
    main(args)