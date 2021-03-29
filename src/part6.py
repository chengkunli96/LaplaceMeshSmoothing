#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implicit Smoothing
"""

import os
import open3d as o3d
import trimesh
from tools import *
import numpy as np
import argparse
from os.path import join as opj


# check whether the data folder exists
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(FILE_PATH, '../data/cw2_meshes/smoothing')
if not os.path.exists(RES_PATH):
    print('cannot find resources folder, please update RES_PATH')
    exit(1)


if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='part3.')
    parser.add_argument('--obj', default='fandisk_ns.obj', type=str, help='the 3D model we want to use.')
    parser.add_argument('--step_size', default=1e-5, type=float, help='step size.')
    parser.add_argument('--iteration', default=50, type=int, help='iteration.')
    parser.add_argument('--saveImg', action='store_true', help='save mesh as image.')
    args = parser.parse_args()

    # Load data file into trimesh-object
    DataFile = args.obj  # fandisk_ns.obj, plane_ns.obj
    mesh_fp = os.path.join(RES_PATH, DataFile)
    assert os.path.exists(mesh_fp), 'cannot found:' + DataFile
    tm = trimesh.load(mesh_fp)

    Obj = LaplaceOperator.LaplaceOperator(tm)
    newtm = Obj.smooth(lr=args.step_size, maxIter=args.iteration, mode='implicit')

    # show the object by open3d
    mesh_o3d = tools.toO3d(newtm, color=np.array(newtm.face_normals))
    if not args.saveImg:
        o3d.visualization.draw_geometries([mesh_o3d])

    if args.saveImg:
        # save as image
        dirs = '../ouput/image/part6'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        model_name = os.path.splitext(DataFile)[0]
        if model_name == 'plane_ns':
            anglex = 0
            angley = -45
        elif model_name == 'fandisk_ns':
            anglex = 0
            angley = -240
        tools.saveImage(mesh_o3d,
                        opj(dirs, '{:}_lambda{:.0e}_iter{:d}.png'.format(model_name, args.step_size, args.iteration)),
                        anglex=anglex, angley=angley)
        # # save original
        # original_mesh_o3d = tools.toO3d(tm, color=np.array(tm.face_normals))
        # tools.saveImage(original_mesh_o3d, opj(dirs, '{:}_original.png'.format(model_name)), anglex, angley)