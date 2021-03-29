# -*- coding: utf-8 -*-

"""
mean Curvature by uniform Laplace, and Gaussian Curvature
"""

import os
from os.path import join as opj
import open3d as o3d
import trimesh
from tools import *
import numpy as np
import argparse


# check whether the data folder exists
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(FILE_PATH, '../data/cw2_meshes/curvatures')
if not os.path.exists(RES_PATH):
    print('cannot find resources folder, please update RES_PATH')
    exit(1)


if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='part1.')
    parser.add_argument('--obj', default='lilium_s.obj', type=str, help='the 3D model we want to use.')
    # parser.add_argument('--saveImg', action='store_true', help='save mesh as image.')
    parser.add_argument('--mode', default='mean', type=str, help='show Gaussian or Mean curvature.')
    args = parser.parse_args()

    # Load data file into trimesh-object
    DataFile = args.obj  # lilium_s.obj, plane.obj
    mesh_fp = os.path.join(RES_PATH, DataFile)
    assert os.path.exists(mesh_fp), 'cannot found:' + DataFile
    tm = trimesh.load(mesh_fp)

    # compute to curvature
    Obj = LaplaceOperator.LaplaceOperator(tm)

    if args.mode == 'mean':
        H = Obj.getMeanCurvature('uniform')
        rgb = tools.toRGB(H)  # convert curvature to rgb for showing
    elif args.mode == 'gaussian':
        K = Obj.getGaussianCurvature()
        rgb = tools.toRGB(K)
    else:
        raise ValueError("python file input parameter 'mode' should only be mean or gaussian")

    # show the object by open3d
    mesh_o3d = tools.toO3d(tm, color=rgb)
    o3d.visualization.draw_geometries([mesh_o3d])

    # if args.saveImg:
    #     dirs = '../ouput/image/part1'
    #     if not os.path.exists(dirs):
    #         os.makedirs(dirs)
    #     model_name = os.path.splitext(args.obj)[0]
    #
    #     angley = -45
    #     anglexs = np.linspace(0, 360, 6).astype(int)
    #     for (i, anglex) in enumerate(anglexs):
    #         tools.saveImage(mesh_o3d, opj(dirs, '{:}_{:d}.png'.format(model_name, i)), anglex, angley)















