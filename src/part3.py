# -*- coding: utf-8 -*-

"""
mean Curvature by cotan laplace
"""

import os
import open3d as o3d
import trimesh
from tools import *
import argparse


# check whether the data folder exists
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(FILE_PATH, '../data/cw2_meshes/curvatures')
if not os.path.exists(RES_PATH):
    print('cannot find resources folder, please update RES_PATH')
    exit(1)


if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='part3.')
    parser.add_argument('--obj', default='lilium_s.obj', type=str, help='the 3D model we want to use.')
    # parser.add_argument('--saveImg', action='store_true', help='save mesh as image.')
    args = parser.parse_args()

    # Load data file into trimesh-object
    DataFile = args.obj   # lilium_s.obj, plane.obj
    mesh_fp = os.path.join(RES_PATH, DataFile)
    assert os.path.exists(mesh_fp), 'cannot found:' + DataFile
    tm = trimesh.load(mesh_fp)

    # compute to curvature
    Obj = LaplaceOperator.LaplaceOperator(tm)
    H = Obj.getMeanCurvature('cotan')
    print(H)

    # convert curvature to rgb for showing
    rgb_H = tools.toRGB(H)

    # show the object by open3d
    mesh_o3d = tools.toO3d(tm, color=rgb_H)
    o3d.visualization.draw_geometries([mesh_o3d])















