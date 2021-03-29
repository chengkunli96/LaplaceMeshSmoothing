#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decompose
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
RES_PATH = os.path.join(FILE_PATH, '../data/cw2_meshes/decompose')
if not os.path.exists(RES_PATH):
    print('cannot find resources folder, please update RES_PATH')
    exit(1)


if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='part3.')
    parser.add_argument('--k', default=15, type=int, help='decompose to k dimensions')
    args = parser.parse_args()

    # Load data file into trimesh-object
    DataFile = 'armadillo.obj'
    mesh_fp = os.path.join(RES_PATH, DataFile)
    assert os.path.exists(mesh_fp), 'cannot found:' + DataFile
    tm = trimesh.load(mesh_fp)

    Obj = LaplaceOperator.LaplaceOperator(tm)
    # newtm = Obj.decompose(k=np.array(tm.vertices).shape[0])
    newtm = Obj.decompose(args.k)

    # show the object by open3d
    mesh_o3d = tools.toO3d(newtm, color=np.array(newtm.face_normals))
    o3d.visualization.draw_geometries([mesh_o3d])

    # save mesh as .obj file
    dir = '../ouput/mesh/part4'
    if not os.path.exists(dir):
        os.makedirs(dir)
    newtm.export(opj(dir, 'q4_arma_k{:d}.obj'.format(args.k)))

    # # save as image
    # dirs = '../ouput/image/part4'
    # if not os.path.exists(dirs):
    #     os.makedirs(dirs)
    # model_name = os.path.splitext(DataFile)[0]
    # tools.saveImage(mesh_o3d, opj(dirs, '{:}_{:d}.png'.format(model_name, args.k)), 20, 5)
    # # save original
    # original_mesh_o3d = tools.toO3d(tm, color=np.array(newtm.face_normals))
    # tools.saveImage(original_mesh_o3d, opj(dirs, '{:}_original.png'.format(model_name)), 20, 5)

