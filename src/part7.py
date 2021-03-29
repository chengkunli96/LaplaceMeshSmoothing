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
    parser.add_argument('--iteration', default=100, type=int, help='iteration.')
    parser.add_argument('--noise', default=0, type=float, help='save mesh as image.')
    parser.add_argument('--saveImg', action='store_true', help='save mesh as image.')
    args = parser.parse_args()

    # Load data file into trimesh-object
    DataFile = args.obj  # armadillo.obj, fandisk_ns.obj, plane_ns.obj
    mesh_fp = os.path.join(RES_PATH, DataFile)
    assert os.path.exists(mesh_fp), 'cannot found:' + DataFile
    tm = trimesh.load(mesh_fp)

    # add Gaussian noise
    std = args.noise
    vertices = np.array(tm.vertices)
    box_scale = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    vertices_noise = vertices + std * box_scale.reshape(1, -1) * np.random.randn(vertices.shape[0], vertices.shape[1])
    faces_noise = np.array(tm.faces)
    tm_noise = trimesh.Trimesh(vertices=vertices_noise, faces=faces_noise)
    # show
    if not args.saveImg:
        noise_mesh_o3d = tools.toO3d(tm_noise, color=np.array(tm_noise.face_normals))
        o3d.visualization.draw_geometries([noise_mesh_o3d])

    # smoothing
    Obj = LaplaceOperator.LaplaceOperator(tm_noise)
    newtm = Obj.smooth(lr=args.step_size, maxIter=args.iteration, mode='implicit')

    # show the object by open3d
    mesh_o3d = tools.toO3d(newtm, color=np.array(newtm.face_normals))
    if not args.saveImg:
        o3d.visualization.draw_geometries([mesh_o3d])

    if args.saveImg:
        # save as image
        dirs = '../ouput/image/part7'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        model_name = os.path.splitext(DataFile)[0]
        if model_name == 'plane_ns':
            anglex = 0
            angley = -45
        elif model_name == 'fandisk_ns':
            anglex = 0
            angley = -240
        else:
            anglex = 0
            angley = -45
        tools.saveImage(mesh_o3d,
                        opj(dirs, '{:}_noise{:}_lambda{:.0e}_iter{:d}.png'.
                            format(model_name, args.noise, args.step_size, args.iteration)),
                        anglex=anglex, angley=angley)
        # save noise Image
        original_mesh_o3d = tools.toO3d(tm_noise, color=np.array(tm_noise.face_normals))
        tools.saveImage(original_mesh_o3d, opj(dirs, '{:}_noise{:}.png'.format(model_name, args.noise)), anglex, angley)