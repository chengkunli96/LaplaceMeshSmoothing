#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import open3d as o3d
import matplotlib as mpl
import matplotlib.cm as cm


def toO3d(tm, color):
    """put trimesh object into open3d object"""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tm.faces)
    mesh_o3d.compute_vertex_normals()
    vex_color_rgb = np.array(color)
    if vex_color_rgb.ndim < 2:
        vex_color_rgb = np.tile(vex_color_rgb, (tm.vertices.shape[0], 1))
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vex_color_rgb)
    return mesh_o3d


def toRGB(H):
    """make a N*1 ndarray to N*3 rgb matrix"""
    rgba = cm.hot(H)
    rgb = rgba[:, :3]
    return rgb


def saveImage(mesh_o3d, path, anglex, angley):
    """save the open3d mesh as image"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_o3d)
    visCtrl = vis.get_view_control()
    vis.update_geometry(mesh_o3d)
    visCtrl.rotate(1000 / 180 * anglex, 1000 / 180 * angley)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename=path)
    vis.destroy_window()

