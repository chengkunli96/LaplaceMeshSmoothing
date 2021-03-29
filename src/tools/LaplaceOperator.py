#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uniform Laplace and
@author: Chengkun Li
"""

import numpy as np
import trimesh
import math
from scipy.linalg import sqrtm
from scipy import sparse
from scipy.sparse.linalg import spsolve


class LaplaceOperator(object):
    def __init__(self, tm: trimesh):
        self.tm = tm
        self.M = None
        self.C = None
        self.L = None
        self.H = None
        self.K = None

    def __getM(self, mode='uniform'):
        num_vertices = self.tm.vertices.shape[0]
        if mode == 'uniform':
            self.M = []
            for i in range(num_vertices):
                self.M.append(len(self.tm.vertex_neighbors[i]))
            self.M = np.diag(self.M)
        elif mode == 'cotan':
            self.M = []
            for i in range(num_vertices):
                area = 0
                face_neighbors = self.tm.vertex_faces[i][self.tm.vertex_faces[i] != -1]
                for face_id in face_neighbors:
                    [id_a, id_b, id_c] = self.tm.faces[face_id]
                    vertex_a = np.array(self.tm.vertices[id_a])
                    vertex_b = np.array(self.tm.vertices[id_b])
                    vertex_c = np.array(self.tm.vertices[id_c])
                    area += self.__computeTriangleArea(vertex_a, vertex_b, vertex_c)
                self.M.append(area / 3)
            self.M = np.diag(self.M)
        else:
            raise ValueError("parameter mode should only be 'uniform' or 'cotan'.")

    def __getC(self, mode='uniform'):
        num_vertices = self.tm.vertices.shape[0]
        if mode == 'uniform':
            self.C = np.zeros((num_vertices, num_vertices))
            for i in range(num_vertices):
                for id_vertex_neighbor in self.tm.vertex_neighbors[i]:
                    self.C[i, id_vertex_neighbor] = 1
                cii = np.sum(self.C[i, :])
                self.C[i, i] = -cii
        elif mode == 'cotan':
            self.C = np.zeros((num_vertices, num_vertices))
            for i in range(num_vertices):
                vertex_neighbors = self.tm.vertex_neighbors[i]
                vertex_in_boundary = False
                for j in vertex_neighbors:
                    id_pts = self.__findPtsAdjacency(i, j)
                    if len(id_pts) == 0 or len(id_pts) == 1:  # vertex i in the boundary
                        vertex_in_boundary = True
                    elif len(id_pts) == 2:
                        vertex_a1 = np.array(self.tm.vertices[id_pts[0]])
                        vertex_a2 = np.array(self.tm.vertices[id_pts[1]])
                        vertex_b = np.array(self.tm.vertices[i])
                        vertex_c = np.array(self.tm.vertices[j])
                        alpha = self.__computeAngle(vertex_a1, vertex_b, vertex_c)
                        beta = self.__computeAngle(vertex_a2, vertex_b, vertex_c)
                        self.C[i, j] = 0.5 * (1 / math.tan(alpha) + 1 / math.tan(beta))
                    else:
                        raise ValueError('there are more than 2 vertices facing a same edge.')
                cii = np.sum(self.C[i, :])
                self.C[i, i] = -cii
                if vertex_in_boundary:
                    self.C[i, :] = 0

        else:
            raise ValueError("parameter mode should only be 'uniform' or 'cotan'.")

    def __computeTriangleArea(self, a, b, c):
        vec_ab = b - a
        vec_ac = c - a
        area = np.linalg.norm(np.cross(vec_ab, vec_ac)) / 2.
        return area

    def __computeAngle(self, a, b, c):
        vec_ab = b - a
        vec_ac = c - a
        angle = vec_ab @ vec_ac / (np.linalg.norm(vec_ab) * np.linalg.norm(vec_ac))
        angle = np.arccos(angle)
        return angle

    def __findPtsAdjacency(self, id_pt1, id_pt2):
        """Find pts that face the same edge, which we call here ‘adjacent’."""
        face_neighbors = self.tm.vertex_faces[id_pt1][self.tm.vertex_faces[id_pt1] != -1]
        id_pts = []
        for face_id in face_neighbors:
            if id_pt2 in self.tm.faces[face_id]:
                id_pt = list(self.tm.faces[face_id].copy())
                id_pt.remove(id_pt1)
                id_pt.remove(id_pt2)
                id_pt = id_pt[0]
                id_pts.append(id_pt)
        return id_pts
        # id_vetex1_neighbors = list(self.tm.vertex_neighbors[id_pt1].copy())
        # id_vetex2_neighbors = list(self.tm.vertex_neighbors[id_pt2].copy())
        # id_pts = list(set(id_vetex1_neighbors).intersection(set(id_vetex2_neighbors)))
        # return id_pts

    def getMeanCurvature(self, mode='uniform'):
        if mode != 'uniform' and mode != 'cotan':
            raise ValueError("parameter mode should only be 'uniform' or 'cotan'.")
        self.__getM(mode)
        self.__getC(mode)
        self.L = np.linalg.inv(self.M) @ self.C
        self.H = 0.5 * np.linalg.norm(self.L @ self.tm.vertices, axis=1)
        return self.H

    def getGaussianCurvature(self):
        num_vertices = self.tm.vertices.shape[0]
        self.K = np.zeros((num_vertices, 1))
        for i in range(num_vertices):
            # whether the vertex i is in the boundary
            vertex_neighbors = self.tm.vertex_neighbors[i]
            vertex_in_boundary = False
            for j in vertex_neighbors:
                id_pts = self.__findPtsAdjacency(i, j)
                if len(id_pts) == 0 or len(id_pts) == 1:  # vertex i in the boundary
                    vertex_in_boundary = True

            if not vertex_in_boundary:
                angle = 0
                area = 0
                face_neighbors = self.tm.vertex_faces[i][self.tm.vertex_faces[i] != -1]
                for face_id in face_neighbors:
                    vertex_ids = list(self.tm.faces[face_id])
                    vertex_ids.remove(i)
                    if len(vertex_ids) != 2:
                        raise ValueError
                    vertex_a = self.tm.vertices[i]
                    vertex_b = self.tm.vertices[vertex_ids[0]]
                    vertex_c = self.tm.vertices[vertex_ids[1]]
                    angle += self.__computeAngle(vertex_a, vertex_b, vertex_c)
                    area += self.__computeTriangleArea(vertex_a, vertex_b, vertex_c)
                area /= 3
                self.K[i] = (2 * np.pi - angle) / area
            else:
                self.K[i] = 0
        self.K = np.squeeze(self.K)
        return self.K

    def __getEig(self):
        self.__getM('cotan')
        self.__getC('cotan')
        invM_half = np.sqrt(np.linalg.inv(self.M))
        A = invM_half @ self.C @ invM_half

        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvectors = invM_half @ eigenvectors
        return eigenvalues, eigenvectors

    def decompose(self, k=5):
        eigenvalues, eigenvectors = self.__getEig()
        ids_sorted = np.argsort(np.abs(eigenvalues))
        vertices = np.array(self.tm.vertices)
        vertices_recon = np.zeros_like(vertices)
        for i in range(k):
            eigenvector = eigenvectors[:, ids_sorted[i]].real.reshape(-1, 1)
            tmp = vertices.T @ self.M @ eigenvector
            vertices_recon[:, 0] += (tmp[0] * eigenvector).squeeze()
            vertices_recon[:, 1] += (tmp[1] * eigenvector).squeeze()
            vertices_recon[:, 2] += (tmp[2] * eigenvector).squeeze()
        faces_recon = np.array(self.tm.faces)
        newtm = trimesh.Trimesh(vertices=vertices_recon, faces=faces_recon)
        return newtm

    def smooth(self, lr=1e-7, maxIter=100, mode='explicit'):
        self.__getM('cotan')
        self.__getC('cotan')
        if mode == 'explicit':
            self.L = np.linalg.inv(self.M) @ self.C
            vertices = np.array(self.tm.vertices)
            vertices_new = vertices.copy()
            for i in range(maxIter):
                print('\rSmoothing iteration: {:d}/{:d} ...'.format(i + 1, maxIter), end='', flush=True)
                vertices_new += lr * self.L @  vertices_new
            faces_new = np.array(self.tm.faces)
            newtm = trimesh.Trimesh(vertices=vertices_new, faces=faces_new)
        elif mode == 'implicit':
            # A = np.linalg.inv(self.M - lr * self.C)
            A = sparse.csc_matrix(self.M - lr * self.C)
            vertices = np.array(self.tm.vertices)
            vertices_new = vertices.copy()
            for i in range(maxIter):
                print('\rSmoothing iteration: {:d}/{:d} ...'.format(i + 1, maxIter), end='', flush=True)
                # vertices_new = A @ self.M @ vertices_new
                vertices_new = spsolve(A, self.M @ vertices_new)
            faces_new = np.array(self.tm.faces)
            newtm = trimesh.Trimesh(vertices=vertices_new, faces=faces_new)
        else:
            raise ValueError("parameter mode should only be 'explicit' or 'implicit'.")
        return newtm

























