# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Author: Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import argparse

try:
    input = raw_input
except NameError:
    pass

import open3d as o3d

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import numpy as np
import tqdm

from loguru import logger
import trimesh
# from psbody.mesh import Mesh
from pytorch_sdf import sdf


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mesh-fn', type=str, dest='mesh_fn',
                        help='A mesh file (.obj, .ply, e.t.c.) to be checked' +
                        ' for collisions')
    parser.add_argument('--num-query-points', type=int, default=1,
                        dest='num_query_points',
                        help='Number of random query points')
    parser.add_argument('--seed', type=int, default=None,
                        help='If given then set the seed')

    args, _ = parser.parse_known_args()

    mesh_fn = args.mesh_fn
    num_query_points = args.num_query_points
    seed = args.seed

    # input_mesh = Mesh(filename=mesh_fn)
    input_mesh = trimesh.load(mesh_fn)

    if seed is not None:
        torch.manual_seed(seed)

    logger.info(f'Number of triangles = {input_mesh.faces.shape[0]}')

    v = input_mesh.vertices

    vertices = torch.tensor(v, dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.faces.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    min_vals, _ = torch.min(vertices, dim=0, keepdim=True)
    max_vals, _ = torch.max(vertices, dim=0, keepdim=True)

    batch_size = 3
    query_points = torch.rand([batch_size, num_query_points, 3], dtype=torch.float32,
                              device=device) * (max_vals - min_vals) + min_vals
    query_points = vertices[:num_query_points, :]
    query_points = query_points[None, ].repeat(batch_size, 1, 1)

    # make query points that go from -1 to 1
    # query_points = torch.linspace(-1, 1, num_query_points, device=device).reshape(1, -1, 1).repeat(batch_size, 1, 3)



    triangles = vertices[faces].unsqueeze(dim=0)
    triangles = triangles.expand(batch_size, -1, -1, -1)
    triangles = triangles.contiguous()
    vertices = vertices.unsqueeze(dim=0)
    vertices = vertices.expand(batch_size, -1, -1)
    vertices = vertices.contiguous()
    query_points[:, :, :] += 0.1

    query_points = query_points.requires_grad_()
    query_points_np = query_points[0:1].detach().cpu().numpy().squeeze(
        axis=0).astype(np.float32).reshape(-1, 3)

    gt_dist = torch.norm(query_points - vertices[:, :num_query_points], dim=-1)
    print(query_points.shape, vertices.shape)
    print("gt distance", gt_dist)

    methods = ['bvh', 'knn', 'cdist']
    for method in methods:
        print(f'Running method {method}')
        print(query_points.shape, triangles.shape)
        m = sdf.SDF(distance_method=method)

        torch.cuda.synchronize()
        start = time.perf_counter()
        min_distance, inside = m.sdf_with_winding_numbers(query_points, triangles, vertices)
        torch.cuda.synchronize()
        logger.info(f'CUDA Elapsed time {time.perf_counter() - start}')
        loss = torch.sum(min_distance)
        loss.backward()
        # print(query_points.grad)
        print(min_distance)


        distances = min_distance.detach().cpu().numpy()[0]
        inside = inside.detach().cpu().numpy()[0]
        # closest_points = closest_points.detach().cpu().numpy()[0]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(input_mesh.faces.astype(np.int64))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        query_pcl = o3d.geometry.PointCloud()
        query_pcl.points = o3d.utility.Vector3dVector(
            query_points.detach().cpu().numpy()[0].reshape(-1, 3))
        # paint the points that are inside the mesh
        for inside_point in inside:
            query_pcl.colors.append([0.9, 0.3, 0.3] if inside_point else [0.3, 0.9, 0.3])

        # draw spheres around the query points with radius equal to the distance
        spheres = []
        for i in range(num_query_points):
            spheres.append(o3d.geometry.TriangleMesh.create_sphere(radius=distances[i].item()))
            spheres[-1].translate(query_points_np[i])
            spheres[-1].paint_uniform_color([0.3, 0.9, 0.3] if inside[i] else [0.9, 0.3, 0.3])

        o3d.visualization.draw_geometries([
            mesh,
            query_pcl,
            *spheres,
        ])

