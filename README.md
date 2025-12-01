# Diferentiable Sign Distance Field (SDF) for pytorch
This package provides a PyTorch module that returns the SDF using fast winding numbers. The distance is differentiable and the mesh doesn't need to be watertight!

The package is faster than the Trimesh and [previous](https://github.com/muelea/tuch) implementations. 


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Examples](#examples)


## Description
 This repository provides a PyTorch wrapper around a CUDA kernel that implements Signed Distance Functions (SDF).

 The implementation includes:
 - A CUDA kernel for fast winding numbers, based on the paper [Fast Winding Numbers for Soups and Clouds](https://www.dgp.toronto.edu/projects/fast-winding-numbers/), to determine the sign.
    - This is our cuda kernel re-implementation of fast winding numbers code from [TUCH](https://github.com/muelea/tuch).
 - A Bounding Volume Hierarchy (BVH) for distance calculations, inspired by the paper [Maximizing parallelism in the construction of BVHs, octrees, and k-d trees](https://dl.acm.org/citation.cfm?id=2383801).
    - The code is borrowed from [bvh-distance-queries](https://github.com/YuliangXiu/bvh-distance-queries).

 The process involves:
 1. Given a batch of meshes (triangles = vertices[faces]), a BVH tree is built for each mesh.
 2. The BVH tree is used to get the distance between the queries and the mesh.
 3. The sign of the queries is computed using winding numbers.

## Installation
Before installing anything please make sure to set the environment variable
*$CUDA_SAMPLES_INC* to the path that contains the header `helper_math.h` , which
can be found in the [CUDA Samples repository](https://github.com/NVIDIA/cuda-samples).
To install the module run the following commands:

**1. Install the dependencies**
```Shell
pip install -r requirements.txt
```
**2. Run the *setup.py* script**
```Shell
pip install . --no-build-isolation
```

If you want to modify any part of the code then use the following command:
```Shell
pip install -e . --no-build-isolation
```


## Examples

Our code obtains the SDF using winding numbers to calculate the inside and outside sign, we use 3 ways of getting the distance.

1. **BVH**. Distance between point to mesh.
    ```python
    from pytorch_sdf import sdf

    m = sdf.SDF(distance_method='bvh')

    # query_points = Tensor[B, Q, 3]
    # triangles = Tensor[B, F, 3, 3]
    # vertices = None, Not needed
    signed_distance, inside = m.sdf_with_winding_numbers(query_points, triangles, vertices)
    ```

2. **KNN**. Distance between point to the closest vertex distance, it is slightly faster than BVH, but uses pytorch3d and it doesn't give you the real distance between the point and the mesh.

    ```python
    from pytorch_sdf import sdf

    m = sdf.SDF(distance_method='knn')

    # query_points = Tensor[B, Q, 3]
    # triangles = Tensor[B, F, 3, 3]
    # vertices = Tensor[B, V, 3]
    signed_distance, inside = m.sdf_with_winding_numbers(query_points, triangles, vertices)
    ```

3. **cdist**. Similar to KNN, gives you the point to vertex distance. It uses cdist implementation from pytorch. It has some numerical precision issues.

    ```python
    from pytorch_sdf import sdf

    m = sdf.SDF(distance_method='cdist')

    # query_points = Tensor[B, Q, 3]
    # triangles = Tensor[B, F, 3, 3]
    # vertices = Tensor[B, V, 3]
    signed_distance, inside = m.sdf_with_winding_numbers(query_points, triangles, vertices)
    ```

## Dependencies

1. [PyTorch](https://pytorch.org)


## Example dependencies

1. [open3d](http://www.open3d.org/)
1. [mesh](https://github.com/MPI-IS/mesh)

