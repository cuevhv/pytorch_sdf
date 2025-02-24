# Diferentiable Sign Distance Field (SDF) for pytorch
This package provides a PyTorch module that SDF using fast winding numbers. The distance is differentiable.


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
  * [Examples](#examples)


## Description
This repository provides a PyTorch wrapper around a CUDA kernel that implements SDF. The implementation has a cuda kernel for the fast winding numbers from [Fast Winding Numbers for Soups and Clouds](https://www.dgp.toronto.edu/projects/fast-winding-numbers/) to find the sign, and
BVH [Maximizing parallelism in the construction of BVHs,
octrees, and k-d trees](https://dl.acm.org/citation.cfm?id=2383801) to find the distance. More
specifically, given a batch of meshes (triangles = vertices[faces]) it builds a
BVH tree for each one, which can then be used for distance queries, then, it computes the sign of the queries using winding numbers.

1. The winding numbers is our cuda kernel re-implementation from [TUCH](https://github.com/muelea/tuch).
2. The BVH kernel code is borrowed from [bvh-distance-queries](https://github.com/YuliangXiu/bvh-distance-queries).

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
python setup.py install
```

If you want to modify any part of the code then use the following command:
```Shell
python setup.py build develop
```


## Examples

TODO

## Dependencies

1. [PyTorch](https://pytorch.org)


## Example dependencies

1. [open3d](http://www.open3d.org/)
1. [mesh](https://github.com/MPI-IS/mesh)

