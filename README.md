# TensorMatching
CUDA Implementation of Tensor Matching Algorithm

This is a implementation research program I did during the summer of 2015. The goal is to implement the Tensor Matching Algorithm on GPUs using CUDA. 

Parts of the algorithm already parallelized:
* Generate artificial data
* Assign nodes to triangles
* Compute feature
* Tensor construciton
* Compute score of tensor matching

Parts left to parallelize:
* K-nearest neighbor search

Right now the parallel implemetation is only slightly faster than serial implementation. I am working on finishing parallelizing kNN and get some speed-up.
