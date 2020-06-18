Tile Sparsity
===

## A repository for SC2020 Artifact Description.

This repository provides artifacts to reproduce the results of the manuscript, which introduces a pruning method: Tile Sparsity. The artifacts have two major parts: Pruning and Stream Masked GEMM. The file structure shows below:
```
/Pruning: Implemented the EW, VW, BW, and TW pruning pattern.
   |--/VGG:  Pruning VGG16.
   |--/BERT: Pruning BERT.
   |--/NMT:  Pruning NMT.
/StreamMaskedGEMM: The open-source CUDA implementation for Tile Sparsity based on CUTLASS 1.3.
```
Every subfolder has README to explain how to run it.

## Environment 

We prune the networks, VGG, NMT, and BERT on several Linux servers with different GPU versions (GTX 1080, GTX 1080 Ti, RTX 2080 Ti, and Quadro GP100 GPU) without impact on the accuracy of models. The Stream Masked GEMM only runs on the Tesla V100 to profile performance. You can find the detail of the environment [here](https://github.com/clevercool/TileSparsity/environment.txt) collected by this [script](https://github.com/SC-Tech-Program/Author-Kit).

If you have any questions about this repository, please contact guocong@sjtu.edu.cn.

## License
Licensed under an Apache-2.0 license.