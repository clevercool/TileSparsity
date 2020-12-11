
# Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise Sparsity [[arXiv]](https://arxiv.org/abs/2008.13006)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3900188.svg)](https://doi.org/10.5281/zenodo.3900188)
  
```bash
@inproceedings{guo2020accelerating,
  title={Accelerating sparse DNN models without hardware-support via tile-wise sparsity},
  author={Guo, Cong and Hsueh, Bo Yang and Leng, Jingwen and Qiu, Yuxian and Guan, Yue and Wang, Zehuan and Jia, Xiaoying and Li, Xipeng and Guo, Minyi and Zhu, Yuhao},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--15},
  year={2020}
}
```
# Tile Sparsity

This repository provides artifacts to reproduce the results of the paper, which introduces a pruning method: Tile Sparsity. The artifacts have two major parts: Pruning and Stream Masked GEMM. The file structure shows below:
```
/Pruning: Implemented the EW, VW, BW, and TW pruning pattern.
   |--/VGG:  Pruning VGG16.
   |--/BERT: Pruning BERT.
   |--/NMT:  Pruning NMT.
/StreamMaskedGEMM: The open-source CUDA implementation for Tile Sparsity based on CUTLASS 1.3.
```
Every subfolder has README to explain how to run it.

## Environment 

We prune the networks, VGG, NMT, and BERT on several Linux servers with different GPU versions (GTX 1080, GTX 1080 Ti, RTX 2080 Ti, and Quadro GP100 GPU) without impact on the accuracy of models. The Stream Masked GEMM only runs on the Tesla V100 to profile performance. You can find the detail of the environment [here](https://github.com/clevercool/TileSparsity/blob/master/environment.txt) collected by this [script](https://github.com/SC-Tech-Program/Author-Kit).

## License
Licensed under an Apache-2.0 license.