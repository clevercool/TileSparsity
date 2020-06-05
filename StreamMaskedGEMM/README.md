Stream masked GEMM based on CUTLASS 1.3
===================

## Build and Run
+ Prerequisites: CUDA >= 10.1
```shell
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cd perf
$ make

$ #75% Sparsity GEMM Test for BERT on V100 Tensor Core.
$ #./test_basic_nn <M> <N> <K> <N_pruned> <K_pruned>
$ ./test_basic_nn 12800 768 768 384 384        # 4 times.
CUTLASS GEMM : 252.878403 us.
CUTLASS GEMM : 252.665283 us.
Stream  GEMM : 118.173439 us.

$ ./test_basic_nn 12800 768 3072 384 1536
CUTLASS GEMM : 874.368286 us.
CUTLASS GEMM : 831.650879 us.
Stream  GEMM : 298.622406 us.

$ ./test_basic_nn 12800 3072 768 1536 384
CUTLASS GEMM : 920.689941 us.
CUTLASS GEMM : 849.862732 us.
Stream  GEMM : 417.427826 us.
```

The 75% sparsity stream masked GEMM achieves about 2.26x speedup of CUTLASS dense GEMM  using volta884 API, which is only for the V100 tensor core. The stream masked GEMM can also run on the CUDA core.


## Referrence

[CUTLASS 1.3](https://github.com/NVIDIA/cutlass/tree/v1.3.0)

[FasterTransformer V1](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v1)