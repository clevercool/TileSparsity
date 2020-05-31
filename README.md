CUTLASS Profiling
===================

## Build and Run
```shell
$ git clone ...
$ cd SCTest
$ mkdir -p build && cd build
$ cmake -DSM=75 -DCMAKE_BUILD_TYPE=Release ..
$ cd perf
$ make

$ #Base GEMM size for BERT
$ #./test_basic_nn <M> <N> <K> <N_pruned> <K_pruned>
$ ./test_basic_nn 12800 768 768 768 768
$ ./test_basic_nn 12800 768 3072 768 3072
$ ./test_basic_nn 12800 3072 768 3072 768

$ # Naive 75% Sparsity Test
$ ./test_basic_nn 12800 768 768 384 384
$ ./test_basic_nn 12800 768 3072 384 1536
$ ./test_basic_nn 12800 3072 768 1536 384
```
 