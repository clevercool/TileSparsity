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
$ ./test_basic_nn 768 768 768 768 768
```
 