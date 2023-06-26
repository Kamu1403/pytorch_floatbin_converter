# Pytorch float_binary converter

## Introduction
 It seems that it is difficult to use pytorch for bit manipulation of floating point numbers, so a module that can 
 quickly complete pytorch floating point and binary conversion is implemented. **Need pytorch-cuda support.**

## Dependencies
- Pytorch
- Cuda

## How to use

1) Install as a module of python, use `python setup.py install`, then he will be installed as a module named "utils_cuda"
2) Use `utils_cuda.float2bin` and `utils_cuda.bin2float` to convert float32 and int32 to each other.
3) Now that float has been converted to int type, it can be operated using pytorch bit operations. You can refer to [utils.py](./utils.py) as an example.

