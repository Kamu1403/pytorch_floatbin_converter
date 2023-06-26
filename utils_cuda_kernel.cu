#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <ATen/ATen.h>

void float2bin_cuda(
    torch::Tensor from,
    torch::Tensor to) {
    try{
//        float* pp=from.data_ptr<float>();
//        printf("ptr:%p",pp);
//        printf("numel:%d",from.numel());
//        auto shape=from.sizes();
//        auto output=torch::from_blob(ptr,shape,torch::TensorOptions().dtype(torch::kInt));
//        printf("size of float:%d,int:%d,long:%d",sizeof(float),sizeof(int),sizeof(long));
        auto cudaError=cudaMemcpy(to.data_ptr<int>(),from.data_ptr<float>(),sizeof(float)*from.numel(),cudaMemcpyKind(3));
        if (cudaError!=0) {
            printf("Error in cudaMemcpy: %d",cudaError);
        }
        return;
    }catch(const std::exception& e){
        printf("%s\n", e.what());
    }
    return;
}

void bin2float_cuda(
    torch::Tensor from,
    torch::Tensor to) {
    try{
        auto cudaError=cudaMemcpy(to.data_ptr<float>(),from.data_ptr<int>(),sizeof(float)*from.numel(),cudaMemcpyKind(3));
        if (cudaError!=0) {
            printf("Error in cudaMemcpy: %d",cudaError);
        }
        return;
    }catch(const std::exception& e){
        printf("%s\n", e.what());
    }
    return;
}