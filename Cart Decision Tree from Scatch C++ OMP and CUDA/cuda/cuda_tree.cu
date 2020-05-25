#include <cuda.h>
#include "cuda_tree.h"
#include <algorithm>
#include <cstdio>

namespace cuda{
  __global__ 
  void sq_error_kernel(double *arr, int n, double avg, float *err) { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
        atomicAdd(err, pow(arr[index] - avg, 2));
    }
  }


  double find_err(double *arr, int arrSize, double avg){
      double *deviceVals;
      int threadsPerBlock = 1024;
      int numBlocks = 1 + (arrSize / threadsPerBlock);
      int size = arrSize * sizeof(double);
      float *deviceErr;
      float err;
      cudaMalloc(&deviceVals, size);
      cudaMalloc(&deviceErr, sizeof(float));
      cudaMemcpy(deviceVals, arr, size, cudaMemcpyHostToDevice);
      cudaMemset(deviceErr, 0, sizeof(float));
    
      sq_error_kernel<<<numBlocks, threadsPerBlock>>>(deviceVals, arrSize, avg, deviceErr);
      cudaDeviceSynchronize();
      cudaMemcpy(&err, deviceErr, sizeof(float), cudaMemcpyDeviceToHost);

      cudaFree(deviceErr);
      cudaFree(deviceVals);
      return err;
  }
}
