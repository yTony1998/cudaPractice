#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

int recursiveReduce(int *data, int const size)
{
    if(size==0) return data[0];
    int const stride = size /2;
    for (int i = 0; i < stride; i++)
    {
        data[i] = data[i] + data[stride +i];
    }
    
    recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x + blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride*=2)
    {
        if((tid%(2*stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int *idata =  g_idata + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride*=2)
    {
        int index = tid * 2 * stride;
        if(index < blockDim.x)
        {
            idata[index] += idata[index+stride];
        }
        __syncthreads();
    }
    if(tid ==0 ) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceIntereaved(int *g_idata, int *g_odata, unsigned int const n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x + blockDim.x;

    int *idata = g_idata + blockIdx.x + blockDim.x;
    if(idx >= n) return;
    for(int stride = blockDim.x / 2; stride > 0; stride>>=1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = idata [0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned const int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if(idx + blockDim.x < n)
        g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }
    if(tid == 0)
        g_odata[blockIdx.x] = idata[0];
}