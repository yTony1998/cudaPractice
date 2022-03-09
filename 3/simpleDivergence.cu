#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
__global__ void mathkernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if(tid %2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a+b;
}
__global__ void mathkernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a , b;
    a = b = 0.0f;
    if( (tid/warpSize)%2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a+b;  
}
__global__ void mathkernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia , ib;
    ia = ib = 0.0f;
    bool ipred = (tid%2 == 0);
    if(ipred)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathkernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x +threadIdx.x;
    float ia , ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;
    if(itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a , b;
    if( (tid/warpSize)%2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a+b;  
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec+(double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv)
{
    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Deveice %d: %s\n", argv[0], dev, deviceProp.name);

    //set up data size
    int size = 64;
    int blocksize = 64;
    if(argc > 1 ) blocksize = atoi(argv[1]);
    if(argc > 2 ) size = atoi(argv[2]);
    printf("Data size %d", size);

    //set up execution configuretion
    dim3 block(blocksize,1);
    dim3 grid((size+block.x-1)/ block.x, 1);
    printf("Execution Configure (block %d grid %d)\n",  block.x, grid.x);

    //allocate gpu memory
    float *d_C;
    size_t nBytes = size*sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    //run a warmup kernel to remove overhead
    size_t iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup   <<<%4d %4d>>> elapsed %f sec \n",grid.x, block.x, iElaps);

    // run kernel 1
    iStart = seconds();
    mathkernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;

    printf("mathkernel1   <<<%4d %4d>>> elapsed %f sec \n",grid.x, block.x, iElaps);


    // run kernel 2
    iStart = seconds();
    mathkernel2<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathkernel2   <<<%4d %4d>>> elapsed %f sec \n",grid.x, block.x, iElaps);

    // run kernel 3
    iStart = seconds();
    mathkernel3<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathkernel3   <<<%4d %4d>>> elapsed %f sec \n",grid.x, block.x, iElaps);

    // run kernek 4
    iStart = seconds();
    mathkernel4<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathkernel4   <<<%4d %4d>>> elapsed %f sec \n",grid.x, block.x, iElaps);

    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
