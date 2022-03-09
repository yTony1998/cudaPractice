#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **agrv)
{
    int nElem = 1024;

    dim3 block(1024);
    dim3 grid((nElem+block.x-1)/block.x);

    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = 512;
    grid.x = ((nElem+block.x -1)/block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);


    block.x = 256;
    grid.x = ((nElem+block.x -1)/block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = 128;
    grid.x = ((nElem+block.x -1)/block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    cudaDeviceReset();
    return 0;
}