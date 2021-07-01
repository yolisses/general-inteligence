#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

#define SIZE 10

#define cudaCheckError()                                                                       \
    {                                                                                          \
        cudaError_t e = cudaGetLastError();                                                    \
        if (e != cudaSuccess)                                                                  \
        {                                                                                      \
            printf("\nCuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                                      \
    }

struct Node;
__device__ Node *nodes;

struct Node
{
    Node *childs[4];

    __device__ int id()
    {
        return (int)(this - nodes);
    }
};

__global__ void allocNodes()
{
    cudaMalloc(&nodes, sizeof(Node) * SIZE);
}

__global__ void initializeRandom()
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    curandState_t state;
    curand_init(clock64(), i * blockDim.x + j, 0, &state);

    nodes[i].childs[j] = &nodes[curand(&state) % SIZE];

    // printf("nodes[%d].child[%d] = %d\n", i, j, nodes[i].childs[j]->id());
}

int main()
{
    allocNodes<<<1, 1>>>();
    cudaDeviceSynchronize();
    initializeRandom<<<SIZE, 4>>>();
    cudaDeviceSynchronize();

    cudaCheckError();
}

//UriSE tenha DETERMINAÇÃO