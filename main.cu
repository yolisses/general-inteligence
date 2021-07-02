#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

#define SIZE 10

#define TYPE_AND 0

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
    char actualIndex;
    char type;
    bool alreadyUsed = false;
    float lastValue;

    Node *childs[4];
    Node *calledBy;

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

    nodes[i].calledBy = &nodes[curand(&state) % SIZE]; //impede receive para algo fora da memoria

    nodes[i].lastValue = i * 111; //marcador do tipo Node[3].lastvalue = 333

    nodes[i].alreadyUsed = false; //por segurança

    nodes[i].type = TYPE_AND; //para testes iniciais
}

__global__ void call(Node *caller);

__global__ void receive(Node *receiver, float value)
{
    receiver->actualIndex++;

    printf("Node[%d] (and) recebeu valor %.0f\n", receiver->id(), value);

    switch (receiver->type)
    {
    //Aqui tem maior ou igual porquw actualIndex varia bastante
    case TYPE_AND:
        if (receiver->actualIndex >= 4)
        {
            receiver->actualIndex = 0;
            receive<<<1, 1>>>(receiver->calledBy, value);
            // printf("Node[%d] (and) recebeu quatro vezes\n", receiver->id());
        }
        break;
    }
    receiver->lastValue = value;
    // receiver->alreadyUsed = false; //allow node to be called several times
}

__global__ void call(Node *caller)
{
    caller->alreadyUsed = true;

    for (int i = 0; i < 4; i++)
    {
        if (!caller->childs[i]->alreadyUsed)
        {
            caller->childs[i]->alreadyUsed = true;
            caller->childs[i]->calledBy = caller;
            call<<<1, 1>>>(caller->childs[i]);
            // printf("Node[%d], type %d, call Node[%d]\n", caller->id(), caller->type, caller->childs[i]->id());
        }
        else
        {
            receive<<<1, 1>>>(caller, caller->childs[i]->lastValue);
            printf("Node[%d] tentou chamar Node[%d] usado, recebe %.0f\n", caller->id(), caller->childs[i]->id(), caller->childs[i]->lastValue);
        }
    }
}

__global__ void callFirst()
{
    call<<<1, 1>>>(nodes);
}

int main()
{
    allocNodes<<<1, 1>>>();
    cudaDeviceSynchronize();
    initializeRandom<<<SIZE, 4>>>();
    cudaDeviceSynchronize();
    callFirst<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaCheckError();
}

//UriSE tenha DETERMINAÇÃO