#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

#define SIZE 20

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

    //Posso substituir por ints ou shorts, já que nodes é uma memória contígua
    Node *childs[4];
    Node *calledBy;
    Node *parent;

    float weights[4];

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

    if (j == 0)
    {
        nodes[i].calledBy = &nodes[curand(&state) % SIZE]; //impede receive para algo fora da memoria
        nodes[i].parent = &nodes[curand(&state) % SIZE];   //TESTING
        // printf("%d->Node[%d]\n", nodes[i].parent->id(), nodes[i].id());
    }

    nodes[i].lastValue = i * 111; //marcador, exemplo: Node[3].lastvalue = 333
    nodes[i].weights[j] = i;      //marcador, exemplo: Node[3].weight[i] = 3

    nodes[i].alreadyUsed = false; //por segurança

    nodes[i].type = TYPE_AND; //para testes iniciais
}

__global__ void descendent(Node *parent, Node *parallelParent, int limit = 0)
{
    if (parent->alreadyUsed == true || limit >= 1)
        return;

    printf("Parent Node[%d] (%d, %d, %d, %d)\n",
           parent->id(),
           parent->childs[0]->id(),
           parent->childs[1]->id(),
           parent->childs[2]->id(),
           parent->childs[3]->id());

    parent->alreadyUsed = true;

    curandState_t state;
    curand_init(clock64(), 0, 0, &state);
    // Copy in parallel parent
    for (int i = 0; i < 4; i++)
    {
        parallelParent->childs[i] = &nodes[curand(&state) % SIZE];

        // And copy parent->childs[i] status ..
        for (int j = 0; j < 4; j++)
        {
            parallelParent->childs[i]->weights[j] = parent->childs[i]->weights[j];
        }
    }
    printf("Parallel parent Node[%d] (%d, %d, %d, %d)\n",
           parallelParent->id(),
           parallelParent->childs[0]->id(),
           parallelParent->childs[1]->id(),
           parallelParent->childs[2]->id(),
           parallelParent->childs[3]->id());

    // Iterate next steps
    for (int i = 0; i < 4; i++)
    {
        parallelParent->childs[i] = &nodes[curand(&state) % SIZE];
        descendent<<<1, 1>>>(parent->childs[i], parallelParent->childs[i], limit + 1);
        __syncthreads();
    }
}

__global__ void reproduce()
{
    curandState_t state;
    curand_init(clock64(), 0, 0, &state);
    Node *parallelParent = &nodes[curand(&state) % SIZE];
    descendent<<<1, 1>>>(nodes, parallelParent);
}

// setter set node
__device__ bool canSet(Node *setter, Node *node)
{
    // Posso adicionar mais e mais coisa como node->parent->parent == setter->parent
    // a depender de ajuste fino para manter a cadeia
    return (
        node->parent == setter ||
        node->parent == setter->parent ||
        node->parent->parent == setter ||
        node->parent->parent == setter->parent);
}

__global__ void call(Node *caller);

__global__ void receive(Node *receiver, float value)
{
    receiver->actualIndex++;

    // printf("Node[%d] recebeu valor %.0f\n", receiver->id(), value);

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
    receiver->alreadyUsed = false; //allow node to be called several times
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
            if (canSet(caller, caller->childs[i]))
            {
                // printf("%d->Node[%d] can set %d->%d->Node[%d]\n",
                //        caller->parent->id(),
                //        caller->id(),
                //        caller->childs[i]->parent->parent->id(),
                //        caller->childs[i]->parent->id(),
                //        caller->childs[i]->id());
            }
        }
        else
        {
            receive<<<1, 1>>>(caller, caller->childs[i]->lastValue);
            // printf("Node[%d] tentou chamar Node[%d] usado, recebe %.0f\n", caller->id(), caller->childs[i]->id(), caller->childs[i]->lastValue);
        }
    }

    // cudaCheckError();
}

__global__ void callFirst()
{
    call<<<1, 1>>>(nodes);
}

//NESCESSARY!
__global__ void resetAlreadyUseds()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    nodes[i].alreadyUsed = false;
}

__global__ void logNodesLastValues()
{
    for (int i = 0; i < SIZE; i++)
    {
        printf("%d ", (int)nodes[i].weights[0]);
    }
}

int main()
{
    allocNodes<<<1, 1>>>();
    cudaDeviceSynchronize();
    initializeRandom<<<SIZE, 4>>>();
    cudaDeviceSynchronize();
    for (int i = 0; i < 1; i++)
    {
        callFirst<<<1, 1>>>();
        cudaDeviceSynchronize();
        resetAlreadyUseds<<<SIZE, 1>>>();
        cudaDeviceSynchronize();
        reproduce<<<1, 1>>>();
        cudaDeviceSynchronize();
        logNodesLastValues<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    cudaCheckError();
}

//UriSE tenha DETERMINAÇÃO