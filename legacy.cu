#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>

//Tenho que aumentar muito
#define SIZE 1000
#define INPUT_SIZE 15

//Desnecessario quando em escala
#define FIRST_AVA 0
#define AVA_LIMITER 10

//Considero por ser um problema de "baixo nivel"
//Nao sei se eh o melhor modelo
#define TYPE_AND 0
#define TYPE_SUM 1
#define TYPE_CON 2
#define TYPE_SET 3
#define TYPE_AVA 4
#define TYPE_INP 5
//type 5 (INP) gives assertion error
__device__ const char NODE_TYPE_LIMITS[]{4, 2, 2, 1, 1};

//Sinto que vai dar problema
#define AVA_NEVER_USED 0
#define AVA_JUST_IN_LIST 1
#define AVA_CALLED_IN_LIST 2
#define AVA_CATCHED 13
#define AVA_TAKED_IN_REVERSE_ORDER 8
#define AVA_TAKED_IN_FORWARD_ORDER 9

//Parece feio, talvez fosse melhor colocar em outro arquivo
#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                                    \
    }

//Eh...
#define BREAK_LINE "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|\n"

//Meio avulso, talvez fosse melhor colocar em outro arquivo
struct Node;
__device__ Node *nodes;
__device__ Node *actualAvaliator;
__device__ bool noMutation = true;

//Tenho de melhorar o uso de memoria
struct Node
{
    char actualIndex;
    char type;
    bool alreadyUsed = false;
    float lastValue;

    //Penso em cortar redundancias mudando o tipo para uint
    Node *childs[4];
    Node *calledBy;

    __device__ int id()
    {
        return (int)(this - nodes);
    }
};

//Isso serve so uma vez
//Nao eh muito claro
//Quando altero a funcao mutar fica confuso
//Talvez seja melhor colocar em outro arquivo
__global__ void initializeRandom()
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    curandState_t state;
    curand_init(clock64(), i * blockDim.x + j, 0, &state);

    nodes[i].childs[j] = &nodes[curand(&state) % SIZE];
    nodes[i].calledBy = &nodes[curand(&state) % (SIZE - INPUT_SIZE)]; //DANGEROUS

    if (i >= SIZE - INPUT_SIZE)
    {
        nodes[i].type = TYPE_INP;
        nodes[i].lastValue = curand(&state) % 5 + 10 + 0.0001; //REAL INPUT GOES HERE
        nodes[i].alreadyUsed = true;                           //DANGEROUS
    }
    else
    {
        int type = curand(&state) % 4;
        nodes[i].type = type;
        //temporario, reducao no numero de avaliators
        if (i % AVA_LIMITER == 0)
            nodes[i].type = TYPE_AVA;

        if (type == TYPE_SUM)
        {
            nodes[i].actualIndex = (curand(&state) % 4) * 2;
        }
        else if (type == TYPE_SET)
        {
            nodes[i].actualIndex = curand(&state) % 4;
        }
    }
}

//Idem
__global__ void startAvaliator()
{
    actualAvaliator = &nodes[FIRST_AVA];
    actualAvaliator->type = TYPE_AVA;
    actualAvaliator->actualIndex = AVA_JUST_IN_LIST;
    //printf("O primeiro AVA eh Node[%d], type %d\n", actualAvaliator->id(), actualAvaliator->type);
}

__global__ void call(Node *caller, int byIndex = -1);

__global__ void receive(Node *receiver, float value)
{
    //Faz sentido, mas custa performance, pode virar so um comentario
    //if (receiver->type == TYPE_INP) printf("ERRO DE ASSERCAO Node[%d] input recebendo\n", receiver->id());

    //Nao eh muito claro
    if (receiver->type != TYPE_AVA && receiver->type != TYPE_SET)
    {
        receiver->actualIndex++;
    }
    //printf("Node[%d], lastValue %f, received %f\n", receiver->id(), receiver->lastValue, value);

    switch (receiver->type)
    {
        //Aqui tem maior ou igual e menor ou igual, por actualIndex variar bastante
    case TYPE_AND:
        if (receiver->actualIndex >= 4)
        {
            receiver->actualIndex = 0;
            receive<<<1, 1>>>(receiver->calledBy, value);
            //printf("Node[%d] (and) recebeu quatro vezes\n", receiver->id());
        }
        break;
    //ISSO PARECE QUE VAI DAR MUITO
    //MUITO PROBLEMA
    //eu sugiro uma operação bit a bit
    case TYPE_SUM:
        //1,2
        if (receiver->actualIndex <= 2)
        {
            receiver->actualIndex = 0;
            receive<<<1, 1>>>(receiver->calledBy, value + receiver->lastValue);
            //printf("Node[%d] (sum) recebeu duas vezes\n", receiver->id());
        }
        //3,4
        else if (receiver->actualIndex == 4)
        {
            receiver->actualIndex = 2;
            receive<<<1, 1>>>(receiver->calledBy, value - receiver->lastValue);
            //printf("Node[%d] (sum) recebeu duas vezes\n", receiver->id());
        }
        //5,6
        else if (receiver->actualIndex == 6)
        {
            receiver->actualIndex = 4;
            receive<<<1, 1>>>(receiver->calledBy, -value + receiver->lastValue);
            //printf("Node[%d] (sum) recebeu duas vezes\n", receiver->id());
        }
        //7,8
        else if (receiver->actualIndex >= 8)
        {
            receiver->actualIndex = 6;
            receive<<<1, 1>>>(receiver->calledBy, -value - receiver->lastValue);
            //printf("Node[%d] (sum) recebeu duas vezes\n", receiver->id());
        }
        break;
    //Um pouco confuso
    case TYPE_CON:
        if (receiver->actualIndex == 2)
        {
            //printf("Node[%d] (con) recebeu duas vezes\n", receiver->id());

            int selectedIndex = receiver->lastValue > value ? 2 : 3;
            call<<<1, 1>>>(receiver, selectedIndex);
            receiver->lastValue = value;
            return;
        }
        else if (receiver->actualIndex >= 3)
        {
            receiver->actualIndex = 0;
            receive<<<1, 1>>>(receiver->calledBy, value);
            //printf("Node[%d] (con) recebeu resposta\n", receiver->id());
        }
        break;
    case TYPE_SET:
        //ISSO PARECE NAO FAZER SENTIDO ALGUM
        //era o que estava escrito em ingles aqui
        receiver->childs[0]->childs[receiver->actualIndex] = receiver->childs[1];
        receive<<<1, 1>>>(receiver->calledBy, value);
        //printf("Node[%d] eh SET, Node[%d] agora tem child %d igual a Node[%d]\n", receiver->id(), receiver->childs[0]->id(), receiver->actualIndex, receiver->childs[1]->id());
        break;
    //Esse parece ok!
    case TYPE_AVA:
        receive<<<1, 1>>>(receiver->calledBy, value);
        //printf("Node[%d] (ava) recebeu uma vez actualIndex %d\n", receiver->id(), receiver->actualIndex);

        if (receiver == actualAvaliator)
        {
            //vai ser testato para salvar ou mutar
            //printf("Node[%d] ACTUAL_AVALIATOR ganhou valor %f\n", receiver->id(), value);
            receiver->lastValue = value;
        }
        receiver->alreadyUsed = false;
        return;
        break;
    }

    //Isso pode ser um problema
    //Minha solucao eh montar um diagrama a parte de como funciona cada tipo
    receiver->lastValue = value;
    receiver->alreadyUsed = false;
}

__device__ void addAsActualAvaNext(Node *newAva)
{
    newAva->actualIndex = AVA_JUST_IN_LIST;
    //adiciona a fila
    newAva->childs[1] = actualAvaliator->childs[1];
    actualAvaliator->childs[1] = newAva;
    //double linked array
    newAva->childs[2] = actualAvaliator;
    newAva->childs[1]->childs[2] = newAva;

    //não sei os efeitos de tirar do calledBy,
    //mas nunca programei essa função e parece funcionar
}

__global__ void call(Node *caller, int byIndex)
{
    caller->alreadyUsed = true;

    //Ava things
    if (caller->type == TYPE_AVA && caller->actualIndex == AVA_NEVER_USED) //AVA nao usado ainda
    {
        addAsActualAvaNext(caller);
    }

    int limit = 0;
    int i = 0;
    if (byIndex == -1)
    {
        limit = NODE_TYPE_LIMITS[caller->type];
    }
    else
    {
        //printf("Node[%d], type %d, chamando so o index %d\n", caller->id(), caller->type, byIndex);
        i = byIndex;
        limit = byIndex + 1;
    }

    for (; i < limit; i++)
    {
        if (!caller->childs[i]->alreadyUsed)
        {
            caller->childs[i]->alreadyUsed = true;
            caller->childs[i]->calledBy = caller;

            if (caller->childs[i]->type == TYPE_INP)
            {
                receive<<<1, 1>>>(caller, caller->childs[i]->lastValue);
                //printf("Node[%p] chama Node[%p] input recebe %f\n", caller, caller->childs[i], caller->childs[i]->lastValue);
                //printf("Node[%d] chama Node[%d] input recebe %f\n", caller->id(), caller->childs[i]->id(), caller->childs[i]->lastValue);
            }
            else
            {
                call<<<1, 1>>>(caller->childs[i]);
                //printf("Node[%d], type %d, call Node[%d]\n", caller->id(), caller->type, caller->childs[i]->id());
            }
        }
        else
        {
            //Eh...
            //DANGEROUS
            receive<<<1, 1>>>(caller, caller->childs[i]->lastValue);
            //printf("Node[%d] tentou chamar Node[%d] usado, recebe %f\n", caller->id(), caller->childs[i]->id(), caller->childs[i]->lastValue);
        }
    }
}

/*
    AGORA EU PERCEBI QUE ISSO ESTA MUITO ACANAIADO
    , OU PELO MENOS PARECE,
    PORQUE PARA PEGAR A LISTA E ORDENAR TEM AQUELE MUIDO TODO
    DE CONFERIR SE FOI ADICIONADO DENTRO DOS CONFORMES,
    SE NAO VAI FICAR ANDANDO EM CIRCULOS,
    SE EH MESMO UMA LISTA DUPLAMENTE LINKADA.

    MAS AQUI, PRA PASSAR DE UM AVA PARA OUTRO
    SO EH CHECADO O TIPO E O ACTUALINDEX

    TA PARECENDO QUE EH SO FALTA DE MEMORIA EXTERNA PRA ORDENAR OS AVA
    EMBORA A PRINCIPIO PARECA CONTRA O PRINCIPIO DE TUDO SER ACESSIVEL DE DENTO
*/

__global__ void callFirst()
{
    actualAvaliator->actualIndex = AVA_CALLED_IN_LIST;
    //printf("|| Chamando PREMERO o avaliator atual Node[% 4d] actualIndex %d% *c|\n", actualAvaliator->id(), actualAvaliator->actualIndex, 35, '|');
    printf(BREAK_LINE);
    call<<<1, 1>>>(actualAvaliator);
}

//Os prints não vão funcionar perfeitamente
__global__ void callActualAvaliator(bool *leftAvaliators)
{
    //printf("\n");
    //printf(BREAK_LINE);
    if (actualAvaliator->childs[1]->type == TYPE_AVA &&
        actualAvaliator->childs[1]->actualIndex == AVA_JUST_IN_LIST)
    {
        //printf("|| Pode passar ");
        actualAvaliator = actualAvaliator->childs[1];
        //Digamos que... (callAvaliator)
        //actualAvaliator->actualIndex = AVA_CALLED_IN_LIST;
        actualAvaliator->actualIndex = AVA_CALLED_IN_LIST;

        __syncthreads();

        //printf("|| Chamando o avaliator atual Node[% 4d] actualIndex %d% *c|\n", actualAvaliator->id(), actualAvaliator->actualIndex, 35, '|');
        //printf(BREAK_LINE);
        call<<<1, 1>>>(actualAvaliator);

        //unnescessary, at this moment
        *leftAvaliators = true;
    }
    else
    {
        //printf("/!\\ Deu ruim passar ");
        *leftAvaliators = false;
    }
    //printf("de Node[%d], type %d actualIndex %d,para Node[%d] type %d actualIndex %d\n", actualAvaliator->id(), actualAvaliator->type, actualAvaliator->actualIndex, actualAvaliator->childs[1]->id(), actualAvaliator->childs[1]->type, actualAvaliator->childs[1]->actualIndex);
}

//NESCESSARY!
__global__ void resetAlreadyUseds()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    nodes[i].alreadyUsed = false;
}

//Confuso: isso precisa ser separado do anterior?
__global__ void resetAvasActualIndexes()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodes[i].type == TYPE_AVA)
        nodes[i].actualIndex = AVA_NEVER_USED;
}

__global__ void mutate(Node *mutater);
//Esta meio dificil de ler, mude isso
__global__ void findByChilds(Node *parent, bool save = true, int deep = 0)
{
    if (parent->actualIndex == AVA_CATCHED)
    {
        //DANGEROUS
        //Pode, por algum acaso, nenhum Bang! acontecer
        //E aí a evolução artificial para
        if (save)
        {
            //printf("SAVE tentou pegar Node[%d] mas tinha index 13\n", parent->id());
        }
        else
        {
            //printf("Bang! tentou pegar Node[%d] mas tinha index 13\n", parent->id());
        }
        return;
    }

    //DANGEROUS
    if (parent->type == TYPE_AVA && parent->actualIndex == AVA_TAKED_IN_FORWARD_ORDER)
    {
        //printf("Na veemente intencao de nao baguncar, Node[%d] AVA foi deixado queto\n", parent->id());
        return;
    }
    if (parent->type == TYPE_INP)
    {
        //printf("Na veemente intencao de nao baguncar, Node[%d] INP foi deixado queto\n", parent->id());
        return;
    }

    parent->actualIndex = AVA_CATCHED;
    if (deep < 3)
    {
        for (int i = 0; i < NODE_TYPE_LIMITS[parent->type]; i++)
        {
            findByChilds<<<1, 1>>>(parent->childs[i], save, deep + 1);
        }
    }
    if (save)
    {
        //printf("SAVE Node[%d] in deep %d actualIndex %d\n", parent->id(), deep, parent->actualIndex);
    }
    else
    {
        //printf("Bang! Node[%d] in deep %d actualIndex %d\n", parent->id(), deep, parent->actualIndex);
        mutate<<<1, 1>>>(parent);
    }
}

__global__ void mutate(Node *mutater)
{
    //contra-intuitivo
    noMutation = false;

    //printf("Mutate Node[%d]\n", mutater->id());
    curandState_t state;
    curand_init(clock64(), 0, 0, &state);

    for (int i = 0; i < 4; i++)
    {
        mutater->childs[i] = &nodes[curand(&state) % SIZE];
    }
    mutater->calledBy = &nodes[curand(&state) % (SIZE - INPUT_SIZE)]; //DANGEROUS

    int type = curand(&state) % 5;
    mutater->type = type;

    //Isso parece ser so um prototipo, uma gambiarra
    //porque nao inclui os outros tipos
    if (type == TYPE_SUM)
    {
        mutater->actualIndex = (curand(&state) % 4) * 2;
    }
    else if (type == TYPE_SET)
    {
        mutater->actualIndex = curand(&state) % 4;
    }
    else
    { //DANGEROUS
        mutater->actualIndex = 0;
    }
}

//JA DEVERIA TER TROCADO ESSES NUMEROS MAGICOS

__device__ void iterateInAvaLogic(Node *ava, bool foward, Node **returnLastAva, int *returnCount, int marker, int rejectMarker = AVA_CALLED_IN_LIST)
{
    int n = 1;
    int next = 1, back = 2;
    if (!foward)
    {
        next = 2;
        back = 1;
    }

    while (true)
    {
        ava->actualIndex = marker;
        if (ava->childs[next]->type != TYPE_AVA)
        {
            //printf("Node[%d] nao eh um avaliator\n", ava->childs[next]->id());
            break;
        }
        if (ava->childs[next]->actualIndex != AVA_CALLED_IN_LIST && ava->childs[1]->actualIndex != rejectMarker) //ISSO PARECE MUITO ERRADO, ESSE NUMERO NO INDEX
        {
            //printf("Node[%d] AVA tem actualIndex %d, invalido\n", ava->childs[next]->id(), ava->childs[next]->actualIndex);
            break;
        }
        //possibily unnescessary
        if (ava->childs[next]->childs[back] != ava)
        {
            //printf("Node[%d] nao esta em cadeia com Node[%d]\n", ava->childs[next]->id(), ava->id());
            break;
        }
        if (ava->childs[next] == ava)
        {
            //printf("Node[%d] tentou chamar Node[%d], a si mesmo dentro do processo valido, uma raridade\n", ava->id(), ava->childs[next]->id());
            break;
        }
        //printf("passa de Node[%d], actualIndex: %d, para Node[%d], actualIndex: %d\n", ava->id(), ava->actualIndex, ava->childs[next]->id(), ava->childs[next]->actualIndex);
        ava = ava->childs[next];
        n++;
    }

    if (returnLastAva)
        *returnLastAva = ava;

    if (returnCount)
        *returnCount = n;
}

__device__ void sortAvaliatorList(Node **avaList, int numberOfAvas)
{
    //Poderia ser um algoritmo paralelo de ordenação
    //E provavelmente depois vai precisar de memória adicional
    //conforme a forma de avaliar vai ficando mais conpleta
    for (int i = 1; i < numberOfAvas; i++)
    {
        for (int j = 0; j < numberOfAvas - i; j++)
        {
            if (avaList[j]->lastValue > avaList[j + 1]->lastValue)
            {
                Node *temp = avaList[j];
                avaList[j] = avaList[j + 1];
                avaList[j + 1] = temp;
            }
        }
    }
}

__device__ void callByPontuation(Node **avaList, int numberOfAvas)
{
    //Esse for pode ser substituido por configurar a funcao para o numero de elementos
    //e em cada chamada verificar qual das duas opçoes se enquadra
    int half = numberOfAvas / 2;
    for (int i = 0; i < half; i++)
    {
        //printf("Salve ava Node[%d], child[0]: Node[%d]\n", avaList[i]->id(), avaList[i]->childs[0]->id());
        findByChilds<<<1, 1>>>(avaList[i]->childs[0], true);
        //printf("Bang ava Node[%d], child[0]: Node[%d]\n", avaList[numberOfAvas - i - 1]->id(), avaList[numberOfAvas - i - 1]->childs[0]->id());
        findByChilds<<<1, 1>>>(avaList[numberOfAvas - i - 1]->childs[0], false);
    }
    if (numberOfAvas != half * 2)
    {
        //printf("E sobra Salve ava Node[%d], child[0]: Node[%d]\n", avaList[half]->id(), avaList[half]->childs[0]->id());
        findByChilds<<<1, 1>>>(avaList[half]->childs[0], true);
    }
}

//Esse tipo de coisa deveria ficar separado?
__device__ void showAvaList(Node **avaList, int numberOfAvas)
{
    printf("\n");
    for (int i = 0; i < numberOfAvas; i++)
    {
        printf("Node[%d], type %d, actualIndex %d, lastValue: %f\n", avaList[i]->id(), avaList[i]->type, avaList[i]->actualIndex, avaList[i]->lastValue);
    }
    printf("\n");
}

__global__ void avaluation()
{
    Node *first;
    int numberOfAvas = 0;
    iterateInAvaLogic(actualAvaliator, false, &first, NULL, AVA_TAKED_IN_REVERSE_ORDER);
    iterateInAvaLogic(first, true, NULL, &numberOfAvas, AVA_TAKED_IN_FORWARD_ORDER, AVA_TAKED_IN_REVERSE_ORDER);
    //printf("The first avaliator: Node[%d]\n", first->id());
    //printf("The number of avas: %d\n", numberOfAvas);

    Node **avaList;
    cudaMalloc(&avaList, sizeof(Node) * numberOfAvas);
    avaList[0] = first;
    for (int i = 1; i < numberOfAvas; i++)
    {
        avaList[i] = avaList[i - 1]->childs[1];
    }

    //provavelmente paralelisavel
    sortAvaliatorList(avaList, numberOfAvas);
    printf("ordened\n");
    //showAvaList(avaList, numberOfAvas);

    //provavelmente paralelisavel
    callByPontuation(avaList, numberOfAvas);

    if (noMutation)
    {
        printf("Forced mutation at Node[%d]\n", avaList[0]->childs[0]->id());
        mutate<<<1, 1>>>(avaList[0]->childs[0]);
    }
    else
    {
        noMutation = true;
    }
}

__global__ void allocNodes()
{
    cudaMalloc(&nodes, sizeof(Node) * SIZE);
}
__global__ void deallocNodes()
{
    cudaFree(&nodes);
}

//Por favor depois revise os nomes
int main()
{
    allocNodes<<<1, 1>>>();
    cudaDeviceSynchronize();
    initializeRandom<<<SIZE, 4>>>();
    cudaDeviceSynchronize();
    startAvaliator<<<1, 1>>>();
    cudaDeviceSynchronize();

    bool *leftAvaliators;
    cudaMallocManaged(&leftAvaliators, sizeof(bool));
    *leftAvaliators = true;

    for (int i = 0; i < 10000; i++)
    {
        callFirst<<<1, 1>>>();
        cudaDeviceSynchronize();
        while (*leftAvaliators)
        {
            resetAlreadyUseds<<<SIZE, 1>>>();
            cudaDeviceSynchronize();
            callActualAvaliator<<<1, 1>>>(leftAvaliators);
            cudaDeviceSynchronize();
        }

        printf("\niteracao %d\n", i);

        avaluation<<<1, 1>>>();
        resetAvasActualIndexes<<<SIZE, 1>>>();
        resetAlreadyUseds<<<SIZE, 1>>>();

        cudaDeviceSynchronize();
        *leftAvaliators = true;
        cudaCheckError();
    }
    deallocNodes<<<1, 1>>>();
    cudaCheckError();
}

//UriSE tenha DETERMINAÇÃO