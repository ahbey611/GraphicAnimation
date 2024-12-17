#include "collusion.cuh"
#include "ball.hpp"
#include "coor.hpp"

__device__ float CalcLength(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

__device__ float CalcLength(Coor &p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__device__ bool IsCollision(Ball &a, Ball &b)
{
    float dist = 0;
    float distX = a.position.x - b.position.x;
    float distY = a.position.y - b.position.y;
    float distZ = a.position.z - b.position.z;
    dist = CalcLength(distX, distY, distZ);
    // return dist < (a.radius + b.radius);
    if (dist < a.radius + b.radius)
        return true;
    else
        return false;
}

__device__ void WallCollision(Ball &ball, float rangeX, float rangeZ, float rangeY)
{
    if (ball.position.x - ball.radius < -rangeX)
    {
        ball.position.x = -rangeX + ball.radius;
        ball.speed.x = -ball.speed.x;
    }
    else if (ball.position.x + ball.radius > rangeX)
    {
        ball.position.x = rangeX - ball.radius;
        ball.speed.x = -ball.speed.x;
    }
    if (ball.position.z - ball.radius < -rangeZ)
    {
        ball.position.z = -rangeZ + ball.radius;
        ball.speed.z = -ball.speed.z;
    }
    else if (ball.position.z + ball.radius > rangeZ)
    {
        ball.position.z = rangeZ - ball.radius;
        ball.speed.z = -ball.speed.z;
    }
    if (ball.position.y - ball.radius < 0)
    {
        ball.position.y = ball.radius;
        ball.speed.y = -ball.speed.y;
    }
    else if (ball.position.y + ball.radius > rangeY)
    {
        ball.position.y = rangeY - ball.radius;
        ball.speed.y = -ball.speed.y;
    }
}

__device__ void UpdateBallSpeed(Ball &a, Ball &b)
{
    // 处理碰撞
    float dist = 0;
    float diffX = b.position.x - a.position.x;
    float diffY = b.position.y - a.position.y;
    float diffZ = b.position.z - a.position.z;
    dist = CalcLength(diffX, diffY, diffZ);

    float rateCollideA = (a.speed.x * diffX + a.speed.y * diffY + a.speed.z * diffZ) / dist / dist;
    float rateCollideB = (b.speed.x * diffX + b.speed.y * diffY + b.speed.z * diffZ) / dist / dist;

    float speedCollideA_X = diffX * rateCollideA;
    float speedCollideA_Y = diffY * rateCollideA;
    float speedCollideA_Z = diffZ * rateCollideA;

    float speedCollideB_X = diffX * rateCollideB;
    float speedCollideB_Y = diffY * rateCollideB;
    float speedCollideB_Z = diffZ * rateCollideB;

    float unchangedSpeedA_X = a.speed.x - speedCollideA_X;
    float unchangedSpeedA_Y = a.speed.y - speedCollideA_Y;
    float unchangedSpeedA_Z = a.speed.z - speedCollideA_Z;

    float unchangedSpeedB_X = b.speed.x - speedCollideB_X;
    float unchangedSpeedB_Y = b.speed.y - speedCollideB_Y;
    float unchangedSpeedB_Z = b.speed.z - speedCollideB_Z;

    float newSpeedA_X = (speedCollideA_X * (a.weight - b.weight) + 2 * b.weight * speedCollideB_X) / (a.weight + b.weight);
    float newSpeedA_Y = (speedCollideA_Y * (a.weight - b.weight) + 2 * b.weight * speedCollideB_Y) / (a.weight + b.weight);
    float newSpeedA_Z = (speedCollideA_Z * (a.weight - b.weight) + 2 * b.weight * speedCollideB_Z) / (a.weight + b.weight);

    float newSpeedB_X = (speedCollideA_X * 2 * a.weight + (b.weight - a.weight) * speedCollideB_X) / (a.weight + b.weight);
    float newSpeedB_Y = (speedCollideA_Y * 2 * a.weight + (b.weight - a.weight) * speedCollideB_Y) / (a.weight + b.weight);
    float newSpeedB_Z = (speedCollideA_Z * 2 * a.weight + (b.weight - a.weight) * speedCollideB_Z) / (a.weight + b.weight);

    a.speed.x = unchangedSpeedA_X + newSpeedA_X;
    a.speed.y = unchangedSpeedA_Y + newSpeedA_Y;
    a.speed.z = unchangedSpeedA_Z + newSpeedA_Z;

    b.speed.x = unchangedSpeedB_X + newSpeedB_X;
    b.speed.y = unchangedSpeedB_Y + newSpeedB_Y;
    b.speed.z = unchangedSpeedB_Z + newSpeedB_Z;
}

__device__ void UpdateBallStatus(Ball &ball, float refreshInterval, float rangeX, float rangeZ, float rangeY)
{
    ball.position.x += ball.speed.x * refreshInterval;
    ball.position.y += ball.speed.y * refreshInterval;
    ball.position.z += ball.speed.z * refreshInterval;
    WallCollision(ball, rangeX, rangeZ, rangeY);
}

__global__ void UpdateBallsStatus(Ball *balls, float refreshInterval, float rangeX, float rangeZ, float rangeY, int N)
{
    // 更新每个球的状态
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        UpdateBallStatus(balls[i], refreshInterval, rangeX, rangeZ, rangeY);
    }
}

// 空间划分算法
__global__ void InitCells(uint32_t *cells, uint32_t *objects, Ball *balls, float rangeX, float rangeZ, float rangeY, float cellSize, int cellX, int cellY, int cellZ, int N)
{
    unsigned int count = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 这里是使用GPU进行计算的，因此需要使用GPU的线程来计算，一个线程计算一个球
    for (int i = index; i < N; i += stride)
    {
        // 每一个球最多会占在 1个home cell + 7个phantom cell = (8个cell)
        int cellOffset = i * 8; // Fig32-9 Cell ID Array 的第一行黄色的下标

        // Coor pos = balls[i].position;
        float x = balls[i].position.x;
        float y = balls[i].position.y;
        float z = balls[i].position.z;
        float radius = balls[i].radius;

        // 计算小球所在的cell
        int cellIndexX = (int)((x + rangeX) / cellSize);
        int cellIndexY = (int)((y) / cellSize);
        int cellIndexZ = (int)((z + rangeZ) / cellSize);

        // 计算小球的cell信息 + 1bit的home cell标志
        int homeCellInfo = cellIndexX << 17 | cellIndexY << 9 | cellIndexZ << 1 | HOME_CELL;

        // 计算小球的object信息，这里i是 object ID
        int homeObjInfo = i << 1 | HOME_OBJECT;

        // 将小球的cell信息和object信息写入到cells和objects中
        cells[cellOffset] = homeCellInfo;
        objects[cellOffset] = homeObjInfo;

        // 主的cell写完了，接下来处理phantom cell
        cellOffset++;
        count++;

        int phantomCellCount = 1;

        // 遍历附近的cell
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    // 如果dx, dy, dz都为0，则表示是home cell，不需要处理，上面已经处理过了
                    if (!(dx | dy | dz))
                        continue;

                    // 计算phantom cell的下标
                    int pCellIndexX = cellIndexX + dx;
                    int pCellIndexY = cellIndexY + dy;
                    int pCellIndexZ = cellIndexZ + dz;

                    // 检测是否有越界
                    if (pCellIndexX < 0 || pCellIndexX >= cellX || pCellIndexY < 0 || pCellIndexY >= cellY || pCellIndexZ < 0 || pCellIndexZ >= cellZ)
                        continue;

                    // 计算phantom cell的cell信息
                    // int pCellInfo = pCellIndexX << 17 | pCellIndexY << 9 | pCellIndexZ << 1 | PHANTOM_CELL;

                    // // 将phantom cell的cell信息写入到cells中
                    // cells[cellOffset] = pCellInfo;

                    // 计算的是球体中心到相邻网格边界的相对距离
                    float relativeDistanceX = 0;
                    float relativeDistanceY = 0;
                    float relativeDistanceZ = 0;

                    switch (dx)
                    {
                    case -1:
                        relativeDistanceX = cellIndexX * cellSize - rangeX;
                        break;
                    case 0:
                        relativeDistanceX = x;
                        break;
                    case 1:
                        relativeDistanceX = (cellIndexX + 1) * cellSize - rangeX;
                        break;
                    }

                    switch (dy)
                    {
                    case -1:
                        relativeDistanceY = cellIndexY * cellSize;
                        break;
                    case 0:
                        relativeDistanceY = y;
                        break;
                    case 1:
                        relativeDistanceY = (cellIndexY + 1) * cellSize;
                        break;
                    }

                    switch (dz)
                    {
                    case -1:
                        relativeDistanceZ = cellIndexZ * cellSize - rangeZ;
                        break;
                    case 0:
                        relativeDistanceZ = z;
                        break;
                    case 1:
                        relativeDistanceZ = (cellIndexZ + 1) * cellSize - rangeZ;
                        break;
                    }

                    relativeDistanceX -= x;
                    relativeDistanceY -= y;
                    relativeDistanceZ -= z;

                    float length = CalcLength(relativeDistanceX, relativeDistanceY, relativeDistanceZ);

                    // 如果这个距离小于球体半径，说明小球的边缘处在了对应的相邻网格里，需要在该网格中创建一个"幻影"记录
                    if (length < radius)
                    {
                        int pCellInfo = pCellIndexX << 17 | pCellIndexY << 9 | pCellIndexZ << 1 | PHANTOM_CELL;
                        int pObjInfo = i << 1 | PHANTOM_OBJECT; // 这里i是 object ID
                        cells[cellOffset] = pCellInfo;
                        objects[cellOffset] = pObjInfo;
                        cellOffset++;
                        count++;
                        phantomCellCount++;
                    }
                }
            }
        }

        // 如果phantom cell没有填满，则需要填充0
        while (phantomCellCount < 8)
        {
            cells[cellOffset] = EMPTY_CELL;
            objects[cellOffset] = i << 2;
            cellOffset++;
            phantomCellCount++;
        }
    }
}

// 并行计算前缀和
__device__ void GetPrefixSum(uint32_t *radixSums, unsigned int nBits)
{
    int offset = 1;
    int a;
    uint32_t temp;

    // reduction
    for (int d = nBits / 2; d; d /= 2)
    {
        __syncthreads();

        if (threadIdx.x < d)
        {
            a = (threadIdx.x * 2 + 1) * offset - 1;
            radixSums[a + offset] += radixSums[a];
        }

        offset *= 2;
    }

    if (!threadIdx.x)
    {
        radixSums[nBits - 1] = 0;
    }

    // reverse
    for (int d = 1; d < nBits; d *= 2)
    {
        __syncthreads();
        offset /= 2;

        if (threadIdx.x < d)
        {
            a = (threadIdx.x * 2 + 1) * offset - 1;
            temp = radixSums[a];
            radixSums[a] = radixSums[a + offset];
            radixSums[a + offset] += temp;
        }
    }
}

// 计算每个radix的sum
__global__ void GetRedixSum(uint32_t *cells, uint32_t *radixSums, int N, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int numRadices = 1 << RADIX_BITS_STRIDE;

    // 0 至 255 （8位）
    for (int i = index; i < numRadices; i++)
        radixSums[i] = 0;

    // 同步线程
    __syncthreads();

    /*
    例如：对于0b10110101，shift=0时 （以RADIX_BITS_STRIDE=4为例）
    cells[i] >> shift = 0b10110101
    num_indices - 1 = 0b1111
    & 操作后 = 0b0101 (5)
     */

    for (int i = index; i < N; i += stride)
        for (int j = 0; j < blockDim.x; j++)
            if (threadIdx.x % blockDim.x == j)
                radixSums[(cells[i] >> offset) & (numRadices - 1)]++;

    // 同步线程
    __syncthreads();

    GetPrefixSum(radixSums, numRadices);
    __syncthreads();
}

// 使用radixSums 与 前缀和的结果，将cells和objects重新排列
__global__ void RearrangeCells(uint32_t *cells, uint32_t *cellsTemp, uint32_t *objects, uint32_t *objectsTemp, uint32_t *radixSums, int nBits, int N)
{
    // 判断是串行还是并行的
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断是串行还是并行的
    if (index != 0)
        return;

    int numRadices = 1 << RADIX_BITS_STRIDE;

    for (int i = 0; i < N; i++)
    {
        int radix = (cells[i] >> nBits) & (numRadices - 1);
        uint32_t index = radixSums[radix];
        cellsTemp[index] = cells[i];
        objectsTemp[index] = objects[i];
        radixSums[radix]++;
    }
}

// 交换cells和cellsTemp，objects和objectsTemp
void SwapCells(uint32_t *cells, uint32_t *cellsTemp, uint32_t *objects, uint32_t *objectsTemp)
{
    uint32_t *temp = cells;
    cells = cellsTemp;
    cellsTemp = temp;
    temp = objects;
    objects = objectsTemp;
    objectsTemp = temp;
}

// 获取Figure 32-19 的 “Cell index & Size Array”
__global__ void GetCellIndices(uint32_t *cells, uint32_t *indices, uint32_t *indicesNum, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断是串行还是并行的
    if (index != 0)
        return;

    // uint32_t indicesNum = 0;
    indicesNum[0] = 0;

    uint32_t mask = (1 << 24) - 1; // 为什么是1<<24？ 因为cell的hash值是24位，最高的8位是不用的
    uint32_t prev = EMPTY_CELL;    // 空cell
    uint32_t curr = EMPTY_CELL;

    for (int i = 0; i < N; i++)
    {
        // 获取cell的hash值，最低的一位是 HOME_CELL 或 PHANTOM_CELL 标志位
        curr = (cells[i] >> 1) & mask;

        // 如果prev为EMPTY_CELL，则说明是第一个元素
        if (prev == EMPTY_CELL)
            prev = curr;

        // 如果curr与prev不相等，则说明是新的cell
        if (curr != prev)
        { // 记录下标
            indices[indicesNum[0]] = i;
            indicesNum[0]++;
        }

        // 更新prev
        prev = curr;
    }
    indices[indicesNum[0]] = N;
    indicesNum[0]++;

    return;
}

void RadixSortCells(uint32_t *cells, uint32_t *objects,
                    uint32_t *cellsTemp, uint32_t *objectsTemp,
                    uint32_t *radixSums, uint32_t *indices, uint32_t *indicesNum, int N,
                    unsigned int blocksNum,
                    unsigned int threadsPerBlock)

{
    uint32_t *cellsSwap;
    uint32_t *objectsSwap;
    for (int i = 0; i < 32; i += RADIX_BITS_STRIDE)
    {
        GetRedixSum<<<blocksNum, threadsPerBlock>>>(cells, radixSums, N, i);
        RearrangeCells<<<blocksNum, threadsPerBlock>>>(cells, cellsTemp, objects, objectsTemp, radixSums, i, N);
        // SwapCells(cells, cellsTemp, objects, objectsTemp);
        // uint32_t *temp = cells;
        cellsSwap = cells;
        cells = cellsTemp;
        cellsTemp = cellsSwap;
        objectsSwap = objects;
        objects = objectsTemp;
        objectsTemp = objectsSwap;
    }
    GetCellIndices<<<blocksNum, threadsPerBlock>>>(cells, indices, indicesNum, N);
}

__global__ void HandleCollision(uint32_t *cells, int cellsNum, uint32_t *objects,
                                Ball *balls, int ballsNum,
                                uint32_t *indices, uint32_t indicesSize,
                                float rangeX, float rangeZ, float rangeY,
                                int cellSize, int cellX, int cellY, int cellZ,
                                unsigned int blocksNum,
                                unsigned int threadsPerBlock)
{
    unsigned int TOTAL_THREADS = blocksNum * threadsPerBlock;
    unsigned int GROUP_PER_THREAD = indicesSize / TOTAL_THREADS + 1;

    // printf("handle collision\n");

    // 获取当前线程的index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int group = 0; group < GROUP_PER_THREAD; group++)
    {
        int cellIndex = index * GROUP_PER_THREAD + group;
        // 如果index大于indicesSize，则说明已经处理完了
        if (cellIndex >= indicesSize)
            break;

        // printf("group: %d, cellIndex: %d\n", group, cellIndex);

        int start = 0;
        int end = indices[cellIndex];

        if (cellIndex == 0)
            start = 0;
        else
            start = indices[cellIndex - 1];

        int homeCellNum = 0; // 连续的home cell的个数
        for (int i = start; i < end; i++)
        {
            // 最低位是1，表示是home cell
            int type = cells[i] & 1;
            if (type == HOME_CELL)
                homeCellNum++;
            else
                break;
        }

        // printf("homeCellNum: %d\n", homeCellNum);

        for (int i = start; i < start + homeCellNum; i++)
        {
            if (cells[i] == EMPTY_CELL)
                break;

            // int objectIndex1 = objects[i];
            int ballIndex1 = (objects[i] >> 1) & 65535;

            for (int j = i + 1; j < end; j++)
            {
                if (cells[j] == EMPTY_CELL)
                    break;

                // int objectIndex2 = objects[j];
                int ballIndex2 = (objects[j] >> 1) & 65535;

                // printf("???\n");

                // 两个球在同一个home cell里面，则需要处理碰撞
                if (j < start + homeCellNum)
                {
                    // printf("balls[ballIndex1].speed: %f, balls[ballIndex2].speed: %f\n", balls[ballIndex1].speed, balls[ballIndex2].speed);

                    // 如果两个球发生碰撞，则更新速度
                    if (IsCollision(balls[ballIndex1], balls[ballIndex2]))
                        UpdateBallSpeed(balls[ballIndex1], balls[ballIndex2]);
                }
                else
                {
                    int homeIndexI = (cells[i] >> 1) & ((1 << 24) - 1);
                    int jX = (balls[ballIndex2].position.x + rangeX) / cellSize;
                    int jY = (balls[ballIndex2].position.y) / cellSize;
                    int jZ = (balls[ballIndex2].position.z + rangeZ) / cellSize;
                    int homeIndexJ = jX << 16 | jY << 8 | jZ;

                    if (homeIndexI < homeIndexJ)
                    {
                        // printf("balls[ballIndex1].speed: %f, balls[ballIndex2].speed: %f\n", balls[ballIndex1].speed, balls[ballIndex2].speed);

                        if (IsCollision(balls[ballIndex1], balls[ballIndex2]))
                            UpdateBallSpeed(balls[ballIndex1], balls[ballIndex2]);
                    }
                }
            }
        }
    }
}

void SpatialSubdivision(Ball *balls,
                        float refreshInterval,
                        float rangeX, float rangeZ, float rangeY,
                        float cellSize, int cellX, int cellY, int cellZ,
                        int N,
                        unsigned int blocksNum,
                        unsigned int threadsPerBlock)
{
    // 见官网Fig 32-9，Cell ID Array中的每个小球最多会占据8个cell里面，每一个cell有32bit，即info的hash值
    unsigned int totalCellSize = N * 8 * sizeof(uint32_t);
    // Cell ID Array：对于每一个球，有1个 home cell + max 7个phantom cell （描述小球所在的cell）
    uint32_t *cells;
    uint32_t *cellsTemp;
    // Object ID Array：对于每一个球，包含小球的ID，还有d + 2^d control bits
    uint32_t *objects;
    uint32_t *objectsTemp;
    //
    uint32_t *indices;
    uint32_t *indicesNum;
    uint32_t *radixSums;

    int numRadices = 1 << RADIX_BITS_STRIDE;

    cudaMalloc((void **)&cells, totalCellSize);
    cudaMalloc((void **)&cellsTemp, totalCellSize);
    cudaMalloc((void **)&objects, totalCellSize);
    cudaMalloc((void **)&objectsTemp, totalCellSize);
    cudaMalloc((void **)&indices, totalCellSize);                   // Figure 32-19 的 “Cell index & Size Array”
    cudaMalloc((void **)&indicesNum, sizeof(uint32_t));             // 记录indices的个数，不是数组，而是1个uint32_t
    cudaMalloc((void **)&radixSums, numRadices * sizeof(uint32_t)); // 大小为256的数组，记录0~255每一位的sum

    // 初始化
    InitCells<<<blocksNum, threadsPerBlock, threadsPerBlock * sizeof(unsigned int)>>>(cells, objects, balls, rangeX, rangeZ, rangeY, cellSize, cellX, cellY, cellZ, N);

    // Radix Sort
    RadixSortCells(cells, objects,
                   cellsTemp, objectsTemp,
                   radixSums, indices, indicesNum, N * 8,
                   blocksNum,
                   threadsPerBlock);

    // 获取indicesNum
    uint32_t indicesSize;
    cudaMemcpy((void *)&indicesSize, (void *)indicesNum, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 处理碰撞
    HandleCollision<<<blocksNum, threadsPerBlock>>>(cells, N * 8, objects,
                                                    balls, N,
                                                    indices, indicesSize,
                                                    rangeX, rangeZ, rangeY,
                                                    cellSize, cellX, cellY, cellZ,
                                                    blocksNum,
                                                    threadsPerBlock);

    // printf("indicesSize: %d\n", indicesSize);

    // 释放内存
    cudaFree(cells);
    cudaFree(cellsTemp);
    cudaFree(objects);
    cudaFree(objectsTemp);
    cudaFree(indices);
    cudaFree(indicesNum);
    cudaFree(radixSums);
}

__global__ void testSync()
{
    __syncthreads();
    // printf("testSync\n");
    return;
}

void CollisionDetection(Ball *balls, float refreshInterval,
                        float rangeX, float rangeZ, float rangeY,
                        float cellSize, int cellX, int cellY, int cellZ,
                        int N)
{
    // 计算需要分配的线程块数量
    unsigned int blocksNum = BLOCKS_NUM;
    unsigned int objSize = (N - 1) / THREADS_PER_BLOCK + 1;
    if (objSize < blocksNum)
        blocksNum = objSize;

    Ball *ballsGPU;

    // 分配内存和拷贝数据到GPU
    unsigned int nBytes = N * sizeof(Ball);
    cudaMalloc((void **)&ballsGPU, nBytes);
    cudaMemcpy((void *)ballsGPU, (void *)balls, nBytes, cudaMemcpyHostToDevice);

    // 更新球的位置
    UpdateBallsStatus<<<blocksNum, THREADS_PER_BLOCK>>>(ballsGPU, refreshInterval, rangeX, rangeZ, rangeY, N);
    cudaDeviceSynchronize();

    // 处理碰撞
    SpatialSubdivision(ballsGPU,
                       refreshInterval,
                       rangeX, rangeZ, rangeY,
                       cellSize, cellX, cellY, cellZ,
                       N,
                       blocksNum,
                       THREADS_PER_BLOCK);
    cudaDeviceSynchronize();

    // 将更新后的数据拷贝回CPU
    cudaMemcpy((void *)balls, (void *)ballsGPU, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(ballsGPU);
}