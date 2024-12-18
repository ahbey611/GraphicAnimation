#include "collusion.cuh"
#include "ball.hpp"

using namespace CollisionConstants;
using namespace SceneConfig;

struct Coord
{
    float x;
    float y;
    float z;

    __device__ __host__ Coord() : x(0), y(0), z(0) {}
    __device__ __host__ Coord(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// 向量运算工具
namespace VectorMath
{

    __device__ float magnitude(float x, float y, float z)
    {
        return sqrtf(x * x + y * y + z * z);
    }

    __device__ float dotProduct(const Coord &a, const Coord &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
}

// 物理碰撞处理
namespace CollisionPhysics
{

    // 处理小球与边界的碰撞
    __device__ void handleBoundaryCollision(Ball &ball)
    {
        //
        if (ball.position.x - ball.radius < -LENGTH)
        {
            ball.position.x = -LENGTH + ball.radius;
            ball.speed.x *= -1;
        }
        else if (ball.position.x + ball.radius > LENGTH)
        {
            ball.position.x = LENGTH - ball.radius;
            ball.speed.x = -ball.speed.x;
        }

        //
        if (ball.position.y - ball.radius < 0)
        {
            ball.position.y = ball.radius;
            ball.speed.y = -ball.speed.y;
        }
        else if (ball.position.y + ball.radius > HEIGHT)
        {
            ball.position.y = HEIGHT - ball.radius;
            ball.speed.y = -ball.speed.y;
        }

        //
        if (ball.position.z - ball.radius < -WIDTH)
        {
            ball.position.z = -WIDTH + ball.radius;
            ball.speed.z = -ball.speed.z;
        }
        else if (ball.position.z + ball.radius > WIDTH)
        {
            ball.position.z = WIDTH - ball.radius;
            ball.speed.z = -ball.speed.z;
        }
    }

    // 检测两个小球是否发生碰撞
    __device__ bool checkBallCollision(const Ball &a, const Ball &b)
    {
        float dx = a.position.x - b.position.x;
        float dy = a.position.y - b.position.y;
        float dz = a.position.z - b.position.z;
        float distance = VectorMath::magnitude(dx, dy, dz);
        return distance < (a.radius + b.radius);
    }

    // 处理两个小球的碰撞
    __device__ void resolveBallCollision(Ball &a, Ball &b)
    {
        // 1. 计算碰撞参数
        Coord relativePos = {
            b.position.x - a.position.x,
            b.position.y - a.position.y,
            b.position.z - a.position.z};

        Coord relativeVel = {
            b.speed.x - a.speed.x,
            b.speed.y - a.speed.y,
            b.speed.z - a.speed.z};

        float dist = VectorMath::magnitude(relativePos.x, relativePos.y, relativePos.z);

        // 2. 计算碰撞法向量（归一化）
        Coord normal = {
            relativePos.x / dist,
            relativePos.y / dist,
            relativePos.z / dist};

        // 3. 计算相对速度在法向量方向的投影
        float velAlongNormal = VectorMath::dotProduct(relativeVel, normal);

        // 4. 如果物体正在分离，不处理碰撞
        if (velAlongNormal > 0)
            return;

        // 5. 计算碰撞参数
        const float restitution = 0.8f; // 弹性系数
        float totalMass = a.weight + b.weight;
        float reducedMass = (a.weight * b.weight) / totalMass;

        // 6. 计算冲量
        float impulse = -(1.0f + restitution) * velAlongNormal * reducedMass;

        // 7. 分解冲量到各个方向
        Coord impulseVector = {
            impulse * normal.x,
            impulse * normal.y,
            impulse * normal.z};

        // 8. 应用冲量，更新速度
        float massRatioA = a.weight / totalMass;
        float massRatioB = b.weight / totalMass;

        a.speed.x -= impulseVector.x * massRatioB;
        a.speed.y -= impulseVector.y * massRatioB;
        a.speed.z -= impulseVector.z * massRatioB;

        b.speed.x += impulseVector.x * massRatioA;
        b.speed.y += impulseVector.y * massRatioA;
        b.speed.z += impulseVector.z * massRatioA;

        // 9. 处理重叠
        float overlap = (a.radius + b.radius) - dist;
        if (overlap > 0)
        {
            float separationX = normal.x * overlap * 0.5f;
            float separationY = normal.y * overlap * 0.5f;
            float separationZ = normal.z * overlap * 0.5f;

            a.position.x -= separationX;
            a.position.y -= separationY;
            a.position.z -= separationZ;

            b.position.x += separationX;
            b.position.y += separationY;
            b.position.z += separationZ;
        }
    }

    // 更新全部小球的状态
    __global__ void UpdateBallsStatus(Ball *balls, int N)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < N; i += stride)
        {
            // UpdateBallStatus(balls[i]);
            balls[i].position.x += balls[i].speed.x * DELTA_TIME;
            balls[i].position.y += balls[i].speed.y * DELTA_TIME;
            balls[i].position.z += balls[i].speed.z * DELTA_TIME;
            handleBoundaryCollision(balls[i]);
        }
    }
}

// 空间划分算法
namespace SpatialHashing
{
    struct CellData
    {
        // Cell ID Array：对于每一个球，有1个 home cell + max 7个phantom cell （描述小球所在的cell）
        uint32_t *cells;
        // Object ID Array：对于每一个球，包含小球的ID，还有d + 2^d control bits
        uint32_t *objects;
        // 用以排序
        uint32_t *tempCells;
        uint32_t *tempObjects;
        // Figure 32-19 的 “Cell index & Size Array”
        uint32_t *indices;
        //  记录indices的个数，不是数组，而是1个uint32_t
        uint32_t *indexCount;
        // 大小为256的数组，记录0~255每一位的sum
        uint32_t *radixSums;
    };

    // 用于计算网格索引
    __device__ void calculateGridIndex(const Point3D &position, float cellSize,
                                       int &gridX, int &gridY, int &gridZ)
    {
        gridX = (int)((position.x + LENGTH) / cellSize);
        gridY = (int)(position.y / cellSize);
        gridZ = (int)((position.z + WIDTH) / cellSize);
    }

    // 用于计算cell信息
    __device__ uint32_t calculateCellInfo(int x, int y, int z, bool isHome)
    {
        return (uint32_t)(x << 17 | y << 9 | z << 1 | (isHome ? HOME_CELL : PHANTOM_CELL));
    }

    // 初始化
    __global__ void InitCells(CellData *cellData, Ball *balls, float cellSize, int cellX, int cellY, int cellZ, int N)
    {
        unsigned int count = 0;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        // 这里是使用GPU进行计算的，因此需要使用GPU的线程来计算，一个线程计算一个球
        for (int i = index; i < N; i += stride)
        {
            int cellOffset = i * 8;
            Ball &ball = balls[i];

            // 计算home cell位置
            int homeX, homeY, homeZ;
            calculateGridIndex(ball.position, cellSize, homeX, homeY, homeZ);

            // 设置home cell
            cellData->cells[cellOffset] = calculateCellInfo(homeX, homeY, homeZ, true);
            cellData->objects[cellOffset] = i << 1 | HOME_OBJECT;

            // 处理phantom cells
            int phantomCount = 1;
            cellOffset++;

            // 遍历相邻网格
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        if (!(dx | dy | dz))
                            continue; // 跳过home cell

                        int newX = homeX + dx;
                        int newY = homeY + dy;
                        int newZ = homeZ + dz;

                        // 检查边界
                        if (newX < 0 || newX >= cellX ||
                            newY < 0 || newY >= cellY ||
                            newZ < 0 || newZ >= cellZ)
                            continue;

                        // 计算到相邻网格的距离
                        Coord gridCenter = {
                            (newX * cellSize) - LENGTH,
                            newY * cellSize,
                            (newZ * cellSize) - WIDTH};

                        Coord relativePos = {
                            gridCenter.x - ball.position.x,
                            gridCenter.y - ball.position.y,
                            gridCenter.z - ball.position.z};

                        if (VectorMath::magnitude(relativePos.x, relativePos.y, relativePos.z) < ball.radius)
                        {
                            cellData->cells[cellOffset] = calculateCellInfo(newX, newY, newZ, false);
                            cellData->objects[cellOffset] = i << 1 | PHANTOM_OBJECT;
                            cellOffset++;
                            phantomCount++;
                        }
                    }
                }
            }

            // 填充剩余的slots
            while (phantomCount < 8)
            {
                cellData->cells[cellOffset] = EMPTY_CELL;
                cellData->objects[cellOffset] = i << 1;
                cellOffset++;
                phantomCount++;
            }
        }
    }

    // 并行计算前缀和
    __device__ void ParallelPrefixSum(uint32_t *sums, unsigned int count)
    {
        unsigned int offset = 1;

        // Up-sweep phase
        for (int d = count >> 1; d > 0; d >>= 1)
        {
            __syncthreads();
            if (threadIdx.x < d)
            {
                unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
                unsigned int bi = offset * (2 * threadIdx.x + 2) - 1;
                sums[bi] += sums[ai];
            }
            offset *= 2;
        }

        // Down-sweep phase
        if (threadIdx.x == 0)
            sums[count - 1] = 0;

        for (int d = 1; d < count; d *= 2)
        {
            offset >>= 1;
            __syncthreads();
            if (threadIdx.x < d)
            {
                unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
                unsigned int bi = offset * (2 * threadIdx.x + 2) - 1;
                uint32_t temp = sums[ai];
                sums[ai] = sums[bi];
                sums[bi] += temp;
            }
        }
    }

    // 计算每个radix的sum
    __global__ void GetRedixSum(CellData *cellData, int N, int offset)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int numRadices = 1 << SORT_BITS_PER_PASS;

        // 0 至 255 （8位）
        for (int i = index; i < numRadices; i++)
            cellData->radixSums[i] = 0;

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
                    cellData->radixSums[(cellData->cells[i] >> offset) & (numRadices - 1)]++;

        // 同步线程
        __syncthreads();

        ParallelPrefixSum(cellData->radixSums, numRadices);
        __syncthreads();
    }

    // 使用radixSums 与 前缀和的结果，将cells和objects重新排列
    __global__ void RearrangeCells(CellData *cellData, int N, int offset)
    {
        // 判断是串行还是并行的
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // 判断是串行还是并行的
        if (index != 0)
            return;

        int numRadices = 1 << SORT_BITS_PER_PASS;

        for (int i = 0; i < N; i++)
        {
            int radix = (cellData->cells[i] >> offset) & (numRadices - 1);
            uint32_t index = cellData->radixSums[radix];
            cellData->tempCells[index] = cellData->cells[i];
            cellData->tempObjects[index] = cellData->objects[i];
            cellData->radixSums[radix]++;
        }
    }

    // 获取Figure 32-19 的 “Cell index & Size Array”
    __global__ void GetCellIndices(CellData *cellData, int N)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // 判断是串行还是并行的
        if (index != 0)
            return;

        cellData->indexCount[0] = 0;

        uint32_t mask = (1 << 24) - 1; // 为什么是1<<24？ 因为cell的hash值是24位，最高的8位是不用的
        uint32_t prev = EMPTY_CELL;    // 空cell
        uint32_t curr = EMPTY_CELL;

        for (int i = 0; i < N; i++)
        {
            // 获取cell的hash值，最低的一位是 HOME_CELL 或 PHANTOM_CELL 标志位
            curr = (cellData->cells[i] >> 1) & mask;

            // 如果prev为EMPTY_CELL，则说明是第一个元素
            if (prev == EMPTY_CELL)
                prev = curr;

            // 如果curr与prev不相等，则说明是新的cell
            if (curr != prev)
            { // 记录下标
                cellData->indices[cellData->indexCount[0]] = i;
                cellData->indexCount[0]++;
            }

            // 更新prev
            prev = curr;
        }
        cellData->indices[cellData->indexCount[0]] = N;
        cellData->indexCount[0]++;

        return;
    }

    void RadixSortCells(CellData *cellData, int N,
                        unsigned int blocksNum)

    {
        CellData hostCellData;

        // 每次迭代前，先把当前的 GPU 数据结构复制到 CPU
        cudaMemcpy(&hostCellData, cellData, sizeof(CellData), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 32; i += SORT_BITS_PER_PASS)
        {
            GetRedixSum<<<blocksNum, THREAD_COUNT>>>(cellData, N, i);
            RearrangeCells<<<blocksNum, THREAD_COUNT>>>(cellData, N, i);
            uint32_t *temp = hostCellData.cells;
            hostCellData.cells = hostCellData.tempCells;
            hostCellData.tempCells = temp;
            temp = hostCellData.objects;
            hostCellData.objects = hostCellData.tempObjects;
            hostCellData.tempObjects = temp;

            // 将更新后的结构体复制回 GPU
            cudaMemcpy(cellData, &hostCellData, sizeof(CellData), cudaMemcpyHostToDevice);
        }
        GetCellIndices<<<blocksNum, THREAD_COUNT>>>(cellData, N);
    }

    __global__ void HandleCollision(CellData *cellData, uint32_t indicesSize,
                                    Ball *balls, int ballsNum,
                                    int cellSize, int cellX, int cellY, int cellZ,
                                    unsigned int blocksNum)
    {
        unsigned int TOTAL_THREADS = blocksNum * THREAD_COUNT;
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
            int end = cellData->indices[cellIndex];

            if (cellIndex == 0)
                start = 0;
            else
                start = cellData->indices[cellIndex - 1];

            int homeCellNum = 0; // 连续的home cell的个数
            for (int i = start; i < end; i++)
            {
                // 最低位是1，表示是home cell
                int type = cellData->cells[i] & 1;
                if (type == HOME_CELL)
                    homeCellNum++;
                else
                    break;
            }

            // printf("homeCellNum: %d\n", homeCellNum);

            for (int i = start; i < start + homeCellNum; i++)
            {
                if (cellData->cells[i] == EMPTY_CELL)
                    break;

                // int objectIndex1 = objects[i];
                int ballIndex1 = (cellData->objects[i] >> 1) & 65535;

                for (int j = i + 1; j < end; j++)
                {
                    if (cellData->cells[j] == EMPTY_CELL)
                        break;

                    // int objectIndex2 = objects[j];
                    int ballIndex2 = (cellData->objects[j] >> 1) & 65535;

                    // printf("???\n");

                    // 两个球在同一个home cell里面，则需要处理碰撞
                    if (j < start + homeCellNum)
                    {
                        // printf("balls[ballIndex1].speed: %f, balls[ballIndex2].speed: %f\n", balls[ballIndex1].speed, balls[ballIndex2].speed);

                        // 如果两个球发生碰撞，则更新速度
                        if (CollisionPhysics::checkBallCollision(balls[ballIndex1], balls[ballIndex2]))
                            CollisionPhysics::resolveBallCollision(balls[ballIndex1], balls[ballIndex2]);
                    }
                    else
                    {
                        int homeIndexI = (cellData->cells[i] >> 1) & ((1 << 24) - 1);
                        int jX = (balls[ballIndex2].position.x + LENGTH) / cellSize;
                        int jY = (balls[ballIndex2].position.y) / cellSize;
                        int jZ = (balls[ballIndex2].position.z + WIDTH) / cellSize;
                        int homeIndexJ = jX << 16 | jY << 8 | jZ;

                        if (homeIndexI < homeIndexJ)
                        {
                            // printf("balls[ballIndex1].speed: %f, balls[ballIndex2].speed: %f\n", balls[ballIndex1].speed, balls[ballIndex2].speed);

                            if (CollisionPhysics::checkBallCollision(balls[ballIndex1], balls[ballIndex2]))
                                CollisionPhysics::resolveBallCollision(balls[ballIndex1], balls[ballIndex2]);
                        }
                    }
                }
            }
        }
    }

    void SpatialSubdivision(Ball *balls,
                            float cellSize, int cellX, int cellY, int cellZ,
                            int N,
                            unsigned int blocksNum)
    {
        // 见官网Fig 32-9，Cell ID Array中的每个小球最多会占据8个cell里面，每一个cell有32bit，即info的hash值
        // 1. 首先在 CPU 上分配 CellData 结构体内存
        CellData hostCellData;
        CellData *d_cellData;

        // 2. 计算所需的内存大小
        unsigned int totalCellSize = N * 8 * sizeof(uint32_t);
        int numRadices = 1 << SORT_BITS_PER_PASS;

        // 3. 在 GPU 上为 CellData 的各个成员分配内存
        cudaMalloc((void **)&(hostCellData.cells), totalCellSize);
        cudaMalloc((void **)&(hostCellData.tempCells), totalCellSize);
        cudaMalloc((void **)&(hostCellData.objects), totalCellSize);
        cudaMalloc((void **)&(hostCellData.tempObjects), totalCellSize);
        cudaMalloc((void **)&(hostCellData.indices), totalCellSize);
        cudaMalloc((void **)&(hostCellData.indexCount), sizeof(uint32_t));
        cudaMalloc((void **)&(hostCellData.radixSums), numRadices * sizeof(uint32_t));

        // 4. 将 cellData 结构体复制到 GPU
        cudaMalloc((void **)&d_cellData, sizeof(CellData));
        cudaMemcpy(d_cellData, &hostCellData, sizeof(CellData), cudaMemcpyHostToDevice);
        // printf("d_cellData\n");

        // 初始化
        InitCells<<<blocksNum, THREAD_COUNT, THREAD_COUNT * sizeof(unsigned int)>>>(d_cellData, balls, cellSize, cellX, cellY, cellZ, N);
        cudaDeviceSynchronize();

        // Radix Sort
        RadixSortCells(d_cellData, N * 8, blocksNum);
        cudaDeviceSynchronize();

        // 获取indicesNum
        uint32_t indicesSize;
        cudaMemcpy((void *)&indicesSize, (void *)hostCellData.indexCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // 处理碰撞
        HandleCollision<<<blocksNum, THREAD_COUNT>>>(d_cellData, N * 8,
                                                     balls, N,
                                                     cellSize, cellX, cellY, cellZ,
                                                     blocksNum);

        cudaDeviceSynchronize();

        // 释放内存
        cudaFree(hostCellData.cells);
        cudaFree(hostCellData.tempCells);
        cudaFree(hostCellData.objects);
        cudaFree(hostCellData.tempObjects);
        cudaFree(hostCellData.indices);
        cudaFree(hostCellData.indexCount);
        cudaFree(hostCellData.radixSums);
        cudaFree(d_cellData);
    }

}

void ProcessCollisions(
    Ball *ballArray,
    int ballCount,
    float cellSize,
    int cellX,
    int cellY,
    int cellZ)
{
    // 计算需要分配的线程块数量
    unsigned int blocksNum = BLOCK_COUNT;
    unsigned int objSize = (ballCount - 1) / THREAD_COUNT + 1;
    if (objSize < blocksNum)
        blocksNum = objSize;

    Ball *ballsGPU;

    // 分配内存和拷贝数据到GPU
    unsigned int nBytes = ballCount * sizeof(Ball);
    cudaMalloc((void **)&ballsGPU, nBytes);
    cudaMemcpy((void *)ballsGPU, (void *)ballArray, nBytes, cudaMemcpyHostToDevice);

    // 更新球的位置
    CollisionPhysics::UpdateBallsStatus<<<blocksNum, THREAD_COUNT>>>(ballsGPU, ballCount);
    cudaDeviceSynchronize();

    // 处理碰撞
    SpatialHashing::SpatialSubdivision(ballsGPU,
                                       cellSize, cellX, cellY, cellZ,
                                       ballCount,
                                       blocksNum);
    cudaDeviceSynchronize();

    // 将更新后的数据拷贝回CPU
    cudaMemcpy((void *)ballArray, (void *)ballsGPU, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(ballsGPU);
}
