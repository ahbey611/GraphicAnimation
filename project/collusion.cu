#include "collusion.cuh"
#include "ball.hpp"

using namespace CollisionConstants;
using namespace SceneConfig;

struct INTVector3D
{
    int x, y, z;
};

struct CellSizeInfo
{
    INTVector3D cell;
    float size;
};

// 向量运算工具
namespace VectorMath
{
    // 计算向量的模
    __device__ float magnitude(float x, float y, float z)
    {
        return sqrtf(x * x + y * y + z * z);
    }

    // 计算向量的点积
    __device__ float dotProduct(const Vector3D &a, const Vector3D &b)
    {
        return a.dot(b);
    }
}

// 物理碰撞处理
namespace CollisionPhysics
{

    // 处理小球与边界的碰撞
    __device__ void handleBoundaryCollision(Ball &ball)
    {
        // 使用数组来处理三个维度的碰撞检测，使代码更简洁
        const float bounds[3][2] = {
            {-LENGTH, LENGTH}, // X轴边界
            {0, HEIGHT},       // Y轴边界
            {-WIDTH, WIDTH}    // Z轴边界
        };

        float *positions[3] = {&ball.position.x, &ball.position.y, &ball.position.z};
        float *velocities[3] = {&ball.speed.x, &ball.speed.y, &ball.speed.z};

        // 统一处理三个维度的碰撞
        for (int dim = 0; dim < 3; dim++)
        {
            float &pos = *positions[dim];
            float &vel = *velocities[dim];

            // 检查下边界
            if (pos - ball.radius < bounds[dim][0])
            {
                pos = bounds[dim][0] + ball.radius;
                vel = -vel;
                continue;
            }

            // 检查上边界
            if (pos + ball.radius > bounds[dim][1])
            {
                pos = bounds[dim][1] - ball.radius;
                vel = -vel;
            }
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
        Vector3D relativePos(
            b.position.x - a.position.x,
            b.position.y - a.position.y,
            b.position.z - a.position.z);

        Vector3D relativeVel(
            b.speed.x - a.speed.x,
            b.speed.y - a.speed.y,
            b.speed.z - a.speed.z);

        float dist = relativePos.length();

        // 2. 计算碰撞法向量（归一化）
        Vector3D normal = relativePos * (1.0f / dist);

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
        Vector3D impulseVector = normal * impulse;

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
        //  记录indices的个数，不是数组而是1个uint32_t
        uint32_t *indexCount;
        // 大小为256的数组，记录0~255每一位的sum
        uint32_t *radixSums;
    };

    // 用于计算网格索引
    __device__ void calculateGridIndex(const Vector3D &position, float cellSize,
                                       INTVector3D &grid)
    {
        grid.x = (int)((position.x + LENGTH) / cellSize);
        grid.y = (int)(position.y / cellSize);
        grid.z = (int)((position.z + WIDTH) / cellSize);
    }

    // 用于计算cell信息
    __device__ uint32_t calculateCellInfo(INTVector3D &grid, bool isHome)
    {
        return (uint32_t)(grid.x << 17 | grid.y << 9 | grid.z << 1 | (isHome ? HOME_CELL : PHANTOM_CELL));
    }

    // 检查单元格是否在边界内
    __device__ bool isValidCell(INTVector3D &grid, CellSizeInfo cellSizeInfo)
    {
        return grid.x >= 0 && grid.x < cellSizeInfo.cell.x &&
               grid.y >= 0 && grid.y < cellSizeInfo.cell.y &&
               grid.z >= 0 && grid.z < cellSizeInfo.cell.z;
    }

    // 当phantom cell数量小于8时，补齐empty cell
    __device__ int fillEmptyCell(CellData *cellData, int cellOffset, int phantomCount)
    {
        for (int i = phantomCount; i < 8; i++)
        {
            cellData->cells[cellOffset] = EMPTY_CELL;
            cellData->objects[cellOffset] = i << 1;
            cellOffset++;
            phantomCount++;
        }
        return cellOffset;
    }

    // 判断home cell近邻的格子是否是phantom cell
    __device__ int handlePhantomCell(CellData *cellData, Ball &ball, CellSizeInfo cellSizeInfo, int N, INTVector3D &home, int i, int cellOffset)
    {
        int phantomCount = 1;

        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    if (!(x | y | z))
                        continue;

                    INTVector3D newGrid = {home.x + x, home.y + y, home.z + z};

                    // 检查边界
                    if (!isValidCell(newGrid, cellSizeInfo))
                        continue;

                    // 计算到相邻网格的距离
                    Vector3D gridCenter = {
                        (newGrid.x * cellSizeInfo.size) - LENGTH,
                        newGrid.y * cellSizeInfo.size,
                        (newGrid.z * cellSizeInfo.size) - WIDTH};

                    Vector3D relativePos = {
                        gridCenter.x - ball.position.x,
                        gridCenter.y - ball.position.y,
                        gridCenter.z - ball.position.z};

                    if (VectorMath::magnitude(relativePos.x, relativePos.y, relativePos.z) < ball.radius)
                    {
                        cellData->cells[cellOffset] = calculateCellInfo(newGrid, false);
                        cellData->objects[cellOffset] = i << 1 | PHANTOM_OBJECT;
                        cellOffset++;
                        phantomCount++;
                    }
                }
            }
        }

        // 填充剩余的cell
        cellOffset = fillEmptyCell(cellData, cellOffset, phantomCount);

        return cellOffset;
    }

    // 初始化
    __global__ void InitCells(CellData *cellData, Ball *balls, CellSizeInfo cellSizeInfo, int N)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        // 这里是使用GPU进行计算的，因此需要使用GPU的线程来计算，一个线程计算一个球
        for (int i = index; i < N; i += stride)
        {
            int cellOffset = i * 8;
            Ball &ball = balls[i];

            // 计算home cell位置
            INTVector3D home;
            calculateGridIndex(ball.position, cellSizeInfo.size, home);

            // 设置home cell
            cellData->cells[cellOffset] = calculateCellInfo(home, true);
            cellData->objects[cellOffset] = i << 1 | HOME_OBJECT;

            // 处理phantom cells
            cellOffset++;

            // 遍历相邻网格
            cellOffset = handlePhantomCell(cellData, ball, cellSizeInfo, N, home, i, cellOffset);
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

        const uint32_t CELL_MASK = (1 << 24) - 1;
        uint32_t prev = EMPTY_CELL;
        uint32_t curr = EMPTY_CELL;

        for (int i = 0; i < N; i++)
        {
            // 获取cell的hash值，最低的一位是 HOME_CELL 或 PHANTOM_CELL 标志位
            curr = (cellData->cells[i] >> 1) & CELL_MASK;

            // 如果prev为EMPTY_CELL，则说明是第一个元素
            if (prev == EMPTY_CELL)
                prev = curr;

            // 如果curr与prev不相等，则说明是新的cell
            if (curr != prev)
            { // 记录下标
                cellData->indices[cellData->indexCount[0]++] = i;
            }

            // 更新prev
            prev = curr;
        }
        cellData->indices[cellData->indexCount[0]++] = N;
    }

    void RadixSortCells(CellData *cellData, int N,
                        unsigned int blocksNum)
    {

        // radix sort 的位数
        const int TOTAL_BITS = 32;
        // radix sort 的pass数，按每 SORT_BITS_PER_PASS 位进行一次排序
        const int TOTAL_PASSES = TOTAL_BITS / SORT_BITS_PER_PASS;

        CellData hostCellData;

        // 每次迭代前，先把当前的 GPU 数据结构复制到 CPU
        cudaMemcpy(&hostCellData, cellData, sizeof(CellData), cudaMemcpyDeviceToHost);

        for (int pass = 0; pass < TOTAL_PASSES; pass++)
        {
            int bitShift = pass * SORT_BITS_PER_PASS;

            GetRedixSum<<<blocksNum, THREAD_COUNT>>>(cellData, N, bitShift);
            RearrangeCells<<<blocksNum, THREAD_COUNT>>>(cellData, N, bitShift);

            std::swap(hostCellData.cells, hostCellData.tempCells);
            std::swap(hostCellData.objects, hostCellData.tempObjects);

            // 将更新后的结构体复制回 GPU
            cudaMemcpy(cellData, &hostCellData, sizeof(CellData), cudaMemcpyHostToDevice);
        }
        GetCellIndices<<<blocksNum, THREAD_COUNT>>>(cellData, N);
    }

    __global__ void HandleCollision(CellData *cellData, uint32_t indicesSize,
                                    Ball *balls, int ballsNum,
                                    CellSizeInfo cellSizeInfo,
                                    unsigned int blocksNum)
    {
        const unsigned int TOTAL_THREADS = blocksNum * THREAD_COUNT;
        const unsigned int GROUP_PER_THREAD = indicesSize / TOTAL_THREADS + 1;

        // 获取当前线程的index
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int g = 0; g < GROUP_PER_THREAD; g++)
        {
            int cellIndex = index * GROUP_PER_THREAD + g;
            // 如果index大于indicesSize，则说明已经处理完了
            if (cellIndex >= indicesSize)
                break;

            int groupStart = (cellIndex == 0) ? 0 : cellData->indices[cellIndex - 1];
            int groupEnd = cellData->indices[cellIndex];

            // 连续的home cell的个数
            int homeCellCount = 0;

            // 遍历groupStart到groupEnd，统计连续的home cell的个数
            while (homeCellCount < (groupEnd - groupStart) &&
                   (cellData->cells[groupStart + homeCellCount] & 1) == HOME_CELL)
            {
                homeCellCount++;
            }

            for (int i = groupStart; i < groupStart + homeCellCount; i++)
            {
                if (cellData->cells[i] == EMPTY_CELL)
                    break;

                const int ball1Index = (cellData->objects[i] >> 1) & 0xFFFF;

                for (int j = i + 1; j < groupEnd; j++)
                {
                    if (cellData->cells[j] == EMPTY_CELL)
                        break;

                    const int ball2Index = (cellData->objects[j] >> 1) & 0xFFFF;

                    // 两个球在同一个home cell里面，则需要处理碰撞
                    if ((j < groupStart + homeCellCount))
                    {
                        // 如果两个球发生碰撞，则更新速度
                        if (CollisionPhysics::checkBallCollision(balls[ball1Index], balls[ball2Index]))
                            CollisionPhysics::resolveBallCollision(balls[ball1Index], balls[ball2Index]);
                    }
                    else
                    {
                        int cellSize = (int)cellSizeInfo.size;
                        INTVector3D homeIndex2 = {
                            (balls[ball2Index].position.x + LENGTH) / cellSize,
                            balls[ball2Index].position.y / cellSize,
                            (balls[ball2Index].position.z + WIDTH) / cellSize};

                        const uint32_t homeCell2 = (homeIndex2.x << 16 | homeIndex2.y << 8 | homeIndex2.z);
                        const uint32_t homeCell1 = (cellData->cells[i] >> 1) & ((1 << 24) - 1);

                        if (homeCell1 < homeCell2 &&
                            CollisionPhysics::checkBallCollision(balls[ball1Index], balls[ball2Index]))
                            CollisionPhysics::resolveBallCollision(balls[ball1Index], balls[ball2Index]);
                    }
                }
            }
        }
    }

    void SpatialSubdivision(Ball *balls,
                            CellSizeInfo cellSizeInfo,
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

        // 4. 将 cellData 构体复制到 GPU
        cudaMalloc((void **)&d_cellData, sizeof(CellData));
        cudaMemcpy(d_cellData, &hostCellData, sizeof(CellData), cudaMemcpyHostToDevice);
        // printf("d_cellData\n");

        // 初始化
        InitCells<<<blocksNum, THREAD_COUNT, THREAD_COUNT * sizeof(unsigned int)>>>(d_cellData, balls, cellSizeInfo, N);
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
                                                     cellSizeInfo,
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

    CellSizeInfo cellSizeInfo;
    cellSizeInfo.size = cellSize;
    cellSizeInfo.cell.x = cellX;
    cellSizeInfo.cell.y = cellY;
    cellSizeInfo.cell.z = cellZ;

    // 处理碰撞
    SpatialHashing::SpatialSubdivision(ballsGPU,
                                       cellSizeInfo,
                                       ballCount,
                                       blocksNum);
    cudaDeviceSynchronize();

    // 将更新后的数据拷贝回CPU
    cudaMemcpy((void *)ballArray, (void *)ballsGPU, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(ballsGPU);
}
