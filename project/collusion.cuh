// #ifndef COLLUSION_CUH
// #define COLLUSION_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "ballset.hpp"
#include "ball.hpp"
#include "coor.hpp"

namespace CollisionConstants
{
// Cell 类型标识
#define HOME_CELL 0x00
#define PHANTOM_CELL 0x01
// 特殊值
#define EMPTY_CELL INT32_MAX

    // Object 类型标识
#define HOME_OBJECT 0x01
#define PHANTOM_OBJECT 0x00

    // 排序相关常量
#define SORT_BITS_PER_PASS 8
#define SORT_PASSES 4

    // 并行计算相关常量
#define BLOCK_COUNT 128
#define THREAD_COUNT 512
#define MAX_CELLS_PER_OBJECT 8
}

namespace SceneConfig
{
#define DELTA_TIME 0.02f
#define LENGTH 10.0f
#define HEIGHT 20.0f
#define WIDTH 10.0f
}

void ProcessCollisions(
    Ball *ballArray,
    int ballCount,
    float cellSize,
    int cellX,
    int cellY,
    int cellZ);

// #endif