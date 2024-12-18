#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "ballset.hpp"
#include "ball.hpp"

#define HOME_CELL 0x00
#define PHANTOM_CELL 0x01
#define HOME_OBJECT 0x01
#define PHANTOM_OBJECT 0x00
#define EMPTY_CELL UINT32_MAX

#define RADIX_BITS_STRIDE 8
#define RADIX_LENGTH 8

#define GROUPS_PER_BLOCK 12
// #define THREADS_PER_GROUP 16
#define PADDED_BLOCKS 16
#define PADDED_GROUPS 256

// int NUM_BLOCKS = 128;
// int THREADS_PER_BLOCK = 512;
#define BLOCKS_NUM 128
#define THREADS_PER_BLOCK 512
// #define NUM_BLOCKS 12

// int testNum = 1;

void CollisionDetection(Ball *balls, float RefreshInterval, float XRange, float ZRange, float Height,
                        float GridSize, int GridX, int GridY, int GridZ, int N);