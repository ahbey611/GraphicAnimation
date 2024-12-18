#pragma once

#include "ball.hpp"
#include "const.h"
#include <vector>
#include <math.h>
#include "collusion.cuh"
#include "coor.hpp"
#include <iostream>
using namespace std;

class BallSet
{
public:
    Ball *balls;                  // 所有小球
    float rangeX, rangeY, rangeZ; // 长度、高度、宽度
    int col;                      // 列数
    int N;                        // 球的数量
    float maxRadius;              // 所有球中的最大半径
    float refreshInterval;        // 刷新时间
    float cellSize;               // 每个cell的大小
    float cellX, cellY, cellZ;    // 每个cell的x, y, z方向的cell数量

    BallSet(
        float rangeX, float rangeY, float rangeZ,
        int cols, float maxRadius,
        float refreshInterval)
    {
        this->rangeX = rangeX;
        this->rangeY = rangeY;
        this->rangeZ = rangeZ;
        this->col = cols;
        this->maxRadius = maxRadius;
        this->refreshInterval = refreshInterval;
        this->cellSize = maxRadius * 1.5;
        this->cellX = ceil(rangeX * 2 / cellSize);
        this->cellY = ceil(rangeY / cellSize);
        this->cellZ = ceil(rangeZ * 2 / cellSize);
        this->N = cols * cols * cols;
        this->balls = new Ball[N];

        cout << "cellSize: " << cellSize << endl;
        cout << "cellX: " << cellX << endl;
        cout << "cellY: " << cellY << endl;
        cout << "cellZ: " << cellZ << endl;
    }

    vector<GLfloat> GenerateRandomColor()
    {
        vector<GLfloat> color;
        GLfloat r = rand() % 256 / 255.0;
        GLfloat g = rand() % 256 / 255.0;
        GLfloat b = rand() % 256 / 255.0;
        color.push_back(r);
        color.push_back(g);
        color.push_back(b);
        return color;
    }

    void initBalls()
    {

        // 计算每个球之间的距离
        float diffX = (2 * (rangeX - maxRadius)) / (col - 1);
        float diffY = (rangeY - 2 * maxRadius) / (col - 1);
        float diffZ = (2 * (rangeZ - maxRadius)) / (col - 1);

        for (int i = 0; i < col; i++)
        {
            for (int j = 0; j < col; j++)
            {
                for (int k = 0; k < col; k++)
                {
                    int index = i * col * col + j * col + k;
                    // 让全部小球整齐均匀排列在场景中
                    float posX = i * diffX + maxRadius - rangeX;
                    float posY = k * diffY + maxRadius;
                    float posZ = j * diffZ + maxRadius - rangeZ;
                    Vector3D position(posX, posY, posZ);

                    float speedX = ((rand() % 201) / 100.0f - 1.0f) * 10;
                    float speedY = ((rand() % 201) / 100.0f - 1.0f) * 10;
                    float speedZ = ((rand() % 201) / 100.0f - 1.0f) * 10;
                    Vector3D speed(speedX, speedY, speedZ);

                    float radius = maxRadius * (rand() % 101 / 100.0f);
                    // 避免球半径过小
                    if (radius < maxRadius * 0.5)
                    {
                        radius += maxRadius * 0.7;
                        if (radius > maxRadius)
                            radius = maxRadius;
                    }

                    vector<GLfloat> randomColor = GenerateRandomColor();
                    // cout << randomColor[0] << " " << randomColor[1] << " " << randomColor[2] << endl;
                    GLfloat color[4] = {randomColor[0], randomColor[1], randomColor[2], 1.0};
                    GLfloat ambient[4] = {color[0] * 0.2f, color[1] * 0.2f, color[2] * 0.2f, 1.0};
                    GLfloat diffuse[4] = {color[0], color[1], color[2], 1.0};
                    GLfloat specular[4] = {0.5, 0.5, 0.5, 1.0};
                    GLfloat shininess = 30;
                    Shader shader(color, ambient, diffuse, specular, shininess);
                    balls[index].Init(position, speed, radius, shader);
                }
            }
        }
    }

    void renderBalls()
    {
        for (int i = 0; i < N; i++)
        {
            balls[i].Render();
        }
    }

    void updateBalls()
    {

        // CollisionDetection(balls, refreshInterval, rangeX, rangeZ, rangeY, cellSize, cellX, cellY, cellZ, N);
        ProcessCollisions(balls, N, cellSize, cellX, cellY, cellZ);
        // testhaha();
    }
};