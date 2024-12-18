#pragma once

#include "ball.hpp"
#include "const.h"
#include <vector>
#include "collusion.cuh"
#include "coor.hpp"
#include <iostream>

class ParticleSystem
{
private:
    Ball *particles;
    float spaceX, spaceY, spaceZ; // 场景的x, y, z方向的大小
    int gridSize;                 // 每个维度上的粒子数量
    int particleCount;            // 总粒子数量
    float maxParticleRadius;      // 最大粒子半径
    float updateInterval;         // 更新时间间隔
    float cellSize;               // 空间划分单元格大小
    int cellsX, cellsY, cellsZ;   // 每个维度上的单元格数量

    std::vector<GLfloat> generateRandomColor()
    {
        std::vector<GLfloat> color;
        GLfloat r = rand() % 256 / 255.0f;
        GLfloat g = rand() % 256 / 255.0f;
        GLfloat b = rand() % 256 / 255.0f;
        color.push_back(r);
        color.push_back(g);
        color.push_back(b);
        return color;
    }

public:
    ParticleSystem(float spaceX, float spaceY, float spaceZ,
                   int gridSize, float maxRadius, float updateInterval)
        : spaceX(spaceX), spaceY(spaceY), spaceZ(spaceZ),
          gridSize(gridSize), maxParticleRadius(maxRadius),
          updateInterval(updateInterval)
    {
        particleCount = gridSize * gridSize * gridSize;
        cellSize = maxRadius * 1.5f;
        cellsX = ceil(spaceX * 2 / cellSize);
        cellsY = ceil(spaceY / cellSize);
        cellsZ = ceil(spaceZ * 2 / cellSize);
        particles = new Ball[particleCount];
    }

    ~ParticleSystem()
    {
        delete[] particles;
    }

    void initialize(float length, float height, float width)
    {
        // 计算粒子之间的间距
        float spacingX = (2 * (spaceX - maxParticleRadius)) / (gridSize - 1);
        float spacingY = (spaceY - 2 * maxParticleRadius) / (gridSize - 1);
        float spacingZ = (2 * (spaceZ - maxParticleRadius)) / (gridSize - 1);

        // 在网格中初始化粒子
        for (int i = 0; i < gridSize; i++)
        {
            for (int j = 0; j < gridSize; j++)
            {
                for (int k = 0; k < gridSize; k++)
                {
                    int index = i * gridSize * gridSize + j * gridSize + k;

                    // Position
                    float posX = i * spacingX + maxParticleRadius - spaceX;
                    float posY = k * spacingY + maxParticleRadius;
                    float posZ = j * spacingZ + maxParticleRadius - spaceZ;
                    Point3D position(posX, posY, posZ);

                    // 随机初始速度
                    float velX = ((rand() % 201) / 100.0f - 1.0f) * 10;
                    float velY = ((rand() % 201) / 100.0f - 1.0f) * 10;
                    float velZ = ((rand() % 201) / 100.0f - 1.0f) * 10;
                    Point3D velocity(velX, velY, velZ);

                    // 随机半径
                    float radius = maxParticleRadius * (rand() % 101 / 100.0f);
                    if (radius < maxParticleRadius * 0.5f)
                    {
                        radius += maxParticleRadius * 0.7f;
                        radius = std::min(radius, maxParticleRadius);
                    }

                    // 随机颜色和材质属性
                    std::vector<GLfloat> randomColor = generateRandomColor();
                    GLfloat color[4] = {randomColor[0], randomColor[1], randomColor[2], 1.0f};
                    GLfloat ambient[4] = {color[0] * 0.2f, color[1] * 0.2f, color[2] * 0.2f, 1.0f};
                    GLfloat diffuse[4] = {color[0], color[1], color[2], 1.0f};
                    GLfloat specular[4] = {0.5f, 0.5f, 0.5f, 1.0f};
                    GLfloat shininess = 30.0f;

                    Shader shader(color, ambient, diffuse, specular, shininess);
                    particles[index].Init(position, velocity, radius, shader);
                }
            }
        }
    }

    // 更新粒子
    void update()
    {
        ProcessCollisions(particles, particleCount, cellSize, cellsX, cellsY, cellsZ);
    }

    // 渲染粒子
    void render()
    {
        for (int i = 0; i < particleCount; i++)
            particles[i].Render();
    }
};