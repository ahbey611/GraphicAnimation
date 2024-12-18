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
    float spaceX, spaceY, spaceZ;
    int particleCount;          // 总粒子数量
    float maxParticleRadius;    // 最大粒子半径
    float updateInterval;       // 更新时间间隔
    float cellSize;             // 空间划分单元格大小
    int cellsX, cellsY, cellsZ; // 每个维度上的单元格数量

    // 检查新位置是否与现有粒子重叠
    bool checkOverlap(const Point3D &pos, float radius, int currentIndex) const
    {
        for (int i = 0; i < currentIndex; i++)
        {
            const Point3D &otherPos = particles[i].position;
            float minDist = radius + particles[i].radius;

            // 计算两点之间的距离
            Vector3D diff = pos - otherPos;
            if (diff.lengthSquared() < minDist * minDist)
            {
                return true; // 发生重叠
            }
        }
        return false;
    }

    // 生成随机位置
    Point3D generateRandomPosition(float radius) const
    {
        float x = ((rand() % 2000) / 1000.0f - 1.0f) * (spaceX - radius);
        float y = radius + (rand() % 1000) / 1000.0f * (spaceY - 2 * radius);
        float z = ((rand() % 2000) / 1000.0f - 1.0f) * (spaceZ - radius);
        return Point3D(x, y, z);
    }

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
                   int count, float maxRadius, float updateInterval)
        : spaceX(spaceX), spaceY(spaceY), spaceZ(spaceZ),
          particleCount(std::min(count, 200)), // 限制最大数量为200
          maxParticleRadius(maxRadius),
          updateInterval(updateInterval)
    {
        cellSize = maxRadius * 1.5f;
        cellsX = ceil(spaceX * 2 / cellSize);
        cellsY = ceil(spaceY / cellSize);
        cellsZ = ceil(spaceZ * 2 / cellSize);
        particles = new Ball[particleCount];
        cout << "cellSize: " << cellSize << endl;
        cout << "cellX: " << cellsX << endl;
        cout << "cellY: " << cellsY << endl;
        cout << "cellZ: " << cellsZ << endl;
    }

    ~ParticleSystem()
    {
        delete[] particles;
    }

    void initialize()
    {
        const int MAX_ATTEMPTS = 100; // 每个粒子的最大尝试次数

        for (int i = 0; i < particleCount; i++)
        {
            // 随机半径
            float radius = maxParticleRadius * (0.5f + (rand() % 51) / 100.0f); // 50%~100%的最大半径

            // 尝试找到一个不重叠的位置
            Point3D position;
            bool validPosition = false;
            int attempts = 0;

            while (!validPosition && attempts < MAX_ATTEMPTS)
            {
                position = generateRandomPosition(radius);
                validPosition = !checkOverlap(position, radius, i);
                attempts++;
            }

            if (!validPosition)
            {
                // 如果无法找到合适位置，减小半径继续尝试
                radius *= 0.8f;
                attempts = 0;
                while (!validPosition && attempts < MAX_ATTEMPTS)
                {
                    position = generateRandomPosition(radius);
                    validPosition = !checkOverlap(position, radius, i);
                    attempts++;
                }
            }

            // 随机初始速度
            float velX = ((rand() % 201) / 100.0f - 1.0f) * 10;
            float velY = ((rand() % 201) / 100.0f - 1.0f) * 10;
            float velZ = ((rand() % 201) / 100.0f - 1.0f) * 10;
            Point3D velocity(velX, velY, velZ);

            // 随机颜色和材质属性
            std::vector<GLfloat> randomColor = generateRandomColor();
            GLfloat color[4] = {randomColor[0], randomColor[1], randomColor[2], 1.0f};
            GLfloat ambient[4] = {color[0] * 0.2f, color[1] * 0.2f, color[2] * 0.2f, 1.0f};
            GLfloat diffuse[4] = {color[0], color[1], color[2], 1.0f};
            GLfloat specular[4] = {0.5f, 0.5f, 0.5f, 1.0f};
            GLfloat shininess = 30.0f;

            Shader shader(color, ambient, diffuse, specular, shininess);
            particles[i].Init(position, velocity, radius, shader);
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