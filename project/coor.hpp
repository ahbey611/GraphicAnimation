#pragma once

#include <math.h>

using namespace std;

#define PI 3.14159265358979323846

// 3维坐标
class Coor
{
public:
    float x, y, z;

    Coor()
    {
        x = 0;
        y = 0;
        z = 0;
    };
    Coor(float x, float y, float z) : x(x), y(y), z(z) {
                                      };
    Coor(const Coor &coor) : x(coor.x), y(coor.y), z(coor.z) {
                             };
    ~Coor() {};
    void setCoor(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    Coor operator+(const Coor &coor)
    {
        return Coor(this->x + coor.x, this->y + coor.y, this->z + coor.z);
    }
    Coor operator-(const Coor &coor)
    {
        return Coor(this->x - coor.x, this->y - coor.y, this->z - coor.z);
    }
    Coor operator*(const float &num)
    {
        return Coor(this->x * num, this->y * num, this->z * num);
    }
    Coor operator/(const float &num)
    {
        return Coor(this->x / num, this->y / num, this->z / num);
    }
    float operator*(const Coor &coor)
    {
        return this->x * coor.x + this->y * coor.y + this->z * coor.z;
    }
    float distance()
    {
        return sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
    }
    // 平方和
    float squareSum()
    {
        return this->x * this->x + this->y * this->y + this->z * this->z;
    }
};
