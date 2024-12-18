#pragma once
#include <cmath>

// 3D向量
class Vector3D
{
public:
    float x, y, z;

    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vector3D operator+(const Vector3D &v) const
    {
        return Vector3D(x + v.x, y + v.y, z + v.z);
    }

    Vector3D operator-(const Vector3D &v) const
    {
        return Vector3D(x - v.x, y - v.y, z - v.z);
    }

    Vector3D operator*(float scalar) const
    {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    float dot(const Vector3D &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    Vector3D cross(const Vector3D &v) const
    {
        return Vector3D(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x);
    }

    float length() const
    {
        return std::sqrt(lengthSquared());
    }

    float lengthSquared() const
    {
        return x * x + y * y + z * z;
    }

    Vector3D normalized() const
    {
        float len = length();
        if (len > 0)
        {
            return *this * (1.0f / len);
        }
        return *this;
    }
};

using Point3D = Vector3D;
