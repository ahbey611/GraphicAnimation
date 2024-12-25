#pragma once

#include <math.h>
#include "const.h"
#include "coor.hpp"

using namespace std;

struct Material
{
    GLfloat color[4];
    GLfloat ambient[4];
    GLfloat diffuse[4];
    GLfloat specular[4];
    GLfloat shininess;

    Material() : shininess(20.0f)
    {
        for (int i = 0; i < 4; i++)
        {
            color[i] = 1.0f;
            ambient[i] = 0.2f;
            diffuse[i] = 0.8f;
            specular[i] = 0.5f;
        }
    }

    Material(const GLfloat *c, const GLfloat *a,
             const GLfloat *d, const GLfloat *s,
             GLfloat sh) : shininess(sh)
    {
        for (int i = 0; i < 4; i++)
        {
            color[i] = c[i];
            ambient[i] = a[i];
            diffuse[i] = d[i];
            specular[i] = s[i];
        }
    }
};

class Wall
{
public:
    Vector3D vertices[4];
    Vector3D normal;
    Material material;

    Wall() {}

    void setGeometry(const Vector3D &v1, const Vector3D &v2,
                     const Vector3D &v3, const Vector3D &v4)
    {
        vertices[0] = v1;
        vertices[1] = v2;
        vertices[2] = v3;
        vertices[3] = v4;
        calculateNormal();
    }

    void setMaterial(const Material &mat)
    {
        material = mat;
    }

    float distanceToPoint(const Vector3D &point) const
    {
        return std::abs(normal.dot(point)) / normal.length();
    }

private:
    void calculateNormal()
    {
        Vector3D edge1 = vertices[1] - vertices[0];
        Vector3D edge2 = vertices[2] - vertices[0];
        normal = edge1.cross(edge2).normalized();

        // Ensure normal points inward
        if (normal.dot(vertices[0]) < 0)
        {
            normal = normal * -1.0f;
        }
    }
};