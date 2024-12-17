#pragma once

#include <math.h>
#include "const.h"
#include "coor.hpp"
#include "shader.hpp"

using namespace std;

class Wall
{
public:
    Coor vertexes[4];
    Coor normal;
    Shader shader;

    Wall()
    {
        for (int i = 0; i < 4; i++)
        {
            vertexes[i] = Coor();
        }
        // normal = Coor();
        // shader = Shader();
    }

    void setVertexes(Coor v1, Coor v2, Coor v3, Coor v4)
    {
        vertexes[0] = v1;
        vertexes[1] = v2;
        vertexes[2] = v3;
        vertexes[3] = v4;
        getNormal();
    }

    void getNormal()
    {

        Coor v1 = vertexes[0];
        Coor v2 = vertexes[1];
        Coor v3 = vertexes[2];

        float na = (v2.y - v1.y) * (v3.z - v1.z) - (v2.z - v1.z) * (v3.y - v1.y);
        float nb = (v2.z - v1.z) * (v3.x - v1.x) - (v2.x - v1.x) * (v3.z - v1.z);
        float nc = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);
        float norm = sqrt(na * na + nb * nb + nc * nc);
        na /= norm;
        nb /= norm;
        nc /= norm;
        if (na * v1.x + nb * v1.y + nc * v1.z < 0)
        {
            na = -na;
            nb = -nb;
            nc = -nc;
        }
        normal.setCoor(na, nb, nc);
    }

    float getDistance(Coor point)
    {
        getNormal();
        float distance = abs(normal.x * point.x + normal.y * point.y + normal.z * point.z);
        float norm = sqrt(normal.squareSum());
        distance /= norm;
        return distance;
    }
};