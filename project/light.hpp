#pragma once

#include <GL/freeglut.h>

class Light
{
public:
    GLfloat position[4];
    GLfloat color[4];
    GLfloat ambient[4];
    GLfloat diffuse[4];
    GLfloat specular[4];

    Light()
    {
        for (int i = 0; i < 4; i++)
        {
            position[i] = 10.0;
            color[i] = 0.1;
            ambient[i] = 1.0;
            diffuse[i] = 1.0;
            specular[i] = 1.0;
        }
        color[3] = 1.0;
        position[3] = 1.0;
    }
};