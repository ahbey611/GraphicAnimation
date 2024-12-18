#pragma once

#include <GL/freeglut.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "coor.hpp"
#include "shader.hpp"
#include "const.h"

using namespace std;

class Ball
{
public:
    Point3D position;
    Point3D speed;
    GLfloat radius;
    GLfloat weight;
    Shader shader;

    Ball() {}
    ~Ball() {}

    void Init(Point3D position, Point3D speed, GLfloat radius, Shader shader)
    {
        this->position = position;
        this->speed = speed;
        this->radius = radius;
        this->weight = radius * radius * radius; // *4 / 3 * PI;
        this->shader = shader;
        for (int i = 0; i < 4; i++)
        {
            this->shader.color[i] = shader.color[i];
        }
    }

    void Render()
    {
        glColor3f(shader.color[0], shader.color[1], shader.color[2]);
        glMaterialfv(GL_FRONT, GL_AMBIENT, shader.ambient);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, shader.diffuse);
        glMaterialfv(GL_FRONT, GL_SPECULAR, shader.specular);
        glMaterialfv(GL_FRONT, GL_SHININESS, shader.shininess);

        glPushMatrix();
        glTranslatef(position.x, position.y, position.z);
        glutSolidSphere(radius, BALL_SLICE, BALL_SLICE);
        glPopMatrix();
    }

    GLfloat *GenerateRandomColor()
    {
        GLfloat r = rand() % 256 / 255.0;
        GLfloat g = rand() % 256 / 255.0;
        GLfloat b = rand() % 256 / 255.0;
        GLfloat color[4] = {r, g, b, 1.0};
        return color;
    }
};