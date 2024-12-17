#pragma once

#include <GL/freeglut.h>

using namespace std;

class Shader
{
public:
    GLfloat color[4];
    GLfloat ambient[4];           // 环境光
    GLfloat diffuse[4];           // 漫反射光
    GLfloat specular[4];          // 镜面反射光
    GLfloat shininess[1] = {0.0}; // 光泽度

    Shader()
    {
        for (int i = 0; i < 4; i++)
        {
            color[i] = 0.0;
            ambient[i] = 0.0;
            diffuse[i] = 0.0;
            specular[i] = 0.0;
        }
    }

    Shader(GLfloat color[4], GLfloat ambient[4], GLfloat diffuse[4], GLfloat specular[4], GLfloat shininess)
    {
        for (int i = 0; i < 4; i++)
        {
            this->color[i] = color[i];
            this->ambient[i] = ambient[i];
            this->diffuse[i] = diffuse[i];
            this->specular[i] = specular[i];
        }
        this->shininess[0] = shininess;
    }

    Shader(const Shader &shader)
    {
        for (int i = 0; i < 4; i++)
        {
            this->color[i] = shader.color[i];
            this->ambient[i] = shader.ambient[i];
            this->diffuse[i] = shader.diffuse[i];
            this->specular[i] = shader.specular[i];
        }
        this->shininess[0] = shader.shininess[0];
    }

    void setColor(GLfloat color[4])
    {
        for (int i = 0; i < 4; i++)
        {
            this->color[i] = color[i];
        }
    }

    void setAmbient(GLfloat ambient[4])
    {
        for (int i = 0; i < 4; i++)
        {
            this->ambient[i] = ambient[i];
        }
    }

    void setDiffuse(GLfloat diffuse[4])
    {
        for (int i = 0; i < 4; i++)
        {
            this->diffuse[i] = diffuse[i];
        }
    }

    void setSpecular(GLfloat specular[4])
    {
        for (int i = 0; i < 4; i++)
        {
            this->specular[i] = specular[i];
        }
    }

    void setShininess(GLfloat shininess)
    {
        this->shininess[0] = shininess;
    }

    void setShader(GLfloat color[4], GLfloat ambient[4], GLfloat diffuse[4], GLfloat specular[4], GLfloat shininess)
    {
        for (int i = 0; i < 4; i++)
        {
            this->color[i] = color[i];
            this->ambient[i] = ambient[i];
            this->diffuse[i] = diffuse[i];
            this->specular[i] = specular[i];
        }
        this->shininess[0] = shininess;
    }
};