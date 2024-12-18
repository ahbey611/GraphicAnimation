#pragma once

#include <GL/freeglut.h>

// 灯光
class Light
{
private:
    struct LightProperties
    {
        GLfloat position[4]; // 位置
        GLfloat color[4];    // 颜色
        GLfloat ambient[4];  // 环境光
        GLfloat diffuse[4];  // 漫反射
        GLfloat specular[4]; // 镜面反射
    } props;

public:
    Light()
    {
        setDefaults();
    }

    void configure() const
    {
        glLightfv(GL_LIGHT0, GL_POSITION, props.position);
        glLightfv(GL_LIGHT0, GL_AMBIENT, props.ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, props.diffuse);
        glLightfv(GL_LIGHT0, GL_SPECULAR, props.specular);
    }

    void setPosition(float x, float y, float z, float w = 1.0f)
    {
        props.position[0] = x;
        props.position[1] = y;
        props.position[2] = z;
        props.position[3] = w;
    }

    void setAmbient(float r, float g, float b, float a = 1.0f)
    {
        props.ambient[0] = r;
        props.ambient[1] = g;
        props.ambient[2] = b;
        props.ambient[3] = a;
    }

private:
    void setDefaults()
    {
        // 设置灯光位置
        setPosition(10.0f, 10.0f, 10.0f);

        // 设置中等环境光
        setAmbient(0.3f, 0.3f, 0.3f);

        // 设置明亮漫反射和镜面反射
        for (int i = 0; i < 4; i++)
        {
            props.diffuse[i] = 1.0f;
            props.specular[i] = 1.0f;
            props.color[i] = 0.1f;
        }
        props.color[3] = 1.0f;
    }
};