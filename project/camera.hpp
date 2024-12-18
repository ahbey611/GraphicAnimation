// #pragma once

// #include <math.h>
// #include "coor.hpp"
// #include <iostream>

// using namespace std;

// class Camera
// {
// public:
//     Point3D position;
//     Point3D lookAt;
//     float r_x_z;
//     float a_x_z;
//     float height;
//     int MouseX;
//     int MouseY;
//     const float v_xOz = 0.02;
//     const float v_y = 0.05;
//     const float v_key = 0.5;

//     Camera(float r, float height)
//     {
//         this->r_x_z = r;
//         // this->a_x_z = PI / 4;
//         this->a_x_z = 0.865399;
//         this->height = height;

//         lookAt = Point3D(-16.1924, 0, -17.6596);
//         resetPos();
//         cout << "position: " << position.x << ", " << position.y << ", " << position.z << endl;
//     }

//     void resetPos()
//     {
//         float x = r_x_z * cos(a_x_z) + lookAt.x;
//         float y = height + lookAt.y;
//         float z = r_x_z * sin(a_x_z) + lookAt.z;
//         position.setCoor(x, y, z);
//     }

//     // mouse click
//     void MouseDown(int x, int y)
//     {
//         MouseX = x;
//         MouseY = y;
//     }

//     // mouse move
//     void MouseMove(int x, int y)
//     {
//         int dx = x - MouseX;
//         int dy = y - MouseY;
//         a_x_z = a_x_z + dx * v_xOz;
//         while (a_x_z < 0)
//             a_x_z += 2.0 * PI;
//         while (a_x_z >= 2.0 * PI)
//             a_x_z -= 2.0 * PI;
//         height += dy * v_y;
//         resetPos();
//         MouseX = x;
//         MouseY = y;
//         // cout << "a_x_z: " << a_x_z << endl;
//         // cout << "height: " << height << endl;
//     }

//     // wasd key move
//     void KeyboardMove(int type)
//     {
//         float change_x = 0;
//         float change_z = 0;

//         if (type == 0)
//         {
//             change_x = -cos(a_x_z) * v_key;
//             change_z = -sin(a_x_z) * v_key;
//         }
//         else if (type == 1)
//         {
//             change_x = -sin(a_x_z) * v_key;
//             change_z = cos(a_x_z) * v_key;
//         }
//         else if (type == 2)
//         {
//             change_x = cos(a_x_z) * v_key;
//             change_z = sin(a_x_z) * v_key;
//         }
//         else if (type == 3)
//         {
//             change_x = sin(a_x_z) * v_key;
//             change_z = -cos(a_x_z) * v_key;
//         }
//         lookAt.x += change_x;
//         lookAt.z += change_z;
//         // cout << "lookAt: " << lookAt.x << ", " << lookAt.y << ", " << lookAt.z << endl;
//         resetPos();
//         // cout << "position: " << position.x << ", " << position.y << ", " << position.z << endl;
//     }
// };