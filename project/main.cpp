#include <GL/freeglut.h>
#include <iostream>
#include "collusion.cuh"
#include "const.h"
#include "coor.hpp"
#include "light.hpp"
#include "shader.hpp"
#include "camera.hpp"
#include "ballset.hpp"
#include "ball.hpp"
#include "wall.hpp"

using namespace std;

Wall walls[6];
Light light;
Camera camera(50.0f, 16.6f);
BallSet ballSet(LENGTH, HEIGHT, WIDTH, BALL_COLS, MAX_RADIUS, REFRESH_TIME);

// Print OpenGL and GPU information.
bool InitWindow()
{
	// Set the display mode to use double buffering and RGB.
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// Create a window with initial size and position.
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(WINDOW_PLACE_X, WINDOW_PLACE_Y);
	glutCreateWindow("GPU小球碰撞检测大作业");

	int deviceID = 0;
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, deviceID) == cudaSuccess)
	{
		std::cout << "Use GPU device " << deviceID << ": " << devProps.name << std::endl;						// NVIDIA GeForce GTX 1660 Ti
		std::cout << "SM的数量：" << devProps.multiProcessorCount << std::endl;									// 24
		std::cout << "每个线程块的共享内存大小：" << devProps.sharedMemPerBlock / 1024.0 << " KB" << std::endl; // 48 KB
		std::cout << "每个线程块的最大线程数：" << devProps.maxThreadsPerBlock << std::endl;					// 1024
		std::cout << "每个EM的最大线程数：" << devProps.maxThreadsPerMultiProcessor << std::endl;				// 1024
		std::cout << "每个EM的最大线程束数：" << devProps.maxThreadsPerMultiProcessor / 32 << std::endl;		// 32
		return true;
	}
	return false;
}

// 初始化3面墙，一个正方体只显示3面墙
void InitWall()
{
	// 8 个顶点
	Coor floor1(-LENGTH, 0, -WIDTH);
	Coor floor2(-LENGTH, 0, WIDTH);
	Coor floor3(LENGTH, 0, -WIDTH);
	Coor floor4(LENGTH, 0, WIDTH);
	Coor ceiling1(-LENGTH, HEIGHT, -WIDTH);
	Coor ceiling2(-LENGTH, HEIGHT, WIDTH);
	Coor ceiling3(LENGTH, HEIGHT, -WIDTH);
	Coor ceiling4(LENGTH, HEIGHT, WIDTH);

	// 6面墙壁
	walls[0].setVertexes(floor1, floor2, floor4, floor3);
	walls[1].setVertexes(floor1, floor2, ceiling2, ceiling1);
	walls[3].setVertexes(floor3, floor4, ceiling4, ceiling3);
	walls[2].setVertexes(floor1, floor3, ceiling3, ceiling1);
	walls[4].setVertexes(floor2, floor4, ceiling4, ceiling2);
	walls[5].setVertexes(ceiling1, ceiling2, ceiling4, ceiling3);

	// 设置墙的shader
	GLfloat color[4] = {1.0, 1.0, 1.0, 1.0};
	GLfloat ambient[4] = {0.3, 0.3, 0.3, 1.0};
	GLfloat diffuse[4] = {0.4, 0.4, 0.4, 1.0};
	GLfloat specular[4] = {0.2, 0.2, 0.2, 1.0};
	GLfloat shininess = 20.0;
	Shader shader_floor;
	shader_floor.setShader(color, ambient, diffuse, specular, shininess);
	walls[0].shader = shader_floor;

	GLfloat color_border[4] = {1.0, 1.0, 1.0, 1};
	GLfloat ambient_border[4] = {0.5, 0.5, 0.5, 1};
	GLfloat diffuse_border[4] = {0.2, 0.2, 0.2, 1};
	GLfloat specular_border[4] = {0.2, 0.2, 0.2, 1};
	GLfloat shininess_border = 20;
	Shader shader_border;
	shader_border.setShader(color_border, ambient_border, diffuse_border, specular_border, shininess_border);
	for (int i = 1; i < 5; i++)
	{
		walls[i].shader = shader_border;
	}
}

void RenderWall()
{
	// 只渲染3面墙
	for (int i = 0; i < 3; i++)
	{
		glColor3f(walls[i].shader.color[0], walls[i].shader.color[1], walls[i].shader.color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, walls[i].shader.ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, walls[i].shader.diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, walls[i].shader.specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, walls[i].shader.shininess);

		glBegin(GL_POLYGON);
		glVertex3f(walls[i].vertexes[0].x, walls[i].vertexes[0].y, walls[i].vertexes[0].z);
		glVertex3f(walls[i].vertexes[1].x, walls[i].vertexes[1].y, walls[i].vertexes[1].z);
		glVertex3f(walls[i].vertexes[2].x, walls[i].vertexes[2].y, walls[i].vertexes[2].z);
		glVertex3f(walls[i].vertexes[3].x, walls[i].vertexes[3].y, walls[i].vertexes[3].z);
		glEnd();
		glFlush();
	}
}

void InitLight()
{
	glShadeModel(GL_SMOOTH);
	glClearColor(light.color[0], light.color[1], light.color[2], light.color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light.ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light.diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light.specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light.position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glEnable(GL_DEPTH_TEST);
}

void InitCamera()
{
	glLoadIdentity();
	Coor position = camera.position;
	Coor lookAt = camera.lookAt;
	gluLookAt(position.x, position.y, position.z, lookAt.x, lookAt.y, lookAt.z, 0, 1.0, 0);
}

// This function renders the scene using OpenGL.
void RenderScene()
{
	// Clear the frame buffer by setting it to the clear color.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	InitCamera();
	RenderWall();
	ballSet.updateBalls();
	ballSet.renderBalls();
	glutSwapBuffers();
}

void OnMouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		camera.MouseDown(x, y);
	}
}

void OnMouseMove(int x, int y)
{
	camera.MouseMove(x, y);
}

// keyboard events WASD
void OnKeyClick(unsigned char key, int x, int y)
{
	int type = -1;
	switch (key)
	{
	case 'w':
		type = 0;
		break;
	case 'a':
		type = 1;
		break;
	case 's':
		type = 2;
		break;
	case 'd':
		type = 3;
		break;
	}
	camera.KeyboardMove(type);
}

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0f, (float)w / (float)h, 1.0f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
}

void OnTimer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(33, OnTimer, 1);
}

int main(int argc, char *argv[])
{
	// Initialize GLUT.
	glutInit(&argc, argv);
	bool status = InitWindow();
	if (!status)
	{
		cout << "Failed to initialize window" << endl;
		return -1;
	}

	InitLight();
	InitWall();
	ballSet.initBalls();

	// Set the reshape function to handle window resizing.
	glutReshapeFunc(reshape);

	// Set the display function to render the scene.
	glutDisplayFunc(RenderScene);

	// Set the timer function to update the scene.
	glutTimerFunc(33, OnTimer, 1);

	// Set the mouse click function
	glutMouseFunc(OnMouseClick);
	// Set the mouse move function
	glutMotionFunc(OnMouseMove);
	// Set the keyboard event function
	glutKeyboardFunc(OnKeyClick);

	// Set the clear color to white.
	// glClearColor(1.0, 1.0, 1.0, 1.0);
	// glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// Transfer control over to GLUT.
	glutMainLoop();

	// This code is never reached.
	return 0;
}