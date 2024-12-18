#include <GL/freeglut.h>
#include <iostream>
#include "collusion.cuh"
#include "const.h"
#include "coor.hpp"
#include "light.hpp"
#include "shader.hpp"
#include "ballset.hpp"
#include "ball.hpp"
#include "wall.hpp"
#include "particle_system.hpp"

class SimulationApp
{
private:
	static Wall boundaries[6];
	static Light sceneLight;
	static ParticleSystem particles;

	// 显示
	static void displayCallback()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		setupCamera();
		renderEnvironment();
		particles.update();
		particles.render();
		glutSwapBuffers();
	}

	// 设置相机的位置和角度
	static void setupCamera()
	{
		glLoadIdentity();
		const Vector3D eye(16.2244, 16.6, 20.4081);
		const Vector3D target(-16.1924, 0, -17.6596);
		gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, 0, 1.0, 0);
	}

	// 渲染环境
	static void renderEnvironment()
	{
		for (int i = 0; i < 3; i++)
		{
			const Wall &wall = boundaries[i];
			applyMaterial(wall.material);
			drawWall(wall);
		}
	}

	// 应用材质
	static void applyMaterial(const Material &mat)
	{
		glColor3fv(mat.color);
		glMaterialfv(GL_FRONT, GL_AMBIENT, mat.ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, mat.diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, mat.specular);
		glMaterialf(GL_FRONT, GL_SHININESS, mat.shininess);
	}

	// 绘制墙
	static void drawWall(const Wall &wall)
	{
		glBegin(GL_QUADS);
		for (int i = 0; i < 4; i++)
			glVertex3f(wall.vertices[i].x, wall.vertices[i].y, wall.vertices[i].z);
		glEnd();
	}

	// 重塑
	static void reshapeCallback(int w, int h)
	{
		glViewport(0, 0, w, h);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(75.0f, (float)w / h, 1.0f, 1000.0f);
		glMatrixMode(GL_MODELVIEW);
	}

	// 计时器
	static void timerCallback(int value)
	{
		glutPostRedisplay();
		glutTimerFunc(33, timerCallback, 1);
	}

	// 初始化墙
	static void initializeWalls()
	{
		// 定义墙的顶点
		Vector3D vertices[8] = {
			{-LENGTH, 0, -WIDTH}, {-LENGTH, 0, WIDTH}, {LENGTH, 0, -WIDTH}, {LENGTH, 0, WIDTH}, {-LENGTH, HEIGHT, -WIDTH}, {-LENGTH, HEIGHT, WIDTH}, {LENGTH, HEIGHT, -WIDTH}, {LENGTH, HEIGHT, WIDTH}};

		// 创建墙
		boundaries[0].setGeometry(vertices[0], vertices[1], vertices[3], vertices[2]); // Floor
		boundaries[1].setGeometry(vertices[0], vertices[1], vertices[5], vertices[4]); // Left wall
		boundaries[2].setGeometry(vertices[0], vertices[2], vertices[6], vertices[4]); // Back wall

		// 设置材质
		GLfloat floorColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		GLfloat floorAmbient[4] = {0.3f, 0.3f, 0.3f, 1.0f};
		GLfloat floorDiffuse[4] = {0.4f, 0.4f, 0.4f, 1.0f};
		GLfloat floorSpecular[4] = {0.2f, 0.2f, 0.2f, 1.0f};
		Material floorMat(floorColor, floorAmbient, floorDiffuse, floorSpecular, 20.0f);
		boundaries[0].setMaterial(floorMat);

		GLfloat wallColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		GLfloat wallAmbient[4] = {0.5f, 0.5f, 0.5f, 1.0f};
		GLfloat wallDiffuse[4] = {0.2f, 0.2f, 0.2f, 1.0f};
		GLfloat wallSpecular[4] = {0.2f, 0.2f, 0.2f, 1.0f};
		Material wallMat(wallColor, wallAmbient, wallDiffuse, wallSpecular, 20.0f);

		// 只绘制3面墙壁
		for (int i = 1; i < 3; i++)
		{
			boundaries[i].setMaterial(wallMat);
		}
	}

public:
	// 初始化
	static bool initialize(int argc, char *argv[])
	{
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
		glutInitWindowPosition(WINDOW_PLACE_X, WINDOW_PLACE_Y);
		glutCreateWindow("Particle Collision Simulation");

		if (!initializeGPU())
			return false;

		setupGL();
		initializeWalls();
		particles.initialize(LENGTH, HEIGHT, WIDTH);

		glutReshapeFunc(reshapeCallback);
		glutDisplayFunc(displayCallback);
		glutTimerFunc(33, timerCallback, 1);

		return true;
	}

	static void run()
	{
		glutMainLoop();
	}

private:
	// 初始化GPU
	static bool initializeGPU()
	{
		int deviceId = 0;
		cudaDeviceProp props;

		// 检查GPU是否成功
		if (cudaGetDeviceProperties(&props, deviceId) == cudaSuccess)
		{
			std::cout << "GPU Device: " << props.name << "\n"
					  << "Compute Units: " << props.multiProcessorCount << "\n"
					  << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
			return true;
		}
		return false;
	}

	// 设置GL
	static void setupGL()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glShadeModel(GL_SMOOTH);

		sceneLight.configure();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	}
};

Wall SimulationApp::boundaries[6];
Light SimulationApp::sceneLight;
ParticleSystem SimulationApp::particles(LENGTH, HEIGHT, WIDTH, BALL_COLS, MAX_RADIUS, REFRESH_TIME);

int main(int argc, char *argv[])
{
	if (!SimulationApp::initialize(argc, argv))
	{
		std::cerr << "Failed to initialize application\n";
		return -1;
	}

	SimulationApp::run();
	return 0;
}