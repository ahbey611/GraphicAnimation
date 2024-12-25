#include <GL/freeglut.h>
#include <iostream>
#include "collision.cuh"
#include "const.h"
#include "coor.hpp"
#include "light.hpp"
#include "ball.hpp"
#include "wall.hpp"
#include "particle_system.hpp"

class SimulationApp
{
private:
	static Wall boundaries[6];
	static Light sceneLight;
	static ParticleSystem *particles;

	// 显示
	static void displayCallback()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		setupCamera();
		renderEnvironment();
		particles->update();
		particles->render();
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
		boundaries[0].setGeometry(vertices[0], vertices[1], vertices[3], vertices[2]); // 地面
		boundaries[1].setGeometry(vertices[0], vertices[1], vertices[5], vertices[4]); // 左墙
		boundaries[2].setGeometry(vertices[0], vertices[2], vertices[6], vertices[4]); // 后墙

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
			boundaries[i].setMaterial(wallMat);
	}

public:
	// 添加配置结构体
	struct SimConfig
	{
		int ballCount = 150;	// 默认150个球
		float maxRadius = 1.0f; // 默认最大半径1.0
	};
	static SimConfig config;

	// 添加参数解析函数
	static bool parseArguments(int argc, char *argv[])
	{
		for (int i = 1; i < argc; i++)
		{
			std::string arg = argv[i];

			if (arg == "--ball" || arg == "-b")
			{
				if (i + 1 < argc)
				{
					config.ballCount = std::atoi(argv[++i]);
					if (config.ballCount <= 0 || config.ballCount > 200)
					{
						std::cerr << "the ball count must be between 1 and 200" << std::endl;
						return false;
					}
				}
				else
				{
					std::cerr << "--ball need a number" << std::endl;
					return false;
				}
			}
			else if (arg == "--help" || arg == "-h")
			{
				std::cout << "usage: " << argv[0] << " [options]" << std::endl
						  << "options:" << std::endl
						  << "  -b, --ball <number>        set the ball count (1-200)" << std::endl
						  << "  -h, --help                 show help information" << std::endl;
				return false;
			}
		}
		return true;
	}

	// 初始化
	static bool initialize(int argc, char *argv[])
	{
		if (!parseArguments(argc, argv))
			return false;

		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
		glutInitWindowPosition(WINDOW_PLACE_X, WINDOW_PLACE_Y);
		glutCreateWindow("Particle Collision Simulation");

		if (!initializeGPU())
			return false;

		// 在参数解析后创建粒子系统
		particles = new ParticleSystem(LENGTH, HEIGHT, WIDTH,
									   config.ballCount,
									   config.maxRadius,
									   REFRESH_TIME);

		setupGL();
		initializeWalls();
		particles->initialize();

		std::cout << "simulation scene initialized" << std::endl
				  << "ball count: " << config.ballCount << std::endl
				  << "max radius: " << config.maxRadius << std::endl;

		glutReshapeFunc(reshapeCallback);
		glutDisplayFunc(displayCallback);
		glutTimerFunc(33, timerCallback, 1);

		return true;
	}

	static void run()
	{
		glutMainLoop();
	}

	// 添加清理函数
	static void cleanup()
	{
		delete particles;
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
			std::cout << "GPU device: " << props.name << "\n"
					  << "compute unit: " << props.multiProcessorCount << "\n"
					  << "max thread block: " << props.maxThreadsPerBlock << "\n";
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
SimulationApp::SimConfig SimulationApp::config;
ParticleSystem *SimulationApp::particles = nullptr;

int main(int argc, char *argv[])
{
	if (!SimulationApp::initialize(argc, argv))
	{
		return -1;
	}

	SimulationApp::run();

	SimulationApp::cleanup();
	return 0;
}