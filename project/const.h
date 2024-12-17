#pragma once
#include "coor.hpp"
#include "wall.hpp"
#include "light.hpp"

const int BALL_SLICE = 50;
// const int BALL_NUM

const int WINDOW_WIDTH = 800,
          WINDOW_HEIGHT = 600, WINDOW_PLACE_X = 100, WINDOW_PLACE_Y = 100;
const float REFRESH_TIME = 0.02; // 刷新时间
const int BALL_COLS = 5;
const float LENGTH = 10, WIDTH = 10, HEIGHT = 20, MAX_RADIUS = 1; // 场景的X,Y,Z范围（-X,X),(0,H),(-Z,Z)
