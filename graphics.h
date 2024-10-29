#pragma once
#ifndef GRAPHICS_HEADER
#define GRAPHICS_HEADER

#include "utils.h"
#include "ocean.h"

namespace graphics
{
	void renderString(int x, int y, void* font, unsigned char* string, float3 rgb);
	void refreshDisplay(int value);
	void display();
	void keyboard(unsigned char key, int /*x*/, int /*y*/);
	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
	bool initGL(int* argc, char** argv, char *title);

	namespace cpu
	{
		void cleanup();
		void display();
		bool initGL(int* argc, char** argv);
	}

	namespace gpu
	{
		void cleanup();
		void display();
		bool initGL(int* argc, char** argv);
	}
}

#endif