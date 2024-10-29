#include "graphics.h"

float g_time = 0.0;

// Text buffer
char text_buff[100];

// FPS timers
static unsigned int fps_start = 0;
static unsigned int fps_frames = 0;
double ms;

// camera controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_x = 0.0, translate_y = 0.0, translate_z = -3.0;

void graphics::renderString(int x, int y, void* font, unsigned char* string, float3 rgb) {
    glColor3f(rgb.x, rgb.y, rgb.z);
    glWindowPos2d(x, y);
    glutBitmapString(font, string);
}

void graphics::refreshDisplay(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, refreshDisplay, 0);
    }
}

void graphics::display() {
    // calculate FPS
    fps_frames++;
    int delta_t = glutGet(GLUT_ELAPSED_TIME) - fps_start;
    if (delta_t > 1000) {
        ms = delta_t / (float)fps_frames;
        fps_frames = 0;
        fps_start = glutGet(GLUT_ELAPSED_TIME);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glTranslatef(translate_x, translate_y, translate_z);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo::vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glColor3f(1.0, 0.0, 0.0);

    glDrawArrays(GL_POINTS, 0, N * N);
    glDisableClientState(GL_VERTEX_ARRAY);

    // Print specs
    sprintf(text_buff, "N: %d | L: %2.2f | V: %2.2f", N, LENGTH, V);
    renderString(20, WINDOW_HEIGHT - 50, GLUT_BITMAP_9_BY_15, (unsigned char*)text_buff, make_float3(1.0f, 1.0f, 0.0f));

    // Print timers
    sprintf(text_buff, "     FPS: %3.3f\nms/frame: %3.3f\n VBO(ms): %3.3f", 1000 / ms, ms, vbo_time);
    renderString(20, WINDOW_HEIGHT - 80, GLUT_BITMAP_9_BY_15, (unsigned char*)text_buff, make_float3(1.0f, 1.0f, 1.0f));

    glutSwapBuffers();

    // Update time
    g_time += TIMESTEP / (float)1000;
}

void graphics::keyboard(unsigned char key, int /*x*/, int /*y*/) {
    float speed = 0.5f;
    float sprint = 10.0f;

    switch (key) {
    case 'q':
        glutDestroyWindow(glutGetWindow());
        exit(0);
    case 'A': speed *= sprint;
    case 'a':
        translate_x -= speed; break;
    case 'D': speed *= sprint;
    case 'd':
        translate_x += speed; break;
    case 'W': speed *= sprint;
    case 'w':
        translate_z -= speed; break;
    case 'S': speed *= sprint;
    case 's':
        translate_z += sprint; break;
    case 'R': speed *= sprint;
    case 'r':
        translate_y -= speed; break;
    case 'F': speed *= sprint;
    case 'f':
        translate_y += speed; break;
    default:
        printf("Unkown key: %d\n", (int)key);
    }
}

void graphics::mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN)
        mouse_buttons |= 1 << button;
    else if (state == GLUT_UP)
        mouse_buttons = 0;

    mouse_old_x = x;
    mouse_old_y = y;
}

void graphics::motion(int x, int y) {
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
        translate_z += dy * 0.01f;

    mouse_old_x = x;
    mouse_old_y = y;
}

bool graphics::initGL(int* argc, char** argv, char *title) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow(title);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, refreshDisplay, 0);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0)) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 0.1, 1000.0);

    return true;
}


void graphics::cpu::cleanup() {
    if (vbo::vbo)
        vbo::cpu::deleteVBO(&vbo::vbo);
}

void graphics::cpu::display() {
    // Calculate vbo
    switch (VBO_MODE) {
    case(VBO_DIRECT_ONE_STEP): ocean::cpu::direct::calculate_vbo(&vbo::vbo, g_time); break;
    case(VBO_FFT): ocean::cpu::fft::calculate_vbo(&vbo::vbo, g_time); break;
    }
    
    graphics::display();
}

bool graphics::cpu::initGL(int* argc, char** argv) {
    return graphics::initGL(argc, argv, "CPU: Wave Simulation");
}


void graphics::gpu::cleanup() {
        if (vbo::vbo)
            vbo::gpu::deleteVBO(&vbo::vbo, vbo::cuda_vbo_resource);
}

void graphics::gpu::display() {
    // run CUDA kernel to generate vertex positions
    //ocean::gpu::calculate_vbo(&vbo::cuda_vbo_resource, g_time);
    switch (VBO_MODE) {
    case VBO_DIRECT_ONE_STEP: ocean::gpu::direct::one_step::calculate_vbo(&vbo::cuda_vbo_resource, g_time); break;
    case VBO_DIRECT_TWO_STEP: ocean::gpu::direct::two_step::calculate_vbo(&vbo::cuda_vbo_resource, g_time); break;
    case VBO_FFT: ocean::gpu::fft::calculate_vbo(&vbo::cuda_vbo_resource, g_time);
    }

    graphics::display();
}

bool graphics::gpu::initGL(int* argc, char** argv) {
    return graphics::initGL(argc, argv, "GPU: Wave Simulation");
}