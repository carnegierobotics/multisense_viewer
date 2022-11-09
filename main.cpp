#include <MultiSense/Src/Renderer/Renderer.h>

#define WIN_DEBUG
#ifdef WIN32
    #ifdef WIN_DEBUG
        #pragma comment(linker, "/SUBSYSTEM:CONSOLE")
    #endif
#endif

int main() {
    Renderer app("MultiSense Viewer");
    app.run();
    app.cleanUp();

    return 0;
}