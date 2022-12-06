#include "Viewer/Renderer/Renderer.h"

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