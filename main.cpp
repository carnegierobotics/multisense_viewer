
// If a console is needed in the background then define WIN_DEBUG
// Can be usefull for reading std::out ...
#ifdef WIN32
    #define WIN_DEBUG
#endif

#include <MultiSense/src/Renderer/Renderer.h>
Renderer *application;
int main() {
    application = new Renderer("MultiSense Viewer");
    application->run();
    return 0;
}
