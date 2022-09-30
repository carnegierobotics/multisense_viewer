
// If a console is needed in the background then define WIN_DEBUG
// Can be usefull for reading std::out ...
#ifdef WIN32
    #define WIN_DEBUG
#endif

#include <MultiSense/src/Renderer/Renderer.h>
int main() {
    Renderer app("MultiSense Viewer");
    app.run();
    return 0;
}
