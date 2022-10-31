#include <MultiSense/Src/Renderer/Renderer.h>

int main() {
    Renderer app("MultiSense Viewer");
    app.run();
    app.cleanUp();

    return 0;
}
