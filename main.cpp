
#include <MultiSense/src/Renderer/Renderer.h>

Renderer *application;


int main() {

    application = new Renderer("MULTISENSE PREVIEW APPLICATION");
    application->run();
    return 0;
}
