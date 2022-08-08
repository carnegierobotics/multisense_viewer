
// If a console is needed in the background then define WIN_DEBUG
// Can be usefull for reading std::out ...
#define WIN_DEBUG

// LOGGING_VERBOSE Macro used to enable logging in time critical places. For instance between every draw.
#define LOGGING_VERBOSE

#include <MultiSense/src/Renderer/Renderer.h>

Renderer *application;


int main() {

    application = new Renderer("MULTISENSE PREVIEW APPLICATION");
    application->run();
    return 0;
}
