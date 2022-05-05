//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include "Layer.h"

class InteractionMenu : public Layer {
public:

// Create global object for convenience in other functions
    GuiObjectHandles *handles;

    void onFinishedRender() override {

    }

    void OnUIRender(GuiObjectHandles *_handles) override {

    }
};

#endif //MULTISENSE_INTERACTIONMENU_H
