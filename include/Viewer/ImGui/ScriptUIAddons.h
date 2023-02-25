//
// Created by magnus on 2/25/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPTUIADDONS_H
#define MULTISENSE_VIEWER_SCRIPTUIADDONS_H

#include "Viewer/Core/Definitions.h"

class Widgets {

private:
    struct Element {
        const char *label;
        float *value = nullptr;
        float min = 0.0f;
        float max = 1.0f;
        ScriptWidgetType type{};

        Element(const char *labelVal, float *valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
                                                                                   min(minVal), max(maxVal) {
            type = FLOAT_SLIDER;
        }
    };
    static Widgets* m_Instance;

public:


    std::vector<Element> elements;

    void slider(const char *label, float *value, float min = 0.0f, float max = 1.0f) {
        elements.emplace_back(label, value, min, max);
    }

    static Widgets* make();
};




#endif //MULTISENSE_VIEWER_SCRIPTUIADDONS_H
