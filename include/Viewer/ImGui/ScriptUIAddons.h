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
        float minValue = 0.0f;
        float maxValue = 1.0f;
        ScriptWidgetType type{};

        Element(const char *labelVal, float *valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
            minValue(minVal), maxValue(maxVal) {
            type = WIDGET_FLOAT_SLIDER;
        }

        int *intValue = nullptr;
        int intMin = 0;
        int intMax = 1;
        Element(const char *labelVal, int *valPtr, int minVal, int maxVal) : label(labelVal), intValue(valPtr),
                                                                             intMin(minVal), intMax(maxVal) {
            type = WIDGET_INT_SLIDER;
        }

        Element(const char *labelVal) : label(labelVal) {
            type = WIDGET_TEXT;
        }
    };
    static Widgets* m_Instance;

public:


    std::vector<Element> elements;

    void slider(const char *label, float *value, float min = 0.0f, float max = 1.0f) {
        elements.emplace_back(label, value, min, max);
    }

    void slider(const char *label, int *value, int min = 0, int max = 10) {
        elements.emplace_back(label, value, min, max);
    }

    void text(const char* label){
        elements.emplace_back(label);
    }

    static Widgets* make();
    static void clear();
};




#endif //MULTISENSE_VIEWER_SCRIPTUIADDONS_H
