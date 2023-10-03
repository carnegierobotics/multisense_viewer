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
        bool *button = nullptr;
        bool *checkbox = nullptr;
        int *intValue = nullptr;
        int intMin = 0;
        int intMax = 1;
        char *buf = nullptr;

        ScriptWidgetType type{};

        Element(const char *labelVal, float *valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
                                                                                   minValue(minVal), maxValue(maxVal) {
            type = WIDGET_FLOAT_SLIDER;
        }

        Element(const char *labelVal, int *valPtr, int minVal, int maxVal) : label(labelVal), intValue(valPtr),
                                                                             intMin(minVal), intMax(maxVal) {
            type = WIDGET_INT_SLIDER;
        }

        Element(const char *labelVal) : label(labelVal) {
            type = WIDGET_TEXT;
        }


        Element(const char *labelVal, bool *check) : label(labelVal), checkbox(check) {
            type = WIDGET_CHECKBOX;
        }

        Element(const char *labelVal, bool *btn, ScriptWidgetType _type) : label(labelVal), button(btn), type(_type) {
        }

        Element(const char *labelVal, char *_buf) : label(labelVal), buf(_buf) {
            type = WIDGET_INPUT_TEXT;
        }
    };

    static Widgets *m_Instance;

public:


    std::vector<Element> elements;
    std::vector<Element> elements3D;

    void slider(std::string window, const char *label, float *value, float min = 0.0f, float max = 1.0f) {
        if (window == "default")
            elements.emplace_back(label, value, min, max);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, value, min, max);

    }

    void slider(std::string window, const char *label, int *value,  int min = 0, int max = 10) {
        if (window == "default")
            elements.emplace_back(label, value, min, max);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, value, min, max);
    }

    void text(std::string window, const char *label) {
        if (window == "default")
            elements.emplace_back(label);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label);
    }

    void checkbox(std::string window, const char *label, bool *val) {
        if (window == "default")
            elements.emplace_back(label, val);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, val);
    }
    void button(std::string window, const char *label, bool *val) {
        if (window == "default")
            elements.emplace_back(label, val, WIDGET_BUTTON);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, val, WIDGET_BUTTON);
    }

    void inputText( std::string window, const char *label, char *buf) {
        /**
         * Label names must not overlap
         */
        if (window == "default")
            elements.emplace_back(label, buf);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, buf);
    }

    static Widgets *make();

    static void clear();
};


#endif //MULTISENSE_VIEWER_SCRIPTUIADDONS_H
