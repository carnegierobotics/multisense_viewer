//
// Created by magnus on 2/25/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPTUIADDONS_H
#define MULTISENSE_VIEWER_SCRIPTUIADDONS_H

#include "Viewer/Core/Definitions.h"
#include "Viewer/Tools/Logger.h"

class Widgets {

private:
    struct Element {
        std::string label;
        float *value = nullptr;
        float minValue = 0.0f;
        float maxValue = 1.0f;
        bool *button = nullptr;
        bool *checkbox = nullptr;
        int *intValue = nullptr;
        int intMin = 0;
        int intMax = 1;
        char *buf = nullptr;
        std::string id;
        ScriptWidgetType type{};

        Element(const char *labelVal, float *valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
                                                                                   minValue(minVal), maxValue(maxVal) {
            type = WIDGET_FLOAT_SLIDER;
        }

        Element(const char *labelVal, int *valPtr, int minVal, int maxVal) : label(labelVal), intValue(valPtr),
                                                                             intMin(minVal), intMax(maxVal) {
            type = WIDGET_INT_SLIDER;
        }

        Element(const char *labelVal, std::string _id = "") : label(labelVal), id(_id) {
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

    bool labelExists(const char *label, const std::string &window) {
        std::vector<Element> vec;
        if (window.compare("default") == 0)
            vec = elements;
        else if (window.compare("Renderer3D") == 0)
            vec = elements3D;

        for (const auto &elem: vec) {
            if (elem.label == label) {
                Log::Logger::getInstance()->info("Label {} already exists in window {} Widget maker", label, window);
                return true;
            }
        }
        return false;
    }

public:


    std::vector<Element> elements;
    std::vector<Element> elements3D;

    void slider(std::string window, const char *label, float *value, float min = 0.0f, float max = 1.0f) {
        if (labelExists(label, window))
            return;
        if (window == "default")
            elements.emplace_back(label, value, min, max);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, value, min, max);

    }

    void slider(std::string window, const char *label, int *value, int min = 0, int max = 10) {
        if (labelExists(label, window))
            return;
        if (window == "default")
            elements.emplace_back(label, value, min, max);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, value, min, max);
    }

    void text(std::string window, const char *label, std::string id = "") {
        if (labelExists(label, window))
            return;
        if (window == "default")
            elements.emplace_back(label);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, id);
    }

    void updateText(std::string id, std::string newLabel) {
        // TODO implement for other than renderer3D
        for (auto &el: elements3D) {
            if(el.id == id){
                el.label = newLabel;
            }
        }
    }

    void checkbox(std::string window, const char *label, bool *val) {
        if (labelExists(label, window))
            return;
        if (window == "default")
            elements.emplace_back(label, val);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, val);
    }

    void button(std::string window, const char *label, bool *val) {
        if (labelExists(label, window))
            return;
        if (window == "default")
            elements.emplace_back(label, val, WIDGET_BUTTON);
        else if (window == "Renderer3D")
            elements3D.emplace_back(label, val, WIDGET_BUTTON);
    }

    void inputText(std::string window, const char *label, char *buf) {
        if (labelExists(label, window))
            return;
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
