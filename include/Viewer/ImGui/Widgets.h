//
// Created by magnus on 2/25/23.
//

#ifndef MULTISENSE_VIEWER_WIDGETS_H
#define MULTISENSE_VIEWER_WIDGETS_H

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
        bool *active = nullptr;
        bool activeVal = false;
        std::string id;
        ScriptWidgetType type{};

        Element(const char *labelVal, float *valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
                                                                                   minValue(minVal), maxValue(maxVal) {
            type = WIDGET_FLOAT_SLIDER;
        }

        Element(const char *labelVal, int *valPtr, int minVal, int maxVal, bool *valueChanged) : label(labelVal),
                                                                                                 intValue(valPtr),
                                                                                                 intMin(minVal),
                                                                                                 intMax(maxVal),
                                                                                                 active(valueChanged) {
            type = WIDGET_INT_SLIDER;
            if (valueChanged == nullptr) {
                active = &activeVal;
            }
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

        Element(const char *labelVal, char *_buf, bool *btn, ScriptWidgetType _type) : label(labelVal), button(btn), buf(_buf),  type(_type) {
        }
    };

    static Widgets *m_Instance;

    bool labelExists(const char *label, const std::string &window) {
        for (const auto& elem : elements[window]){
            if (elem.label == label){
                Log::Logger::getInstance()->warning("Label {} already exists in window {}", label, window.c_str());
                return true;
            }
        }

        return false;
    }

public:
    std::unordered_map<std::string, std::vector<Element>> elements;

    void slider(std::string window, const char *label, float *value, float min = 0.0f, float max = 1.0f) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, value, min, max);
    }

    void
    slider(std::string window, const char *label, int *value, int min = 0, int max = 10, bool *valueChanged = nullptr) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, value, min, max, valueChanged);
    }

    void text(std::string window, const char *label) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label);
    }

    void updateText(std::string id, std::string prevLabel, std::string newLabel) {
        for (auto &el: elements[id]) {
            if (el.label == prevLabel) {
                el.label = newLabel;
            }
        }
    }

    void checkbox(std::string window, const char *label, bool *val) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, val);
    }

    void button(std::string window, const char *label, bool *val) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, val, WIDGET_BUTTON);
    }

    void inputText(std::string window, const char *label, char *buf) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, buf);
    }


    void fileDialog(std::string window, const char *label, char *buf, bool* btn) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, buf, btn, WIDGET_SELECT_DIR_DIALOG);

    }

    static Widgets *make();

    static void clear();
};


#endif //MULTISENSE_VIEWER_WIDGETS_H
