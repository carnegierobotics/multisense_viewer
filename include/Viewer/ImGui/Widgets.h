//
// Created by magnus on 2/25/23.
//

#ifndef MULTISENSE_VIEWER_WIDGETS_H
#define MULTISENSE_VIEWER_WIDGETS_H

#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Tools/Logger.h"

typedef enum ScriptWidgetType {
    WIDGET_FLOAT_SLIDER = 0,
    WIDGET_INT_SLIDER = 1,
    WIDGET_INPUT_NUMBER = 2,
    WIDGET_TEXT = 3,
    WIDGET_CHECKBOX = 4,
    WIDGET_INPUT_TEXT = 5,
    WIDGET_BUTTON = 6,
    WIDGET_SELECT_DIR_DIALOG = 7,
    WIDGET_GLM_VEC_3 = 8,
} ScriptWidgetType;

typedef enum ScriptWidgetPlacement {
    WIDGET_PLACEMENT_RENDERER3D = 0,
    WIDGET_PLACEMENT_IMU = 1,
    WIDGET_PLACEMENT_POINTCLOUD = 2,
    WIDGET_PLACEMENT_MULTISENSE_RENDERER = 3
} ScriptWidgetPlacement;

class Widgets {
private:
    struct Element {
        std::string label;
        float* value = nullptr;
        float minValue = 0.0f;
        float maxValue = 1.0f;
        bool* button = nullptr;
        bool* checkbox = nullptr;
        int* intValue = nullptr;
        int intMin = 0;
        int intMax = 1;
        char* buf = nullptr;
        bool* active = nullptr;
        bool activeVal = false;
        std::string id;
        ScriptWidgetType type{};

        struct {
            glm::vec3* vec3{};
            char xBuf[16] = {0};
            char yBuf[16] = {0};
            char zBuf[16] = {0};
        } glm;

        Element(const char* labelVal, float* valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
            minValue(minVal), maxValue(maxVal) {
            type = WIDGET_FLOAT_SLIDER;
        }

        Element(const char* labelVal, int* valPtr, int minVal, int maxVal, bool* valueChanged) : label(labelVal),
            intValue(valPtr),
            intMin(minVal),
            intMax(maxVal),
            active(valueChanged) {
            type = WIDGET_INT_SLIDER;
            if (valueChanged == nullptr) {
                active = &activeVal;
            }
        }

        Element(const char* labelVal, std::string _id = "") : label(labelVal), id(_id) {
            type = WIDGET_TEXT;
        }

        Element(const char* labelVal, glm::vec3* valPtr) : label(labelVal) {
            type = WIDGET_GLM_VEC_3;
            glm.vec3 = valPtr;

            // Initialize the array to zero
            std::fill_n(glm.xBuf, 16, 0);
            std::memcpy(glm.xBuf, &valPtr->x, sizeof(float));

            std::fill_n(glm.yBuf, 16, 0);
            std::memcpy(glm.yBuf, &valPtr->y, sizeof(float));

            std::fill_n(glm.zBuf, 16, 0);
            std::memcpy(glm.zBuf, &valPtr->z, sizeof(float));
        }


        Element(const char* labelVal, bool* check) : label(labelVal), checkbox(check) {
            type = WIDGET_CHECKBOX;
        }

        Element(const char* labelVal, bool* btn, ScriptWidgetType _type) : label(labelVal), button(btn), type(_type) {
        }

        Element(const char* labelVal, char* _buf) : label(labelVal), buf(_buf) {
            type = WIDGET_INPUT_TEXT;
        }

        /**File dialog constructor **/
        std::future<std::string>* future;

        Element(const char* labelVal, std::future<std::string>* _future, ScriptWidgetType _type) : label(labelVal),
            future(_future), type(_type) {
        }
    };

    static Widgets* m_Instance;

    bool labelExists(const char* label, const ScriptWidgetPlacement& window) {
        for (const auto& elem : elements[window]) {
            if (elem.label == label) {
                Log::Logger::getInstance()->warning("Label {} already exists in window {}", label,
                                                    static_cast<int>(window));
                return true;
            }
        }

        return false;
    }

public:
    std::unordered_map<ScriptWidgetPlacement, std::vector<Element>> elements;

    void slider(ScriptWidgetPlacement window, const char* label, float* value, float min = 0.0f, float max = 1.0f) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, value, min, max);
    }

    void vec3(ScriptWidgetPlacement window, const char* label, glm::vec3* value) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, value);
    }

    void
    slider(ScriptWidgetPlacement window, const char* label, int* value, int min = 0, int max = 10,
           bool* valueChanged = nullptr) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, value, min, max, valueChanged);
    }

    void text(ScriptWidgetPlacement window, const char* label) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label);
    }

    void updateText(ScriptWidgetPlacement window, std::string prevLabel, std::string newLabel) {
        for (auto& el : elements[window]) {
            if (el.label == prevLabel) {
                el.label = newLabel;
            }
        }
    }

    void checkbox(ScriptWidgetPlacement window, const char* label, bool* val) {
        auto& windowElements = elements[window];
        for (auto& element : windowElements) {
            if (element.label == label) {
                // Replace the existing value
                element.checkbox = val;
                return;
            }
        }

        elements[window].emplace_back(label, val);
    }

    void button(ScriptWidgetPlacement window, const char* label, bool* val) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, val, WIDGET_BUTTON);
    }

    void inputText(ScriptWidgetPlacement window, const char* label, char* buf) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, buf);
    }


    void fileDialog(ScriptWidgetPlacement window, const char* label, std::future<std::string>* future) {
        if (labelExists(label, window))
            return;
        elements[window].emplace_back(label, future, WIDGET_SELECT_DIR_DIALOG);
    }

    static Widgets* make();

    static void clear();
};


#endif //MULTISENSE_VIEWER_WIDGETS_H
