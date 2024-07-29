//
// Created by magnus on 2/25/23.
//

#ifndef MULTISENSE_VIEWER_WIDGETS_H
#define MULTISENSE_VIEWER_WIDGETS_H

#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/pch.h"


enum ScriptWidgetType {
    WIDGET_FLOAT_SLIDER,
    WIDGET_INT_SLIDER,
    WIDGET_INPUT_NUMBER,
    WIDGET_TEXT,
    WIDGET_CHECKBOX,
    WIDGET_INPUT_TEXT,
    WIDGET_BUTTON,
    WIDGET_SELECT_DIR_DIALOG,
    WIDGET_GLM_VEC_3,
};

enum ScriptWidgetPlacement {
    WIDGET_PLACEMENT_RENDERER3D,
    WIDGET_PLACEMENT_IMU,
    WIDGET_PLACEMENT_POINTCLOUD,
    WIDGET_PLACEMENT_MULTISENSE_RENDERER,
};

class Widgets {
public:
    static Widgets* make();
    static void clear();

    void slider(ScriptWidgetPlacement window, const char* label, float* value, float min = 0.0f, float max = 1.0f);
    void vec3(ScriptWidgetPlacement window, const char* label, glm::vec3* value);
    void slider(ScriptWidgetPlacement window, const char* label, int* value, int min = 0, int max = 10, bool* valueChanged = nullptr);
    void text(ScriptWidgetPlacement window, const char* label);
    void updateText(ScriptWidgetPlacement window, const std::string& prevLabel, const std::string& newLabel);
    void checkbox(ScriptWidgetPlacement window, const char* label, bool* val);
    void button(ScriptWidgetPlacement window, const char* label, bool* val);
    void inputText(ScriptWidgetPlacement window, const char* label, char* buf);
    void fileDialog(ScriptWidgetPlacement window, const char* label, std::future<std::filesystem::path>* future);

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
        std::string id;
        std::future<std::filesystem::path> *future{};
        ScriptWidgetType type{};

        struct {
            glm::vec3 *vec3{};
            char xBuf[16] = {0};
            char yBuf[16] = {0};
            char zBuf[16] = {0};
        } glm;

        /** FLOAT SLIDER **/
        Element(const char *labelVal, float *valPtr, float minVal, float maxVal) : label(labelVal), value(valPtr),
                                                                                   minValue(minVal), maxValue(maxVal),
                                                                                   type(WIDGET_FLOAT_SLIDER) {}
        /** INT SLIDER **/
        Element(const char *labelVal, int *valPtr, int minVal, int maxVal, bool *valueChanged) : label(labelVal),
                                                                                                 intValue(valPtr),
                                                                                                 intMin(minVal),
                                                                                                 intMax(maxVal),
                                                                                                 active(valueChanged),
                                                                                                 type(WIDGET_INT_SLIDER) {}
        /** PLAIN TEXT **/
        explicit Element(const char *labelVal, std::string _id = "") : label(labelVal),
                                                                       id(std::move(_id)),
                                                                       type(WIDGET_TEXT) {}
        /** VEC3 DISPLAY **/
        Element(const char *labelVal, glm::vec3 *valPtr) : label(labelVal),
                                                           type(WIDGET_GLM_VEC_3) {
            glm.vec3 = valPtr;
        }
        /** CHECKBOX **/
        Element(const char *labelVal, bool *check) : label(labelVal),
                                                     checkbox(check),
                                                     type(WIDGET_CHECKBOX) {}
        /** BUTTON WIDGET **/
        Element(const char *labelVal, bool *btn, ScriptWidgetType _type) : label(labelVal),
                                                                           button(btn),
                                                                           type(_type) {}
        /** INPUT TEXT FIELD **/
        Element(const char *labelVal, char *_buf) : label(labelVal),
                                                    buf(_buf),
                                                    type(WIDGET_INPUT_TEXT) {}
        /** FILE DIALOG WIDGET **/
        Element(const char *labelVal, std::future<std::filesystem::path> *_future, ScriptWidgetType _type) : label(labelVal),
                                                                                                   future(_future),
                                                                                                   type(_type) {}
    };
    std::unordered_map<ScriptWidgetPlacement, std::vector<Element>> elements;
private:

    bool labelExists(const char* label, const ScriptWidgetPlacement& window);
    static Widgets* m_Instance;

    Widgets() = default;
    ~Widgets() = default;
};

#endif //MULTISENSE_VIEWER_WIDGETS_H
