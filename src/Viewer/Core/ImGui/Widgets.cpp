//
// Created by magnus on 2/25/23.
//


#include <utility>

#include "Viewer/Core/ImGui/Widgets.h"


// Static member initialization
Widgets *Widgets::m_Instance = nullptr;

// Static methods
Widgets *Widgets::make() {
    if (!m_Instance) {
        m_Instance = new Widgets();
    }
    return m_Instance;
}

void Widgets::clear() {
    delete m_Instance;
    m_Instance = nullptr;
}

// Private methods
bool Widgets::labelExists(const char *label, const ScriptWidgetPlacement &window) {
    for (const auto &elem: elements[window]) {
        if (elem.label == label) {
            Log::Logger::getInstance()->warning("Label {} already exists in window {}", label,
                                                static_cast<int>(window));
            return true;
        }
    }
    return false;
}

// Public methods
void Widgets::slider(ScriptWidgetPlacement window, const char *label, float *value, float min, float max) {
    if (!labelExists(label, window)) {
        elements[window].emplace_back(label, value, min, max);
    }
}

void Widgets::vec3(ScriptWidgetPlacement window, const char *label, glm::vec3 *value) {
    if (!labelExists(label, window)) {
        elements[window].emplace_back(label, value);
    }
}

void
Widgets::slider(ScriptWidgetPlacement window, const char *label, int *value, int min, int max, bool *valueChanged) {
    if (!labelExists(label, window)) {
        elements[window].emplace_back(label, value, min, max, valueChanged);
    }
}

void Widgets::text(ScriptWidgetPlacement window, const char *label) {
    if (!labelExists(label, window)) {
        elements[window].emplace_back(label);
    }
}

void Widgets::updateText(ScriptWidgetPlacement window, const std::string& prevLabel, const std::string& newLabel) {
    for (auto &el: elements[window]) {
        if (el.label == prevLabel) {
            el.label = newLabel;
            break; // Assuming unique labels, we can break after the first match
        }
    }
}

void Widgets::checkbox(ScriptWidgetPlacement window, const char *label, bool *val) {
    if (!labelExists(label, window)) {
        elements[window].emplace_back(label, val);
    }
}

void Widgets::button(ScriptWidgetPlacement window, const char *label, bool *val) {
    if (!labelExists(label, window)) {
        elements[window].emplace_back(label, val, WIDGET_BUTTON);
    }
}

void Widgets::inputText(ScriptWidgetPlacement window, const char *label, char *buf) {
    if (labelExists(label, window))
        return;
    elements[window].emplace_back(label, buf);
}

void Widgets::fileDialog(ScriptWidgetPlacement window, const char *label, std::future<std::filesystem::path> *future) {
    if (labelExists(label, window))
        return;
    elements[window].emplace_back(label, future, WIDGET_SELECT_DIR_DIALOG);
}
