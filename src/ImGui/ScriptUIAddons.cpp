//
// Created by magnus on 2/25/23.
//


#include "Viewer/ImGui/ScriptUIAddons.h"
Widgets *Widgets::m_Instance = nullptr;

Widgets *Widgets::make() {
    if (m_Instance == nullptr){
        m_Instance = new Widgets();
    }
    return m_Instance;
}

void Widgets::clear() {
    if (m_Instance != nullptr) {
        m_Instance->elements.clear();
    }
}
