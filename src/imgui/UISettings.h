//
// Created by magnus on 9/23/21.
//

#ifndef MULTISENSE_UISETTINGS_H
#define MULTISENSE_UISETTINGS_H


#include <utility>
#include <vector>
#include <string>
#include <array>
#include "imgui.h"
#include <memory>

typedef enum UIElement {
    AR_ELEMENT_TEXT,
    AR_ELEMENT_BUTTON,
    AR_ELEMENT_FLOAT_SLIDER,
    AR_UI_ELEMENT_DROPDOWN
} UIElement;


struct Button {
    std::string text;
    ImVec2 size;
    bool clicked = false;

    Button(std::string _name,
           float size_x,
           float size_y) {
        text = std::move(_name);
        size = ImVec2(size_x, size_y);
    }
};


struct Text {
    std::string string;
    ImVec4 color;
    bool sameLine = false;

    explicit Text(std::string setText) {
        string = std::move(setText);
        color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    Text(std::string setText, ImVec4 textColor) {
        string = std::move(setText);
        color = textColor;
    }
};

struct DropDownItem {
    std::string label;
    std::string selected;
    std::vector<std::string> dropDownItems;

    explicit DropDownItem(std::string label){
        this->label = std::move(label);
    }

};

class ElementBase {
public:
    Button* button{};
    Text* text{};
    DropDownItem* dropDown{};
    UIElement type{};

    explicit ElementBase(Button* btn) : button(btn), type(AR_ELEMENT_BUTTON) {}
    explicit ElementBase(Text* txt) : text(txt), type(AR_ELEMENT_TEXT) {}
    explicit ElementBase(DropDownItem* _dropDown) : dropDown(_dropDown), type(AR_UI_ELEMENT_DROPDOWN) {}


};


struct UISettings {

public:


    bool rotate = true;
    bool displayBackground = true;
    bool toggleGridSize = true;

    float movementSpeed = 0.2;
    std::array<float, 50> frameTimes{};
    float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
    float lightTimer = 0.0f;

    /** void* for shared data among scripts. User defined type */
    void* sharedData;
    bool* sharedButtonData;
    /**
     * Listbox containing name of all scripts currently used in the scene
     */
    std::vector<std::string> listBoxNames;
    uint32_t selectedListboxIndex = 0;

    // UI Element creations
    /** A vector containing each element that should be drawn on the ImGUI */
    std::vector<ElementBase> elements;

    /**
     * Creates Text
     * @param text
     */
    void createText(Text *text) {
        elements.push_back((ElementBase(text)));
    }
    void createButton(Button *button) {
        elements.push_back((ElementBase(button)));
    }
    void createDropDown(DropDownItem* dropDown){
        elements.emplace_back(dropDown);
    }
};


#endif //MULTISENSE_UISETTINGS_H
