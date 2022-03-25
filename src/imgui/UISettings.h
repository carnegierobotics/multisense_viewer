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
    AR_UI_ELEMENT_DROPDOWN,
    AR_UI_ELEMENT_INPUT_TEXT
} UIElement;


struct Button {
    std::string string;
    ImVec2 size;
    bool clicked = false;

    Button(std::string _name,
           float size_x,
           float size_y) {
        string = std::move(_name);
        size = ImVec2(size_x, size_y);
    }
};

struct InputText {
    char string[64] = "MultiSense";
    ImVec2 size;
    bool clicked = false;

    InputText(const char *_name,
              float size_x,
              float size_y) {

        memcpy(string, _name, 64);
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

    explicit DropDownItem(std::string label) {
        this->label = std::move(label);
    }

};

class ElementBase {
public:
    Button *button{};
    Text *text{};
    DropDownItem *dropDown{};
    UIElement type{};
    std::string location;
    InputText* inputText;
    struct {
        float x = 0.0f;
        float y = 0.0f;
    } pos;

    explicit ElementBase(Button *btn) : button(btn), type(AR_ELEMENT_BUTTON) {}

    explicit ElementBase(Text *txt) : text(txt), type(AR_ELEMENT_TEXT) {}

    explicit ElementBase(DropDownItem *_dropDown) : dropDown(_dropDown), type(AR_UI_ELEMENT_DROPDOWN) {}

    explicit ElementBase(InputText *_inputText) : inputText(_inputText), type(AR_UI_ELEMENT_INPUT_TEXT) {}

    ElementBase() = default;

};

class SideBarElement : ElementBase {

    SideBarElement() {


    }

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


    bool closeModalPopup = false;


    /** void* for shared data among scripts. User defined type */
    void *sharedData;
    bool *sharedButtonData;
    /**
     * Listbox containing name of all scripts currently used in the scene
     */
    std::vector<std::string> listBoxNames;
    uint32_t selectedListboxIndex = 0;

    // UI Element creations
    /** A vector containing each element that should be drawn on the ImGUI */
    std::vector<ElementBase> elements;
    std::vector<ElementBase> modalElements;

    /**
     * Creates Text
     * @param text
     */
    void createText(Text *text, std::string field, float x, float y) {
        auto elem = ElementBase(text);
        elem.location = std::move(field);
        elem.pos.x = x;
        elem.pos.y = y;
        elements.push_back((elem));
    }

    void createButton(Button *button, std::string field, float x, float y) {
        auto elem = ElementBase(button);
        elem.location = std::move(field);
        elem.pos.x = x;
        elem.pos.y = y;
        elements.push_back((elem));
    }

    void createDropDown(DropDownItem *dropDown, std::string field, float x, float y) {
        auto elem = ElementBase(dropDown);
        elem.location = std::move(field);
        elem.pos.x = x;
        elem.pos.y = y;
        elements.emplace_back(elem);
    }

    void addModalButton(Button *button) {
        auto elem = ElementBase(button);
        modalElements.push_back((elem));
    }
    void addModalText(InputText *text) {
        auto elem = ElementBase(text);
        modalElements.push_back((elem));
    }
    void createSideBarElement() {

    }
};


#endif //MULTISENSE_UISETTINGS_H
