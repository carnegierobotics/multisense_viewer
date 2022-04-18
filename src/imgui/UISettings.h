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
    AR_UI_ELEMENT_INPUT_TEXT,
    AR_UI_ELEMENT_SIDEBAR_DEVICE
} UIElement;


struct Button {
    char string[64] = "Btn";
    ImVec2 size;
    bool clicked = false;

    Button(const std::string name,
           float size_x,
           float size_y) {

        strcpy(string, name.c_str());
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
    char string[64] = "Text";
    ImVec4 color;
    bool sameLine = false;

    explicit Text(std::string setText) {
        strcpy(string, setText.c_str());
        color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    Text(std::string setText, ImVec4 textColor) {
        strcpy(string, setText.c_str());
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
    InputText *inputText;
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

class SideBarElement {
public:

    Text *profileName{};
    Text *cameraName{};
    Button *activityState{};
    UIElement type{};

    struct {
        float x = 0.0f;
        float y = 0.0f;
    }pos;

    SideBarElement(Text *name, Text *camName, Button *btn, float posX, float posY) : profileName(name),
                                                                                     cameraName(camName),
                                                                                     activityState(btn),
                                                                                     type(AR_ELEMENT_BUTTON) {
        this->pos.x = posX;
        this->pos.y = posY;
    }

    ImVec2 getPos() const {
        return ImVec2{pos.x, pos.y};
    }

};


struct UISettings {

public:

    float movementSpeed = 0.2;

    std::array<float, 50> frameTimes{};
    float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;


    bool closeModalPopup = false;

    /** void* for shared data among scripts. User defined type */
    void *physicalCamera = nullptr;
    void *virtualCamera = nullptr;
    /**
     * Listbox containing name of all scripts currently used in the scene
     */
    std::vector<std::string> listBoxNames;
    uint32_t selectedListboxIndex = 0;

    // UI Element creations
    /** A vector containing each element that should be drawn on the ImGUI */
    std::vector<ElementBase> elements;
    std::vector<ElementBase> modalElements;
    std::vector<SideBarElement *> sidebarElements;

    /**
     * Creates Text
     * @param text
     */
    void createText(Text *text, std::string field, float posX, float posY) {
        auto elem = ElementBase(text);
        elem.location = std::move(field);
        elem.pos.x = posX;
        elem.pos.y = posY;
        elements.push_back((elem));
    }

    void createButton(Button *button, std::string field, float posX, float posY) {
        auto elem = ElementBase(button);
        elem.location = std::move(field);
        elem.pos.x = posX;
        elem.pos.y = posY;
        elements.push_back((elem));
    }

    void createDropDown(DropDownItem *dropDown, std::string field, float posX, float posY) {
        auto elem = ElementBase(dropDown);
        elem.location = std::move(field);
        elem.pos.x = posX;
        elem.pos.y = posY;
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

    void createSideBarElement(SideBarElement *elem) {
        sidebarElements.push_back(elem);
    }
};


#endif //MULTISENSE_UISETTINGS_H
