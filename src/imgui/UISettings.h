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
    std::string dropdown;
    std::string selected;
    std::vector<std::string> dropDownItems;

    explicit DropDownItem(std::string type) {
        scriptType = std::move(type);
    }

    std::string scriptType;
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
    uint32_t idCount = 0;

    std::vector<ElementBase> elements;

    void createText(Text *text) {
        elements.push_back((ElementBase(text)));
    }
    void createButton(Button *button) {
        elements.push_back((ElementBase(button)));
    }
    void createDropDown(DropDownItem* dropDown){
        elements.emplace_back(dropDown);
    }

    /*
         struct Button {
        std::string text;
        float x;
        float y;
        bool clicked = false;

        Button(std::string _name,
               float size_x,
               float size_y) {
            text = std::move(_name);
            x = size_x;
            y = size_y;
        }
    };


    struct Text {
        std::string text;
        ImVec4 color;
        bool sameLine = false;

        explicit Text(std::string setText) {
            text = std::move(setText);
            color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        }

        Text(std::string setText, ImVec4 textColor) {
            text = std::move(setText);
            color = textColor;
        }
    };

    //std::vector<Text *> texts;


    //std::vector<Button *> buttons;


    void createButton(Button *button) {

        elements.push_back(std::shared_ptr<ElementBase>(new ElementTypes(button)));

        //buttons.emplace_back(button);
    }

     */
    struct intSlider {
        std::string name;
        int lowRange{};
        int highRange{};
        int val{};

        explicit intSlider(std::string type) {
            scriptType = std::move(type);
        }

        std::string scriptType;

    };

    struct DropDownItem {
        std::string dropdown;
        std::string selected;

        explicit DropDownItem(std::string type) {
            scriptType = std::move(type);
        }

        std::string scriptType;
    };



    bool rotate = true;
    bool displayLogos = true;
    bool displayBackground = true;
    bool toggleGridSize = true;
    bool toggleDepthImage = false;

    float movementSpeed = 0.2;
    std::array<float, 50> frameTimes{};
    float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
    float lightTimer = 0.0f;

    std::vector<std::string> listBoxNames;
    uint32_t selectedListboxIndex = 0;

    void insertListboxItem(const std::string &item) {
        listBoxNames.push_back(item);
    }

    uint32_t getSelectedItem() {
        return selectedListboxIndex;
    }

    std::vector<intSlider *> intSliders;

    void createIntSlider(intSlider *slider) {
        if (slider->scriptType != "Render")
            return;
        intSliders.emplace_back(slider);
    }

    std::vector<std::string> dropDownItems;
    const char *selectedDropDown = "Grayscale";

    void createDropDown(DropDownItem *items) {
        if (items->scriptType != "Render")
            return;
        dropDownItems.emplace_back(items->dropdown);
    }


};


#endif //MULTISENSE_UISETTINGS_H
