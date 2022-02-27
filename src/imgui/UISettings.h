//
// Created by magnus on 9/23/21.
//

#ifndef MULTISENSE_UISETTINGS_H
#define MULTISENSE_UISETTINGS_H


#include <utility>
#include <vector>
#include <string>
#include <array>

struct UISettings {

public:

    bool fa = false;
    bool flag = false;

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
    void createDropDown(DropDownItem* items){
        if (items->scriptType != "Render")
            return;
        dropDownItems.emplace_back(items->dropdown);
    }


};


#endif //MULTISENSE_UISETTINGS_H
