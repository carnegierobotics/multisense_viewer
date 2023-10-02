//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_MAIN3D_H
#define MULTISENSE_VIEWER_MAIN3D_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"


class Main3D: public VkRender::Base, public VkRender::RegisteredInFactory<Main3D>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Main3D() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of Main3D **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Main3D>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "Main3D"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    ScriptType getType() override {return type;}

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = CRL_SCRIPT_TYPE_RENDER;
};


#endif //MULTISENSE_VIEWER_MAIN3D_H
