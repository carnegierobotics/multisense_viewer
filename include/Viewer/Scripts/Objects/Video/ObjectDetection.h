/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Objects/ObjectDetection.h
 *
 * Copyright 2023
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2023-06-13, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_OBJECTDETECTION_H
#define MULTISENSE_VIEWER_OBJECTDETECTION_H


#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/Scripts/Private/YoloV7.h"

/** ObjectDetection class for Scripts. Inherits Base and RegisteredInFactory by default, but can be extended with any class in the ModelLoaders folder to draw a model. See \refitem MultiSenseCamera for an example of drawing models **/
class ObjectDetection: public VkRender::Base, public VkRender::RegisteredInFactory<ObjectDetection>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    ObjectDetection() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of ObjectDetection **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<ObjectDetection>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "ObjectDetection"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    ScriptType getType() override {return type;}
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override { this->type = _type; }

    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;

    void onDestroy() override;
    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = CRL_SCRIPT_TYPE_DEFAULT;

    std::unique_ptr<YOLOV7>  yolo;
    bool key = false;

};





#endif //MULTISENSE_VIEWER_OBJECTDETECTION_H
