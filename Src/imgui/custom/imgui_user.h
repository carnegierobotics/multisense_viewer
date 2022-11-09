//
// Created by magnus on 8/2/22.
//

#ifndef MULTISENSE_VIEWER_IMGUI_USER_H
#define MULTISENSE_VIEWER_IMGUI_USER_H

#include "imgui_internal.h"
#include "string"

/**
 * @brief Custom IMGUI modules made for this projectble
 */
namespace ImGui {
    // Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
    static void HelpMarker(const char *desc) {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }


    struct Funcs {
        static int MyResizeCallback(ImGuiInputTextCallbackData *data) {
            if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                auto *my_str = (std::string *) data->UserData;
                IM_ASSERT(my_str->data() == data->Buf);
                my_str->resize(
                        data->BufSize); // NB: On resizing calls, generally data->BufSize == data->BufTextLen + 1
                data->Buf = my_str->data();
            }
            return 0;
        }

        static bool
        MyInputText(const char *label, std::string *my_str, ImGuiInputTextFlags flags = 0) {
            IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
            return ImGui::InputText(label, my_str->data(), (size_t) my_str->size(),
                                    flags | ImGuiInputTextFlags_CallbackResize,
                                    Funcs::MyResizeCallback, (void *) my_str);
        }
    };

    // Tip: use ImGui::PushID()/PopID() to push indices or pointers in the ID stack.
// Then you can keep 'str_id' empty or the same for all your buttons (instead of creating a string based on a non-string id)
    inline bool
    HoveredInvisibleButton(const char *str_id, bool *hovered, bool *held, const ImVec2 &size_arg,
                           ImGuiButtonFlags flags) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = GetCurrentWindow();
        if (window->SkipItems)
            return false;

        // Cannot use zero-size for InvisibleButton(). Unlike Button() there is not way to fallback using the label size.
        IM_ASSERT(size_arg.x != 0.0f && size_arg.y != 0.0f);

        const ImGuiID id = window->GetID(str_id);
        ImVec2 size = CalcItemSize(size_arg, 0.0f, 0.0f);
        ImVec2 bbMax = window->DC.CursorPos;
        bbMax.x += size.x;
        bbMax.y += size.y;
        const ImRect bb(window->DC.CursorPos, bbMax);
        ItemSize(size);
        if (!ItemAdd(bb, id))
            return false;

        bool pressed = ButtonBehavior(bb, id, hovered, held, flags);

        IMGUI_TEST_ENGINE_ITEM_INFO(id, str_id, g.LastItemData.StatusFlags);
        return pressed;
    }

    inline void
    ImageButtonText(const char *str_id, int *idx, int defaultValue, const ImVec2 btnSize, ImTextureID user_texture_id,
                    const ImVec2 &size,
                    const ImVec2 &uv0,
                    const ImVec2 &uv1, const ImVec4 &tint_col) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = g.CurrentWindow;
        if (window->SkipItems)
            return;

        ImVec2 posMinScreen = ImGui::GetCursorScreenPos();
        ImVec2 posMin = ImGui::GetCursorPos();

        ImGui::SetCursorScreenPos(posMinScreen);
        ImGui::PushID(1);
        bool hovered = false;
        bool held;
        if (HoveredInvisibleButton(str_id, &hovered, &held, btnSize, 0)) {
            if (*idx != defaultValue)
                *idx = defaultValue;
            else if (*idx == defaultValue)
                *idx = -1; // reset variable

        }
        ImGui::PopID();
        ImGui::SameLine();
        // Screen pos for adding to windowDrawList rects
        ImVec2 posMax = posMinScreen;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;
        ImVec4 color = (*idx == defaultValue) ? ImColor(0.666f, 0.674f, 0.658f, 1.0f) :
                       hovered ? ImColor(0.462f, 0.474f, 0.494f, 1.0f) :
                       ImColor(0.411f, 0.419f, 0.407f, 1.0f);
        ImGui::GetWindowDrawList()->AddRectFilled(posMinScreen, posMax,
                                                  ImColor(color), 10.0f, 0);


        // Window relative pos for text and img element
        posMax = posMin;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;
        ImVec2 txtSize = ImGui::CalcTextSize(str_id);
        ImGui::SetCursorPos(ImVec2(posMin.x + 10.0f, posMin.y + ((posMax.y - posMin.y) / 2) - (txtSize.y / 2)));
        ImGui::Text("%s", str_id);
        ImGui::SameLine();
        ImGui::PushID(0);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ((btnSize.y - size.y) / 2));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((btnSize.x - size.x - txtSize.x - 30.0f)));
        ImageButtonEx(window->GetID(str_id), user_texture_id, size, uv0, uv1, color, tint_col);
        ImGui::PopID();
    };

    inline bool ButtonWithGif(const char *str_id, const ImVec2 btnSize, ImTextureID user_texture_id, const ImVec2 &size,
                              const ImVec2 &uv0, const ImVec2 &uv1, const ImVec4 &tint_col, ImVec4 btnColor) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = g.CurrentWindow;
        if (window->SkipItems)
            return false;

        ImVec2 posMinScreen = ImGui::GetCursorScreenPos();
        ImVec2 posMin = ImGui::GetCursorPos();

        ImGui::SetCursorScreenPos(posMinScreen);
        ImGui::PushID(1);
        bool hovered = false;
        bool held;
        bool clicked = false;
        if (HoveredInvisibleButton(str_id, &hovered, &held, btnSize, 0)) {
            clicked = true;
        }
        ImGui::PopID();
        ImGui::SameLine();
        // Screen pos for adding to windowDrawList rects
        ImVec2 posMax = posMinScreen;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;

        ImVec4 color = btnColor;
        ImGui::GetWindowDrawList()->AddRectFilled(posMinScreen, posMax,
                                                  ImColor(color), 10.0f, 0);

        // Window relative pos for text and img element
        posMax = posMin;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;
        ImVec2 txtSize = ImGui::CalcTextSize(str_id);
        ImGui::SetCursorPos(ImVec2(posMin.x + 40.0f, posMin.y + ((posMax.y - posMin.y) / 2) - (txtSize.y / 2)));
        ImGui::Text("%s", str_id);
        ImGui::SameLine();
        ImGui::PushID(0);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ((btnSize.y - size.y) / 2));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((btnSize.x - size.x - txtSize.x - 50.0f)));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        clicked |= ImageButtonEx(window->GetID(str_id), user_texture_id, size, uv0, uv1, color, tint_col);
        ImGui::PopStyleVar();
        ImGui::PopID();
        return clicked;
    };


    static inline bool isHoverable(ImGuiWindow *window, ImGuiHoveredFlags flags) {
        // An active popup disable hovering on other windows (apart from its own children)
        // FIXME-OPT: This could be cached/stored within the window.
        ImGuiContext &g = *GImGui;
        if (g.NavWindow)
            if (ImGuiWindow *focused_root_window = g.NavWindow->RootWindow)
                if (focused_root_window->WasActive && focused_root_window != window->RootWindow) {
                    // For the purpose of those flags we differentiate "standard popup" from "modal popup"
                    // NB: The order of those two tests is important because Modal windows are also Popups.
                    if (focused_root_window->Flags & ImGuiWindowFlags_Modal)
                        return false;
                    if ((focused_root_window->Flags & ImGuiWindowFlags_Popup) &&
                        !(flags & ImGuiHoveredFlags_AllowWhenBlockedByPopup))
                        return false;
                }
        return true;
    }

    inline bool IsWindowHoveredByName(std::string name, ImGuiHoveredFlags flags) {
        IM_ASSERT((flags & (ImGuiHoveredFlags_AllowWhenOverlapped | ImGuiHoveredFlags_AllowWhenDisabled)) ==
                  0);   // Flags not supported by this function
        ImGuiContext &g = *GImGui;
        ImGuiWindow *ref_window = g.HoveredWindow;
        if (ref_window == NULL)
            return false;

        if (g.HoveredWindow->RootWindow->Name != name)
            return false;

        if (!isHoverable(ref_window, flags))
            return false;
        if (!(flags & ImGuiHoveredFlags_AllowWhenBlockedByActiveItem))
            if (g.ActiveId != 0 && !g.ActiveIdAllowOverlap && g.ActiveId != ref_window->MoveId)
                return false;

        return true;
    }

}
#endif //MULTISENSE_VIEWER_IMGUI_USER_H
