//
// Created by magnus on 8/2/22.
//

#ifndef MULTISENSE_VIEWER_IMGUI_USER_H
#define MULTISENSE_VIEWER_IMGUI_USER_H

#include "imgui_internal.h"
#include "string"

namespace ImGui {

//IMGUI_API bool          CustomSelectable(const char* label, bool* p_selected, ImGuiSelectableFlags flags = 0, const ImVec2& size = ImVec2(0, 0));      // "bool* p_selected" point to the selection state (read-write), as a convenient helper.
//IMGUI_API bool          CustomSelectable(const char *label, bool selected, ImGuiSelectableFlags flags, const ImVec2 &size_arg);

    inline bool CustomSelectable(const char *label, bool selected, bool highlight, ImGuiSelectableFlags flags,
                                 const ImVec2 &size_arg) {
        ImGuiWindow *window = GetCurrentWindow();
        if (window->SkipItems)
            return false;

        ImGuiContext &g = *GImGui;
        const ImGuiStyle &style = g.Style;

        // Submit label or explicit size to ItemSize(), whereas ItemAdd() will submit a larger/spanning rectangle.
        ImGuiID id = window->GetID(label);
        ImVec2 label_size = CalcTextSize(label, NULL, true);
        ImVec2 size(size_arg.x != 0.0f ? size_arg.x : label_size.x, size_arg.y != 0.0f ? size_arg.y : label_size.y);
        ImVec2 pos = window->DC.CursorPos;
        pos.y += window->DC.CurrLineTextBaseOffset;
        ItemSize(size, 0.0f);

        // Fill horizontal space
        // We don't support (size < 0.0f) in Selectable() because the ItemSpacing extension would make explicitly right-aligned sizes not visibly match other widgets.
        const bool span_all_columns = (flags & ImGuiSelectableFlags_SpanAllColumns) != 0;
        const float min_x = span_all_columns ? window->ParentWorkRect.Min.x : pos.x;
        const float max_x = span_all_columns ? window->ParentWorkRect.Max.x : window->WorkRect.Max.x;
        if (size_arg.x == 0.0f || (flags & ImGuiSelectableFlags_SpanAvailWidth))
            size.x = ImMax(label_size.x, max_x - min_x);

        // Text stays at the submission position, but bounding box may be extended on both sides
        ImVec2 text_min = pos;
        text_min.x += 10.0f;
        text_min.y += 7.0f;
        const ImVec2 text_max(min_x + size.x, pos.y + size.y);

        // Selectables are meant to be tightly packed together with no click-gap, so we extend their box to cover spacing between selectable.
        ImRect bb(min_x, pos.y, text_max.x, text_max.y);

        //if (g.IO.KeyCtrl) { GetForegroundDrawList()->AddRect(bb.Min, bb.Max, IM_COL32(0, 255, 0, 255)); }

        // Modify ClipRect for the ItemAdd(), faster than doing a PushColumnsBackground/PushTableBackground for every Selectable..
        const float backup_clip_rect_min_x = window->ClipRect.Min.x;
        const float backup_clip_rect_max_x = window->ClipRect.Max.x;
        if (span_all_columns) {
            window->ClipRect.Min.x = window->ParentWorkRect.Min.x;
            window->ClipRect.Max.x = window->ParentWorkRect.Max.x;
        }

        const bool disabled_item = (flags & ImGuiSelectableFlags_Disabled) != 0;
        const bool item_add = ItemAdd(bb, id, NULL, disabled_item ? ImGuiItemFlags_Disabled : ImGuiItemFlags_None);
        if (span_all_columns) {
            window->ClipRect.Min.x = backup_clip_rect_min_x;
            window->ClipRect.Max.x = backup_clip_rect_max_x;
        }

        if (!item_add)
            return false;

        const bool disabled_global = (g.CurrentItemFlags & ImGuiItemFlags_Disabled) != 0;
        if (disabled_item && !disabled_global) // Only testing this as an optimization
            BeginDisabled();

        // FIXME: We can standardize the behavior of those two, we could also keep the fast path of override ClipRect + full push on render only,
        // which would be advantageous since most selectable are not selected.
        if (span_all_columns && window->DC.CurrentColumns)
            PushColumnsBackground();
        else if (span_all_columns && g.CurrentTable)
            TablePushBackgroundChannel();

        // We use NoHoldingActiveID on menus so user can click and _hold_ on a menu then drag to browse child entries
        ImGuiButtonFlags button_flags = 0;
        if (flags & ImGuiSelectableFlags_NoHoldingActiveID) { button_flags |= ImGuiButtonFlags_NoHoldingActiveId; }
        if (flags & ImGuiSelectableFlags_SelectOnClick) { button_flags |= ImGuiButtonFlags_PressedOnClick; }
        if (flags & ImGuiSelectableFlags_SelectOnRelease) { button_flags |= ImGuiButtonFlags_PressedOnRelease; }
        if (flags & ImGuiSelectableFlags_AllowDoubleClick) {
            button_flags |= ImGuiButtonFlags_PressedOnClickRelease |
                            ImGuiButtonFlags_PressedOnDoubleClick;
        }
        if (flags & ImGuiSelectableFlags_AllowItemOverlap) { button_flags |= ImGuiButtonFlags_AllowItemOverlap; }

        const bool was_selected = selected;
        bool hovered, held;
        bool pressed = ButtonBehavior(bb, id, &hovered, &held, button_flags);

        // Auto-select when moved into
        // - This will be more fully fleshed in the range-select branch
        // - This is not exposed as it won't nicely work with some user side handling of shift/control
        // - We cannot do 'if (g.NavJustMovedToId != id) { selected = false; pressed = was_selected; }' for two reasons
        //   - (1) it would require focus scope to be set, need exposing PushFocusScope() or equivalent (e.g. BeginSelection() calling PushFocusScope())
        //   - (2) usage will fail with clipped items
        //   The multi-select API aim to fix those issues, e.g. may be replaced with a BeginSelection() API.
        if ((flags & ImGuiSelectableFlags_SelectOnNav) && g.NavJustMovedToId != 0 &&
            g.NavJustMovedToFocusScopeId == window->DC.NavFocusScopeIdCurrent)
            if (g.NavJustMovedToId == id)
                selected = pressed = true;

        // Update NavId when clicking or when Hovering (this doesn't happen on most widgets), so navigation can be resumed with gamepad/keyboard
        if (pressed || (hovered && (flags & ImGuiSelectableFlags_SetNavIdOnHover))) {
            if (!g.NavDisableMouseHover && g.NavWindow == window && g.NavLayer == window->DC.NavLayerCurrent) {
                SetNavID(id, window->DC.NavLayerCurrent, window->DC.NavFocusScopeIdCurrent,
                         WindowRectAbsToRel(window, bb)); // (bb == NavRect)
                g.NavDisableHighlight = true;
            }
        }
        if (pressed)
            MarkItemEdited(id);

        if (flags & ImGuiSelectableFlags_AllowItemOverlap)
            SetItemAllowOverlap();

        // In this branch, Selectable() cannot toggle the selection so this will never trigger.
        if (selected != was_selected) //-V547
            g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_ToggledSelection;

        // Render
        if (held && (flags & ImGuiSelectableFlags_DrawHoveredWhenHeld))
            hovered = true;

        if (hovered || highlight) {
            const ImU32 col = GetColorU32(ImGuiCol_Header);
            RenderFrame(bb.Min, bb.Max, col, false, 10.0f);
        }
        RenderNavHighlight(bb, id, ImGuiNavHighlightFlags_TypeThin | ImGuiNavHighlightFlags_NoRounding);

        if (span_all_columns && window->DC.CurrentColumns)
            PopColumnsBackground();
        else if (span_all_columns && g.CurrentTable)
            TablePopBackgroundChannel();

        RenderTextClipped(text_min, text_max, label, NULL, &label_size, style.SelectableTextAlign, &bb);

        // Automatically close popups
        if (pressed && (window->Flags & ImGuiWindowFlags_Popup) && !(flags & ImGuiSelectableFlags_DontClosePopups) &&
            !(g.LastItemData.InFlags & ImGuiItemFlags_SelectableDontClosePopup))
            CloseCurrentPopup();

        if (disabled_item && !disabled_global)
            EndDisabled();

        IMGUI_TEST_ENGINE_ITEM_INFO(id, label, g.LastItemData.StatusFlags);
        return pressed; //-V1020
    }


    inline bool CustomSelectable(const char *label, bool *p_selected, bool highlight, ImGuiSelectableFlags flags,
                                 const ImVec2 &size_arg) {
        if (CustomSelectable(label, *p_selected, highlight, flags, size_arg)) {
            *p_selected = !*p_selected;
            return true;
        }
        return false;
    }

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
    CustomInvisibleButton(const char *str_id, bool *hovered, const ImVec2 &size_arg, ImGuiButtonFlags flags) {
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

        bool held;
        bool pressed = ButtonBehavior(bb, id, hovered, &held, flags);

        IMGUI_TEST_ENGINE_ITEM_INFO(id, str_id, g.LastItemData.StatusFlags);
        return pressed;
    }

    inline void
    ImageButtonText(const char *str_id, int *idx, int defaultValue, const ImVec2 btnSize, ImTextureID user_texture_id,
                    const ImVec2 &size,
                    const ImVec2 &uv0,
                    const ImVec2 &uv1,
                    const ImVec4 &bg_col, const ImVec4 &tint_col) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = g.CurrentWindow;
        if (window->SkipItems)
            return;

        ImVec2 posMinScreen = ImGui::GetCursorScreenPos();
        ImVec2 posMin = ImGui::GetCursorPos();

        ImGui::SetCursorScreenPos(posMinScreen);
        ImGui::PushID(1);
        bool hovered = false;
        if (CustomInvisibleButton(str_id, &hovered, btnSize, 0)) {
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
        ImGuiWindow *cur_window = g.CurrentWindow;
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
