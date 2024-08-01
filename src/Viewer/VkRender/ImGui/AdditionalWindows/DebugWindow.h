/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/ImGui/DebugWindow.h
 *
 * Copyright 2022
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
 *   2022-10-25, mgjerde@carnegierobotics.com, Created file.
 **/
#ifndef MULTISENSE_VIEWER_DEBUGWINDOW_H
#define MULTISENSE_VIEWER_DEBUGWINDOW_H

#include <future>

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Core/RendererConfig.h"
#include "Viewer/VkRender/UsageMonitor.h"

class DebugWindow : public VkRender::Layer {
public:
// Usage:
//  static ExampleAppLog my_log;
//  my_log.AddLog("Hello %d world\n", 123);
//  my_log.Draw("title");
    struct ExampleAppLog {
        ImGuiTextBuffer Buf;
        ImGuiTextFilter Filter;
        ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
        bool AutoScroll;  // Keep scrolling if already at the bottom.

        ExampleAppLog() {
            AutoScroll = true;
            Clear();
        }

        void Clear() {
            Buf.clear();
            LineOffsets.clear();
            LineOffsets.push_back(0);
        }

        void AddLog(const char *fmt, ...) IM_FMTARGS(2) {
            int old_size = Buf.size();
            va_list args;
                    va_start(args, fmt);
            Buf.appendfv(fmt, args);
                    va_end(args);
            for (int new_size = Buf.size(); old_size < new_size; old_size++)
                if (Buf[old_size] == '\n')
                    LineOffsets.push_back(old_size + 1);
        }

        void Draw(VkRender::GuiObjectHandles &pHandles) {
            // Options menu
            if (ImGui::BeginPopup("Options")) {
                ImGui::Checkbox("Auto-scroll", &AutoScroll);
                ImGui::EndPopup();
            }

            // Main window
            if (ImGui::Button("Options"))
                ImGui::OpenPopup("Options");
            ImGui::SameLine();
            bool clear = ImGui::Button("Clear");
            ImGui::SameLine();
            bool copy = ImGui::Button("Copy");
            ImGui::SameLine();
            Filter.Draw("Filter", 300.0f);

            ImGui::Separator();
            if (ImGui::BeginChild("scrolling", ImVec2(pHandles.info->debuggerWidth - pHandles.info->metricsWidth, 0),
                                  false,
                                  ImGuiWindowFlags_HorizontalScrollbar)) {
                if (clear)
                    Clear();
                if (copy)
                    ImGui::LogToClipboard();

                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
                const char *buf = Buf.begin();
                const char *buf_end = Buf.end();
                if (Filter.IsActive()) {
                    // In this example we don't use the clipper when Filter is enabled.
                    // This is because we don't have random access to the result of our filter.
                    // A real application processing logs with ten of thousands of entries may want to store the result of
                    // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
                    for (int line_no = 0; line_no < LineOffsets.Size; line_no++) {
                        const char *line_start = buf + LineOffsets[line_no];
                        const char *line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1)
                                                                                : buf_end;
                        if (Filter.PassFilter(line_start, line_end))
                            ImGui::TextUnformatted(line_start, line_end);
                    }
                } else {
                    // The simplest and easy way to display the entire buffer:
                    //   ImGui::TextUnformatted(buf_begin, buf_end);
                    // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward
                    // to skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
                    // within the visible area.
                    // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them
                    // on your side is recommended. Using ImGuiListClipper requires
                    // - A) random access into your data
                    // - B) items all being the  same height,
                    // both of which we can handle since we have an array pointing to the beginning of each line of text.
                    // When using the filter (in the block of code above) we don't have random access into the data to display
                    // anymore, which is why we don't use the clipper. Storing or skimming through the search result would make
                    // it possible (and would be recommended if you want to search through tens of thousands of entries).
                    ImGuiListClipper clipper;
                    clipper.Begin(LineOffsets.Size);
                    while (clipper.Step()) {
                        for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++) {
                            const char *line_start = buf + LineOffsets[line_no];
                            const char *line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] -
                                                                                       1) : buf_end;
                            ImGui::TextUnformatted(line_start, line_end);
                        }
                    }
                    clipper.End();
                }
                ImGui::PopStyleVar();

                if (AutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                    ImGui::SetScrollHereY(1.0f);
            }
            ImGui::EndChild();
        }
    };

/** Called once upon this object creation**/
    void onAttach() override {

    }

/** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    ExampleAppLog window;
    std::future<void> sendUserLogFuture;

/** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles &handles) override {
        if (!handles.showDebugWindow)
            return;
        VkRender::RendererConfig &config = VkRender::RendererConfig::getInstance();
        auto user = config.getUserSetting();
        bool update = false;

        static bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        ImGui::SetNextWindowSize(ImVec2(handles.info->debuggerWidth, handles.info->debuggerHeight),
                                 ImGuiCond_FirstUseEver);
        ImGui::Begin("Debugger Window", &pOpen, window_flags);

        // Make window close on X click. But also close/open on button press
        handles.showDebugWindow = pOpen;
        if (!pOpen)
            pOpen = true;

        window.Draw(handles);
        handles.info->debuggerWidth = ImGui::GetWindowWidth();
        handles.info->debuggerHeight = ImGui::GetWindowHeight();

        auto* log = Log::Logger::getConsoleLogQueue();

        while (!log->empty()) {
            window.AddLog("%s\n", log->front().c_str());
            log->pop();
        }

        ImGui::SameLine();

        ImGui::BeginChild("InfoChild",  ImVec2(0, 0), false, ImGuiWindowFlags_NoDecoration);
        // Update frame time display
        if (handles.info->firstFrame) {
            std::rotate(handles.info->frameTimes.begin(), handles.info->frameTimes.begin() + 1,
                        handles.info->frameTimes.end());
            float frameTime = 1000.0f / (handles.info->frameTimer * 1000.0f);
            handles.info->frameTimes.back() = frameTime;
            if (frameTime < handles.info->frameTimeMin) {
                handles.info->frameTimeMin = frameTime;
            }
            if (frameTime > handles.info->frameTimeMax) {
                handles.info->frameTimeMax = frameTime;
            }
        }

        ImGui::Dummy(ImVec2(5.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PlotLines("##FrameTimes", &handles.info->frameTimes[0], 50, 0, nullptr,
                         handles.info->frameTimeMin,
                         handles.info->frameTimeMax, ImVec2(handles.info->sidebarWidth - 28.0f, 80.0f));
        ImGui::Dummy(ImVec2(5.0f, 0.0f));
        ImGui::Text("Frame time: %.5f", static_cast<double>( handles.info->frameTimer));
        ImGui::Text("Frame: %lu", handles.info->frameID);
        ImGui::Separator();


        ImGui::PushFont(handles.info->font15);
        ImGui::Text("Application Options:");
        ImGui::PopFont();

        if (ImGui::Checkbox("Send Logs on exit", &user.sendUsageLogOnExit)) {
            update = true;
            handles.usageMonitor->setSetting("send_usage_log_on_exit", Utils::boolToString(user.sendUsageLogOnExit));
            handles.usageMonitor->userClickAction("Send Logs on exit", "Checkbox",
                                                   ImGui::GetCurrentWindow()->Name);
        }


#ifdef VKRENDER_MULTISENSE_VIEWER_DEBUG
        static bool showDemo = false;
        ImGui::Checkbox("ShowDemo", &showDemo);
        if (showDemo)
            ImGui::ShowDemoWindow();
#endif

        static bool sendUserLog = false;
        sendUserLog = ImGui::Button("Send user log");
        if (sendUserLog) {
            sendUserLogFuture = std::async(std::launch::async, &DebugWindow::sendUsageLog, this, &handles);
            handles.usageMonitor->userClickAction("Send user log", "Button", ImGui::GetCurrentWindow()->Name);
        }

        if (ImGui::Button("Reset consent")) {
            handles.usageMonitor->setSetting("ask_user_consent_to_collect_statistics", "true");
            handles.usageMonitor->userClickAction("Reset statistics consent", "Button",
                                                   ImGui::GetCurrentWindow()->Name);
            user.askForUsageLoggingPermissions = true;
            update = true;
        }
        if (!user.userConsentToSendLogs)
            user.sendUsageLogOnExit = false;

        // Set log level
        const char *items[] = {"LOG_TRACE", "LOG_INFO"};
        static int itemIdIndex = 1; // Here we store our selection data as an index.
        if (config.getLogLevel() == Log::LOG_LEVEL::LOG_LEVEL_TRACE) {
            itemIdIndex = 0;
        } else
            itemIdIndex = 1;

        const char *previewValue = items[itemIdIndex];  // Pass in the preview value visible before opening the combo (it could be anything)
        ImGui::SetNextItemWidth(100.0f);
        if (ImGui::BeginCombo("Set Log level", previewValue, 0)) {
            for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                const bool is_selected = (itemIdIndex == n);
                if (ImGui::Selectable(items[n], is_selected)) {
                    itemIdIndex = n;
                    auto level = Utils::getLogLevelEnumFromString(items[n]);
                    user.logLevel = level;
                    handles.usageMonitor->setSetting("log_level", items[n]);
                    update |= true;
                    handles.usageMonitor->userClickAction("Set Log level", "combo",
                                                           ImGui::GetCurrentWindow()->Name);

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::Text("Anonymous ID: %s", VkRender::RendererConfig::getInstance().getAnonymousIdentifier().c_str());

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("About: ");
        ImGui::Text("Icons from https://icons8.com");
        ImGui::EndChild();

        ImGui::End();
        if (update)
            config.setUserSetting(user);

    }

/** Called once upon this object destruction **/
    void onDetach()

    override {

    }

    void sendUsageLog(VkRender::GuiObjectHandles *handles) {
        if (handles)
            handles->usageMonitor->sendUsageLog();
    }


};

#endif //MULTISENSE_VIEWER_DEBUGWINDOW_H
