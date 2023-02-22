/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/DebugWindow.h
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

#include "Viewer/ImGui/Layer.h"

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

        void Draw(VkRender::GuiObjectHandles *pHandles) {
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
            if (ImGui::BeginChild("scrolling", ImVec2(pHandles->info->debuggerWidth - pHandles->info->metricsWidth, 0),
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

/** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        if (!handles->showDebugWindow)
            return;
        static bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        ImGui::SetNextWindowSize(ImVec2(handles->info->debuggerWidth, handles->info->debuggerHeight),
                                 ImGuiCond_FirstUseEver);
        ImGui::Begin("Debugger Window", &pOpen, window_flags);

        // Make window close on X click. But also close/open on button press
        handles->showDebugWindow = pOpen;
        if (!pOpen)
            pOpen = true;

        window.Draw(handles);
        handles->info->debuggerWidth = ImGui::GetWindowWidth();
        handles->info->debuggerHeight = ImGui::GetWindowHeight();

        auto &log = Log::Logger::getLogMetrics()->logQueue;

        while (!log.empty()) {
            window.AddLog("%s\n", log.front().c_str());
            log.pop();
        }

        ImGui::SameLine();
        if (ImGui::BeginChild("Metrics", ImVec2(handles->info->metricsWidth, 0.0f), false)) {

            {
                // Renderer Info
                ImGui::Text("Renderer Info");
                ImGui::Dummy(ImVec2(5.0f, 5.0f));

                ImVec2 txtSize = ImGui::CalcTextSize(handles->info->title.c_str());
                float xOffset = (handles->info->sidebarWidth / 2) - (txtSize.x / 2);
                ImGui::Dummy(ImVec2(xOffset, 0.0f));
                ImGui::SameLine();
                ImGui::Text("%s", handles->info->title.c_str());

                txtSize = ImGui::CalcTextSize(handles->info->deviceName.c_str());
                // If it is too long then just remove some word towards the end.
                if (txtSize.x > handles->info->sidebarWidth) {
                    std::string devName = handles->info->deviceName;
                    while (txtSize.x > handles->info->sidebarWidth) {
                        devName.erase(devName.find_last_of(' '), devName.length());
                        txtSize = ImGui::CalcTextSize(devName.c_str());
                    }
                    xOffset = (handles->info->sidebarWidth / 2) - (txtSize.x / 2);
                    ImGui::Dummy(ImVec2(xOffset, 0.0f));
                    ImGui::SameLine();
                    ImGui::Text("%s", devName.c_str());
                } else {
                    xOffset = (handles->info->sidebarWidth / 2) - (txtSize.x / 2);
                    ImGui::Dummy(ImVec2(xOffset, 0.0f));
                    ImGui::SameLine();
                    ImGui::Text("%s", handles->info->deviceName.c_str());
                }


                // Update frame time display
                if (handles->info->firstFrame) {
                    std::rotate(handles->info->frameTimes.begin(), handles->info->frameTimes.begin() + 1,
                                handles->info->frameTimes.end());
                    float frameTime = 1000.0f / (handles->info->frameTimer * 1000.0f);
                    handles->info->frameTimes.back() = frameTime;
                    if (frameTime < handles->info->frameTimeMin) {
                        handles->info->frameTimeMin = frameTime;
                    }
                    if (frameTime > handles->info->frameTimeMax) {
                        handles->info->frameTimeMax = frameTime;
                    }
                }

                ImGui::Dummy(ImVec2(5.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PlotLines("##FrameTimes", &handles->info->frameTimes[0], 50, 0, nullptr,
                                 handles->info->frameTimeMin,
                                 handles->info->frameTimeMax, ImVec2(handles->info->sidebarWidth - 28.0f, 80.0f));

                ImGui::Dummy(ImVec2(5.0f, 0.0f));

                ImGui::Text("Frame time: %.5f", handles->info->frameTimer);
                ImGui::Text("Frame: %lu", handles->info->frameID);

            }
            ImGui::Separator();

            auto met = Log::Logger::getLogMetrics();

            if (met->device.dev != nullptr) {
                const VkRender::ChannelInfo *info;
                try {
                    info = &met->device.dev->channelInfo.at(0);
                } catch (...) {
                    ImGui::Separator();
                    ImGui::EndChild();
                    ImGui::End();
                    return;
                }
                std::stringstream stream;
                std::string res;

                ImGui::Text("Device: %s", met->device.dev->cameraName.c_str());
                ImGui::Text("API build date: %s", met->device.info.apiBuildDate.c_str());

                ImGui::Text("API version: 0x%s", fmt::format("{:x}", met->device.info.apiVersion).c_str());
                ImGui::Text("Firmware build date: %s", met->device.info.firmwareBuildDate.c_str());
                stream << std::hex << met->device.info.firmwareVersion;
                ImGui::Text("Firmware version: 0x%s", fmt::format("{:x}", met->device.info.firmwareVersion).c_str());
                stream << std::hex << met->device.info.hardwareVersion;
                ImGui::Text("Hardware version: 0x%s", fmt::format("{:x}", met->device.info.hardwareVersion).c_str());
                stream << std::hex << met->device.info.hardwareMagic;
                ImGui::Text("Hardware magic: 0x%s", fmt::format("{:x}", met->device.info.hardwareMagic).c_str());
                stream << std::hex << met->device.info.sensorFpgaDna;
                ImGui::Text("FPGA DNA: 0x%s", fmt::format("{:x}", met->device.info.sensorFpgaDna).c_str());
                ImGui::Separator();

                ImGui::Dummy(ImVec2(5.0f, 5.0f));
                ImGui::Dummy(ImVec2(2.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("Application Enabled Sources:");
                for (const auto &enabled: info->enabledStreams) {
                    ImGui::Dummy(ImVec2(10.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLCoolGray);
                    ImGui::Text("%s", enabled.c_str());
                    ImGui::PopStyleColor();
                }
                ImGui::Dummy(ImVec2(2.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("UI Requested Sources:");
                ImVec2 posMax = ImGui::GetItemRectMax();
                for (const auto &req: info->requestedStreams) {

                    ImGui::Dummy(ImVec2(10.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLCoolGray);
                    ImGui::Text("%s", req.c_str());
                    ImGui::PopStyleColor();

                    ImVec2 posMaxTmp = ImGui::GetItemRectMax();
                    if (posMaxTmp.x > posMax.x)
                        posMax = posMaxTmp;
                }


                ImGui::Separator();
                for (const auto &id: met->device.sourceReceiveMapCounter) {
                    for (const auto &src: id.second) {
                        ImGui::Text("%hd :", id.first);
                        ImGui::SameLine();
                        ImGui::Text("%s ", src.first.c_str());
                        ImGui::SameLine();
                        ImGui::Text("%d", src.second);
                    }
                }
                ImGui::Text("Uptime: %.2f", met->device.upTime);
            }
            ImGui::Separator();
            ImGui::Text("Camera: ");

            ImGui::Text("Position: (%f, %f, %f)", met->camera.pos.x, met->camera.pos.y, met->camera.pos.z);
            ImGui::Text("Yaw: %f, Pitch: %f", met->camera.yaw, met->camera.pitch);
            ImGui::Text("Front: (%f, %f, %f)", met->camera.cameraFront.x, met->camera.cameraFront.y,
                        met->camera.cameraFront.z);

            ImGui::Checkbox("IgnoreMissingStatusUpdate", &met->device.ignoreMissingStatusUpdate);
            //ImGui::Checkbox("Display cursor info", &dev.pixelInfoEnable);

            static bool showDemo = false;
            ImGui::Checkbox("ShowDemo", &showDemo);
            if (showDemo)
                ImGui::ShowDemoWindow();

            static bool addTestDevice = false;
            addTestDevice = ImGui::Button("Add test device");
            if (addTestDevice){
                // Add test device to renderer if not present
                bool exists = false;
                for (const auto& device : handles->devices){
                    if (device.cameraName == "Test Device")
                        exists = true;
                }
                if (!exists){
                    VkRender::Device testDevice;
                    testDevice.state = CRL_STATE_ACTIVE;
                    testDevice.cameraName = "Test Device";
                    testDevice.notRealDevice = true;
                    Utils::initializeUIDataBlockWithTestData(testDevice);
                    handles->devices.emplace_back(testDevice);
                }
            }

            ImGui::Text("About: ");
            ImGui::Text("Icons from https://icons8.com");

        }
        ImGui::EndChild();
        //ImGui::ShowDemoWindow();
        ImGui::End();

    }

/** Called once upon this object destruction **/
    void onDetach() override {

    }


};

#endif //MULTISENSE_VIEWER_DEBUGWINDOW_H
