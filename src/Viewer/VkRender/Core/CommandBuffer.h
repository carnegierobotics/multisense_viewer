//
// Created by magnus on 11/15/23.
//

#ifndef MULTISENSE_VIEWER_COMMANDBUFFER_H
#define MULTISENSE_VIEWER_COMMANDBUFFER_H

#include "Viewer/Application/pch.h"
#include "Viewer/Tools/Logger.h"

#include <vulkan/vulkan_core.h>

enum RenderPassType {
    RENDER_PASS_COLOR = 0x00000001,
    RENDER_PASS_DEPTH_ONLY = 0x00000002,
    RENDER_PASS_UI = 0x00000004,
    RENDER_PASS_SECOND = 0x00000008,
};

struct CommandBuffer {
    uint32_t frameIndex = 0; // Current frame index (e.g., for triple buffering)
    uint32_t activeImageIndex = 0; // Current active image index (for swapchain images)
    // Constructor
    CommandBuffer() = default;

    explicit CommandBuffer(uint32_t frameCount) {
        buffers.resize(frameCount);
    }

    // Function to get the active buffer
    VkCommandBuffer getActiveBuffer() const {
        if (frameIndex < buffers.size()) {
            return buffers[frameIndex];
        }
        throw std::runtime_error("Invalid frame index for command buffer!");
    } // Function to get the active buffer

    // Function to get the active buffer
    uint32_t getActiveFrameIndex() const {
            return frameIndex;
    } // Function to get the active buffer

    std::vector<VkCommandBuffer>& getBuffers() {
        return buffers;
    }

    // TODO destroy events during runtime if not in use anymore
    // Function to create or get an event by tag
    void createEvent(const std::string& tag, VkDevice device) {
        if (m_eventMap.find(tag) == m_eventMap.end()) {
            // Create a new event if it does not exist
            VkEventCreateInfo eventInfo = {};
            eventInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;

            VkEvent newEvent;
            VkResult result = vkCreateEvent(device, &eventInfo, nullptr, &newEvent);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create event!");
            }
            m_eventMap[tag] = newEvent; // Store the created event in the map
        }
    }

    // Function to set an event by tag
    void setEvent(const std::string& tag, VkDevice device) {
        if (m_eventMap.find(tag) != m_eventMap.end()) {
            vkSetEvent(device, m_eventMap[tag]);
        }
    }

    // Function to set all events
    void setAllEvents(VkDevice device) {
        for (auto& [tag, event] : m_eventMap) {
            vkSetEvent(device, event);
        }
    }
    // Function to reset an event by tag
    void resetEvent(const std::string& tag, VkDevice device) {
        if (m_eventMap.find(tag) != m_eventMap.end()) {
            vkDestroyEvent(device, m_eventMap[tag], nullptr);
            m_eventMap.erase(tag);
        }
    }

    // Function to check the status of an event by tag
    bool isEventSet(const std::string& tag, VkDevice device) const {
        auto it = m_eventMap.find(tag);
        if (it != m_eventMap.end()) {
            VkResult status = vkGetEventStatus(device, it->second);
            return status == VK_EVENT_SET;
        }
        return false;
    }

    std::unordered_map<std::string, VkEvent>& events(){return m_eventMap;}
private:
    std::vector<VkCommandBuffer> buffers;
    std::unordered_map<std::string, VkEvent> m_eventMap;

};


#endif //MULTISENSE_VIEWER_COMMANDBUFFER_H
