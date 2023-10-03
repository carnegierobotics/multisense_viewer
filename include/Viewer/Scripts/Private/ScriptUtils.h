//
// Created by magnus on 4/1/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPTUTILS_H
#define MULTISENSE_VIEWER_SCRIPTUTILS_H

#include "Viewer/Core/Definitions.h"

namespace VkRender::ScriptUtils {

    struct ZoomParameters {
        glm::vec2 zoomCenter;
        float zoomValue = 1.0f;
        float prevZoomValue = 0.0f;

        glm::vec2 targetZoomCenter = glm::vec2(0.5f, 0.5f);
        glm::vec2 currentZoomCenter = glm::vec2(0.5f, 0.5f);
        glm::vec2 currentCenterPixel = glm::vec2(0.0f, 0.0f); // NDC

        float lerpFactor = 0.5f;

        float offsetX = 0.0f;
        float prevOffsetX = 0.0f;
        float prevOffsetY = 0.0f;
        float offsetY = 0.0f;
        float prevWidth = 960.0f;
        float prevHeight = 600;
        float newMin = 0.0f, newMax = 0.0f;
        float newMinF = 0.0f, newMaxF = 0.0f;
        float newMinY = 0.0f, newMaxY = 0.0f;
        float newMinYF = 0.0f, newMaxYF = 0.0f;
        float translateX = 0.0f;
        float translateY = 0.0f;

        float m_Width = 0, m_Height = 0;

        bool resChanged = false;

        void resolutionUpdated(uint32_t width, uint32_t height) {
            m_Width = static_cast<float>(width);
            m_Height = static_cast<float>(height);
            prevWidth = static_cast<float>(width);
            prevHeight = static_cast<float>(height);
            prevOffsetX = 0.0f;
            prevOffsetY = 0.0f;
            newMin = 0.0f;
            newMax = 0.0f;
            newMinF = 0.0f;
            newMaxF = static_cast<float>(width);
            newMinY = 0.0f;
            newMaxY = 0.0f;
            newMinYF = 0.0f;
            newMaxYF = static_cast<float>(height);
            prevZoomValue = -1.0f; // trigger zoom update
        }
    };

    static inline void handleZoom(ZoomParameters *zoom) {
        float newWidth = (zoom->m_Width / zoom->zoomValue);
        float changeInWidth = zoom->m_Width - newWidth;


        float tx = (zoom->currentZoomCenter.x + 1) / 2; // map from -1, -1 to 0, 1
        zoom->newMinF = (changeInWidth * tx);
        zoom->newMaxF = changeInWidth * (1 - tx);

        float ty = (zoom->currentZoomCenter.y + 1) / 2;
        float newHeight = (zoom->m_Height / zoom->zoomValue);
        zoom->newMinYF = (zoom->m_Height - newHeight) * ty;
        zoom->newMaxYF = ((zoom->m_Height - newHeight) * (1 - ty));

        zoom->offsetX = zoom->currentZoomCenter.x;
        zoom->offsetY = zoom->currentZoomCenter.y;
        zoom->currentCenterPixel.x = zoom->currentZoomCenter.x + zoom->translateX;
        zoom->prevWidth = newWidth;

    }

    static inline void
    handleZoomUiLoop(ZoomParameters *zoom, Device &dev, StreamWindowIndex previewWindowIndex, glm::vec2 mousePos,
                     bool isClickedAndHovered, bool magnifyMode, bool enableZoom = true) {
        zoom->zoomValue = 0.9f * zoom->zoomValue * zoom->zoomValue * zoom->zoomValue + 1 -
                          0.9f; // cubic growth in scaling factor
        bool updateZoom = zoom->zoomValue != zoom->prevZoomValue;

        if ((updateZoom || magnifyMode) && enableZoom) {

            auto mappedX = static_cast<uint32_t>(
                    dev.pixelInfo[previewWindowIndex].x * (zoom->m_Width - zoom->newMaxF - zoom->newMinF) /
                    (zoom->m_Width - 0) +
                    zoom->newMinF);
            auto mappedY = static_cast<uint32_t>(
                    dev.pixelInfo[previewWindowIndex].y * ((zoom->m_Height - zoom->newMaxYF) - zoom->newMinYF) /
                    (zoom->m_Height - 0) +
                    zoom->newMinYF);

            zoom->targetZoomCenter = glm::vec2(2.0f * mappedX / zoom->m_Width - 1.0f,
                                               2.0f * mappedY / zoom->m_Height - 1.0f);
            float interpolationFactor = glm::clamp(1.0f / (zoom->zoomValue * 0.7f), 0.0f, 1.0f);
            zoom->currentZoomCenter = glm::mix(zoom->currentZoomCenter, zoom->targetZoomCenter, interpolationFactor);

            //Log::Logger::getInstance()->trace("mapX {} mapy {}", mappedX, mappedY);
        }
        if (isClickedAndHovered) {
            //Log::Logger::getInstance()->info("x, y: ({}, {}), center: ({}, {}), zoom: {}",   zoom->translateX,   zoom->translateY, zoom->currentZoomCenter.x, zoom->currentZoomCenter.y, zoom->zoomValue);
#ifdef WIN32
            float translateFactor = 1000.0f;
#else
            float translateFactor = 75.0f;
#endif

            float x = (mousePos.x / (translateFactor * (zoom->zoomValue / 2.0f)));
            float y = (mousePos.y /
                       (translateFactor * (zoom->zoomValue / 3.2f))); // Update x and y speed w.r.t aspect ratio

            if (zoom->currentZoomCenter.x + x <= 1.0f && zoom->currentZoomCenter.x + x >= -1.0f)
                zoom->currentZoomCenter.x += x;

            if (zoom->currentZoomCenter.y + y <= 1.0f && zoom->currentZoomCenter.y + y >= -1.0f)
                zoom->currentZoomCenter.y += y;
        }
        if (zoom->zoomValue <= 1.10f) {
            zoom->translateX = 0;
            zoom->translateY = 0;
        }

        auto mappedX = static_cast<uint32_t>(
                dev.pixelInfo[previewWindowIndex].x * (zoom->m_Width - zoom->newMaxF - zoom->newMinF) /
                (zoom->m_Width - 0) +
                zoom->newMinF);
        auto mappedY = static_cast<uint32_t>(
                dev.pixelInfo[previewWindowIndex].y * ((zoom->m_Height - zoom->newMaxYF) - zoom->newMinYF) /
                (zoom->m_Height - 0) +
                zoom->newMinYF);
        if (mappedX <= zoom->m_Width && mappedY <= zoom->m_Height) {
            dev.pixelInfoZoomed[previewWindowIndex].x = mappedX;
            dev.pixelInfoZoomed[previewWindowIndex].y = mappedY;
        }

        zoom->prevZoomValue = zoom->zoomValue;
    }


    static inline bool
    setZoomValue(VkRender::ScriptUtils::ZoomParameters &zoom, std::unordered_map<StreamWindowIndex, float> *map,
                 StreamWindowIndex windowIndex) {
        if (map->empty()) {
            // Handle the case where uiHandle is nullptr
            return false;
        }

        auto it = map->find(windowIndex);
        if (it == map->end()) {
            // Handle the case where CRL_PREVIEW_TWO is not in the map
            return false;
        }

        zoom.zoomValue = it->second;
        return true;
    }

    /**@brief Primitive for a surface */
    struct ImageData {
        struct {
            std::vector<VkRender::Vertex> vertices{};
            std::vector<uint32_t> indices;
            uint32_t vertexCount{};
            //uint32_t *indices{};
            uint32_t indexCount{};
        } quad{};

        /**@brief Generates a Quad with texture coordinates. Arguments are offset values */
        explicit ImageData(float y = 0.0f) {
            int vertexCount = 4;
            int indexCount = 2 * 3;
            quad.vertexCount = vertexCount;
            quad.indexCount = indexCount;
            // Virtual class can generate some m_Mesh data here
            quad.vertices.resize(vertexCount);
            quad.indices.resize(indexCount);

            auto *vP = quad.vertices.data();
            auto *iP = quad.indices.data();

            VkRender::Vertex vertex[4]{};
            vertex[0].pos = glm::vec3(-1.0f, -1.0f + y, 0.0f);
            vertex[1].pos = glm::vec3(1.0f, -1.0f + y, 0.0f);
            vertex[2].pos = glm::vec3(1.0f, 1.0f + y, 0.0f);
            vertex[3].pos = glm::vec3(-1.0f, 1.0f + y, 0.0f);

            vertex[0].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[1].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[2].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[3].normal = glm::vec3(0.0f, 1.0f, 0.0f);

            vertex[0].uv0 = glm::vec2(0.0f, 0.0f + y);
            vertex[1].uv0 = glm::vec2(1.0f, 0.0f + y);
            vertex[2].uv0 = glm::vec2(1.0f, 1.0f + y);
            vertex[3].uv0 = glm::vec2(0.0f, 1.0f + y);

            vP[0] = vertex[0];
            vP[1] = vertex[1];
            vP[2] = vertex[2];
            vP[3] = vertex[3];
            // indices
            iP[0] = 0;
            iP[1] = 1;
            iP[2] = 2;
            iP[3] = 2;
            iP[4] = 3;
            iP[5] = 0;
        }
    };
}

#endif //MULTISENSE_VIEWER_SCRIPTUTILS_H
