//
// Created by magnus on 4/1/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPTUTILS_H
#define MULTISENSE_VIEWER_SCRIPTUTILS_H

#include "Viewer/Core/Definitions.h"

namespace VkRender::ScriptUtils {
    static inline void handleZoom(ZoomParameters *zoom) {
        float newWidth = (zoom->m_Width / zoom->zoomValue);
        float changeInWidth = zoom->prevWidth - newWidth;
        float fullChangeInWidth = zoom->m_Width - newWidth;
        float minRange = zoom->newMinF;
        float maxRange = zoom->m_Width - zoom->newMaxF;
        float zoomedXMapped = (zoom->zoomCenter.x - 0) * (maxRange - minRange) / (zoom->m_Width - 0) + minRange;

        float tx = ((zoomedXMapped) / zoom->m_Width);
        zoom->newMin = (changeInWidth * tx);
        zoom->newMax = changeInWidth * (1 - tx);
        zoom->newMinF = (fullChangeInWidth * tx);
        zoom->newMaxF = fullChangeInWidth * (1 - tx);
        zoom->offsetX = (((zoom->prevWidth - zoom->newMax) + zoom->newMin) / (1 / tx));
        zoom->offsetX /= zoom->prevWidth;
        float delta = zoom->prevOffsetX - zoom->offsetX;
        zoom->offsetX = zoom->offsetX + (delta / 1.5f);

        if (zoom->offsetX >= 1.0f)
            zoom->offsetX = 1;
        zoom->prevOffsetX = zoom->offsetX;

        float newHeight = (zoom->m_Height / zoom->zoomValue);
        float changeInHeight = zoom->prevHeight - newHeight;
        float zoomedYMapped =
                (zoom->zoomCenter.y - 0) * ((zoom->m_Height - zoom->newMaxYF) - zoom->newMinYF) / (zoom->m_Height - 0) + zoom->newMinYF;
        float ty = ((zoomedYMapped) / zoom->m_Height);

        zoom->newMinY = changeInHeight * ty;
        zoom->newMaxY = (changeInHeight * (1 - ty));
        zoom->newMinYF = (zoom->m_Height - newHeight) * ty;
        zoom->newMaxYF = ((zoom->m_Height - newHeight) * (1 - ty));

        zoom->offsetY = (((zoom->prevHeight - zoom->newMaxY) + zoom->newMinY) / (1 / ty)) / zoom->prevHeight;
        delta = zoom->prevOffsetY - zoom->offsetY;
        zoom->offsetY = zoom->offsetY + (delta / 2);
        if (zoom->offsetY >= 1.0f)
            zoom->offsetY = 1;
        zoom->prevOffsetY = zoom->offsetY;

        zoom->prevWidth = newWidth;
        zoom->prevHeight = newHeight;
    }
}

#endif //MULTISENSE_VIEWER_SCRIPTUTILS_H