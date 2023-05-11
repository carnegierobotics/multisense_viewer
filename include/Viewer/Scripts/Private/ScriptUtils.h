//
// Created by magnus on 4/1/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPTUTILS_H
#define MULTISENSE_VIEWER_SCRIPTUTILS_H

#include "Viewer/Core/Definitions.h"

namespace VkRender::ScriptUtils {
    static inline void handleZoom(ZoomParameters *zoom) {
        float newWidth = (zoom->m_Width / zoom->zoomValue);
        float changeInWidth = zoom->m_Width - newWidth;


        float tx = (zoom->currentZoomCenter.x + 1) / 2;
        zoom->newMinF = (changeInWidth * tx);
        zoom->newMaxF = changeInWidth * (1 - tx);

        float ty = (zoom->currentZoomCenter.y  + 1) / 2;
        float newHeight = (zoom->m_Height / zoom->zoomValue);
        zoom->newMinYF = (zoom->m_Height - newHeight) * ty;
        zoom->newMaxYF = ((zoom->m_Height - newHeight) * (1 - ty));

        zoom->offsetX = zoom->currentZoomCenter.x;
        zoom->offsetY = zoom->currentZoomCenter.y;
        zoom->currentCenterPixel.x = zoom->currentZoomCenter.x + zoom->translateX;
        /*
        float newWidth = (zoom->m_Width / zoom->zoomValue);
        float changeInWidth = zoom->prevWidth - newWidth;
        float fullChangeInWidth = zoom->m_Width - newWidth;
        float minRange = zoom->newMinF;
        float maxRange = zoom->m_Width - zoom->newMaxF;
        float zoomedXMapped = (zoom->currentZoomCenter.x - 0) * (maxRange - minRange) / (zoom->m_Width - 0) + minRange;

        float tx = ((zoomedXMapped) / zoom->m_Width);
        zoom->newMin = (changeInWidth * tx);
        zoom->newMax = changeInWidth * (1 - tx);
        zoom->newMinF = (fullChangeInWidth * tx);
        zoom->newMaxF = fullChangeInWidth * (1 - tx);
        zoom->offsetX = (((zoom->prevWidth - zoom->newMax) + zoom->newMin) / (1 / tx));
        zoom->offsetX /= zoom->prevWidth;

        if (zoom->offsetX >= 1.0f)
            zoom->offsetX = 1;

        zoom->prevOffsetX = zoom->offsetX;

        float newHeight = (zoom->m_Height / zoom->zoomValue);
        float changeInHeight = zoom->prevHeight - newHeight;
        float zoomedYMapped =
                (zoom->currentZoomCenter.y - 0) * ((zoom->m_Height - zoom->newMaxYF) - zoom->newMinYF) / (zoom->m_Height - 0) + zoom->newMinYF;
        float ty = ((zoomedYMapped) / zoom->m_Height);

        zoom->newMinY = changeInHeight * ty;
        zoom->newMaxY = (changeInHeight * (1 - ty));
        zoom->newMinYF = (zoom->m_Height - newHeight) * ty;
        zoom->newMaxYF = ((zoom->m_Height - newHeight) * (1 - ty));

        zoom->offsetY = (((zoom->prevHeight - zoom->newMaxY) + zoom->newMinY) / (1 / ty)) / zoom->prevHeight;

        if (zoom->offsetY >= 1.0f)
            zoom->offsetY = 1;
        zoom->prevOffsetY = zoom->offsetY;

        zoom->prevWidth = newWidth;
        zoom->prevHeight = newHeight;
         */
        zoom->prevWidth = newWidth;

    }
}

#endif //MULTISENSE_VIEWER_SCRIPTUTILS_H
