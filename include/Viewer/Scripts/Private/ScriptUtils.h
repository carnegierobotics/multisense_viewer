//
// Created by magnus on 4/1/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPTUTILS_H
#define MULTISENSE_VIEWER_SCRIPTUTILS_H

namespace VkRender {
    namespace ScriptUtils {
        static inline handleZoom(VkRender::ZoomParameters* zoom) {
        float newWidth = (960 / zoom->zoomValue);
        float changeInWidth = zoom->prevWidth - newWidth;
        float fullChangeInWidth = 960.0f - newWidth;
        float minRange = zoom->newMinF;
        float maxRange = 960 - zoom->newMaxF;
        float zoomedXMapped = (zoom->zoomCenter.x - 0) * (maxRange - minRange) / (960 - 0) + minRange;

        float tx = ((zoomedXMapped) / 960.0f);
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

        float newHeight = (600 / zoom->zoomValue);
        float changeInHeight = zoom->prevHeight - newHeight;
        float zoomedYMapped = (zoom->zoomCenter.y - 0) * ((600 - zoom->newMaxYF) - zoom->newMinYF) / (600 - 0) + zoom->newMinYF;
        float ty = ((zoomedYMapped) / 600.0f);

        zoom->newMinY = changeInHeight * ty;
        zoom->newMaxY = (changeInHeight * (1 - ty));
        zoom->newMinYF = (600.0f - newHeight) * ty;
        zoom->newMaxYF = ((600.0f - newHeight) * (1 - ty));

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
}

#endif //MULTISENSE_VIEWER_SCRIPTUTILS_H
