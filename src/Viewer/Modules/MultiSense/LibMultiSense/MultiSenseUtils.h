//
// Created by magnus on 9/23/24.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEUTILS_H
#define MULTISENSE_VIEWER_MULTISENSEUTILS_H

#include <MultiSense/MultiSenseChannel.hh>
#include <MultiSense/MultiSenseTypes.hh>

namespace VkRender::MultiSenseUtils {

    static const std::vector<uint32_t> ViewerAvailableLibMultiSenseSources = {
            crl::multisense::Source_Luma_Left,
            crl::multisense::Source_Luma_Rectified_Left,
            crl::multisense::Source_Disparity_Left,
            crl::multisense::Source_Luma_Right,
            crl::multisense::Source_Luma_Rectified_Right,
            crl::multisense::Source_Chroma_Rectified_Aux,
            crl::multisense::Source_Luma_Aux,
            crl::multisense::Source_Luma_Rectified_Aux,
            crl::multisense::Source_Chroma_Aux,
            crl::multisense::Source_Disparity_Cost,

    };

        static std::string dataSourceToString(crl::multisense::DataSource d) {
        switch (d) {
            case crl::multisense::Source_Raw_Left:
                return "Raw Left";
            case crl::multisense::Source_Raw_Right:
                return "Raw Right";
            case crl::multisense::Source_Luma_Left:
                return "Luma Left";
            case crl::multisense::Source_Luma_Right:
                return "Luma Right";
            case crl::multisense::Source_Luma_Rectified_Left:
                return "Luma Rectified Left";
            case crl::multisense::Source_Luma_Rectified_Right:
                return "Luma Rectified Right";
            case crl::multisense::Source_Chroma_Left:
                return "Color Left";
            case crl::multisense::Source_Chroma_Right:
                return "Source Color Right";
            case crl::multisense::Source_Disparity_Left:
                return "Disparity Left";
            case crl::multisense::Source_Disparity_Cost:
                return "Disparity Cost";
            case crl::multisense::Source_Lidar_Scan:
                return "Source Lidar Scan";
            case crl::multisense::Source_Raw_Aux:
                return "Raw Aux";
            case crl::multisense::Source_Luma_Aux:
                return "Luma Aux";
            case crl::multisense::Source_Luma_Rectified_Aux:
                return "Luma Rectified Aux";
            case crl::multisense::Source_Chroma_Aux:
                return "Color Aux";
            case crl::multisense::Source_Chroma_Rectified_Aux:
                return "Color Rectified Aux";
            case crl::multisense::Source_Disparity_Aux:
                return "Disparity Aux";
            case crl::multisense::Source_Compressed_Left:
                return "Luma Compressed Left";
            case crl::multisense::Source_Compressed_Rectified_Left:
                return "Luma Compressed Rectified Left";
            case crl::multisense::Source_Compressed_Right:
                return "Luma Compressed Right";
            case crl::multisense::Source_Compressed_Rectified_Right:
                return "Luma Compressed Rectified Reight";
            case crl::multisense::Source_Compressed_Aux:
                return "Compressed Aux";
            case crl::multisense::Source_Compressed_Rectified_Aux:
                return "Compressed Rectified Aux";
            case (crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Luma_Rectified_Aux):
                return "Color Rectified Aux";
            case (crl::multisense::Source_Chroma_Aux | crl::multisense::Source_Luma_Aux):
                return "Color Aux";
            case crl::multisense::Source_Imu:
                return "IMU";
            default:
                return "Unknown";
        }
    }

    static crl::multisense::DataSource stringToDataSource(const std::string &d) {
        if (d == "Raw Left") return crl::multisense::Source_Raw_Left;
        if (d == "Raw Right") return crl::multisense::Source_Raw_Right;
        if (d == "Luma Left") return crl::multisense::Source_Luma_Left;
        if (d == "Luma Right") return crl::multisense::Source_Luma_Right;
        if (d == "Luma Rectified Left") return crl::multisense::Source_Luma_Rectified_Left;
        if (d == "Luma Rectified Right") return crl::multisense::Source_Luma_Rectified_Right;
        if (d == "Luma Compressed Rectified Left") return crl::multisense::Source_Compressed_Rectified_Left;
        if (d == "Luma Compressed Left") return crl::multisense::Source_Compressed_Left;
        if (d == "Color Left") return crl::multisense::Source_Chroma_Left;
        if (d == "Source Color Right") return crl::multisense::Source_Chroma_Right;
        if (d == "Disparity Left") return crl::multisense::Source_Disparity_Left;
        if (d == "Disparity Cost") return crl::multisense::Source_Disparity_Cost;
        if (d == "Jpeg Left") return crl::multisense::Source_Jpeg_Left;
        if (d == "Source Rgb Left") return crl::multisense::Source_Rgb_Left;
        if (d == "Source Lidar Scan") return crl::multisense::Source_Lidar_Scan;
        if (d == "Raw Aux") return crl::multisense::Source_Raw_Aux;
        if (d == "Luma Aux") return crl::multisense::Source_Luma_Aux;
        if (d == "Luma Rectified Aux") return crl::multisense::Source_Luma_Rectified_Aux;
        if (d == "Color Aux") return crl::multisense::Source_Chroma_Aux;
        if (d == "Color Rectified Aux") return crl::multisense::Source_Chroma_Rectified_Aux;
        if (d == "Chroma Aux") return crl::multisense::Source_Chroma_Aux;
        if (d == "Chroma Rectified Aux") return crl::multisense::Source_Chroma_Rectified_Aux;
        if (d == "Disparity Aux") return crl::multisense::Source_Disparity_Aux;
        if (d == "IMU") return crl::multisense::Source_Imu;
        if (d == "All") return crl::multisense::Source_All;
        return false;
    }

    // Static utility function to convert hardware revision to a readable string
    static std::string hardwareRevisionToString(uint32_t hardwareRevision) {
        switch (hardwareRevision) {
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_SL:
                return "MultiSense SL";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S7:
                return "MultiSense S7/S";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_M:
                return "MultiSense M";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S7S:
                return "MultiSense S7S";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S21:
                return "MultiSense S21";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_ST21:
                return "MultiSense ST21";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27:
                return "MultiSense C6S2-S27";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30:
                return "MultiSense S30";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S7AR:
                return "MultiSense S7AR";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21:
                return "MultiSense KS21";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM:
                return "MultiSense MonoCam";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_VPB:
                return "Remote Head VPB";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_STEREO:
                return "Remote Head Stereo";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_MONOCAM:
                return "Remote Head MonoCam";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21_SILVER:
                return "MultiSense KS21 Silver";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_ST25:
                return "MultiSense ST25";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_BCAM:
                return "Bcam";
            case crl::multisense::system::DeviceInfo::HARDWARE_REV_MONO:
                return "Mono";
            default:
                return "Unknown Hardware Revision";
        }
    }
}
#endif //MULTISENSE_VIEWER_MULTISENSEUTILS_H
