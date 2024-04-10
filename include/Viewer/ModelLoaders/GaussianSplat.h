//
// Created by mgjer on 10/01/2024.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANSPLAT_H
#define MULTISENSE_VIEWER_GAUSSIANSPLAT_H

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Viewer/Tools/helper_cuda.h>

#include "Viewer/Core/Texture.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Tools/Logger.h"

/*
namespace CUDARenderer {
    struct RasterSettings {

        RasterSettings()= default;
        uint32_t height = 0;
        uint32_t width = 0;
        float tanFovX = 0;
        float tanFovY = 0;
        torch::Tensor bg = torch::empty({0});
        float scaleModifier = 0;
        torch::Tensor viewMat = torch::empty({0});
        torch::Tensor projectionMat = torch::empty({0});
        float shDegree = 3;
        torch::Tensor cameraPosition = torch::empty({0});
        bool prefilter = false;
        bool debug = false;
    };

    struct GaussianData {
        torch::Tensor xyz;
        torch::Tensor rot;
        torch::Tensor scale;
        torch::Tensor opacity;
        torch::Tensor sh;

    };

};
*/
class GaussianSplat {
public:
    GaussianSplat(VulkanDevice *_device) : device(_device) {}

    VulkanDevice *device;
    // CUDA objects
    cudaExternalMemory_t cudaExtMemImageBuffer;
    cudaMipmappedArray_t cudaMipmappedImageArray, cudaMipmappedImageArrayTemp,
            cudaMipmappedImageArrayOrig;
    std::vector<cudaSurfaceObject_t> surfaceObjectList, surfaceObjectListTemp;
    cudaSurfaceObject_t *d_surfaceObjectList, *d_surfaceObjectListTemp;
    cudaTextureObject_t textureObjMipMapInput;

    cudaExternalSemaphore_t cudaExtCudaUpdateVkSemaphore;
    cudaExternalSemaphore_t cudaExtVkUpdateCudaSemaphore;
    /*
// Bind texture to be used by the rasterizer
    CUDARenderer::GaussianData gaussianData;
    CUDARenderer::RasterSettings rasterSettings;

    void cudaFromCpu(){
        torch::Tensor cuda_xyz = gaussianData.xyz.to(torch::kFloat).to(torch::kCUDA).set_requires_grad(false);
    }
     */

    int setCudaVkDevice(uint8_t *vkDeviceUUID) {
        int current_device = 0;
        int device_count = 0;
        int devices_prohibited = 0;

        cudaDeviceProp deviceProp{};
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
            exit(EXIT_FAILURE);
        }

        // Find the GPU which is selected by Vulkan
        while (current_device < device_count) {
            cudaGetDeviceProperties(&deviceProp, current_device);
            auto *ptr = reinterpret_cast<unsigned char *>(&(deviceProp.uuid.bytes)); // Get the address of the first element
            auto *ptr2 = reinterpret_cast<unsigned char *>(vkDeviceUUID); // Get the address of the first element

            for (int i = 0; i < VK_UUID_SIZE; ++i) {
                Log::Logger::getInstance()->info("ptr1:ptr2 --> {} : {}", ptr[i], ptr2[i]);
            }
            if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
                // Compare the cuda device UUID with vulkan UUID
                int ret = memcmp(&ptr, &ptr2, VK_UUID_SIZE);

                checkCudaErrors(cudaSetDevice(current_device));
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
                Log::Logger::getInstance()->info("GPU Device {}: \"{}\" with compute capability {}.{}\n\n",
                                                 current_device, deviceProp.name, deviceProp.major,
                                                 deviceProp.minor);

                return current_device;


            } else {
                devices_prohibited++;
            }

            current_device++;
        }

        if (devices_prohibited == device_count) {
            fprintf(stderr,
                    "CUDA error:"
                    " No Vulkan-CUDA Interop capable GPU found.\n");
            exit(EXIT_FAILURE);
        }

        return -1;
    }

    void createCudaVkImage(uint32_t width, uint32_t height, Texture2D *cudaTexture) {
        // Create optimal tiled target m_Image
        VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.extent = {width, height, 1};
        imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;;
        // Ensure that the TRANSFER_DST bit is set for staging
        if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }

        VkExternalMemoryImageCreateInfoKHR extImageCreateInfo = {};
        /*
         * Indicate that the memory backing this image will be exported in an
         * fd. In some implementations, this may affect the call to
         * GetImageMemoryRequirements() with this image.
         */
        extImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;
        extImageCreateInfo.handleTypes |= VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        imageCreateInfo.pNext = &extImageCreateInfo;

        CHECK_RESULT(
                vkCreateImage(device->m_LogicalDevice, &imageCreateInfo, nullptr, &cudaTexture->m_Image))

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device->m_LogicalDevice, cudaTexture->m_Image, &memReqs);
        VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkExportMemoryAllocateInfoKHR exportInfo = {};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        memAlloc.pNext = &exportInfo;

        CHECK_RESULT(
                vkAllocateMemory(device->m_LogicalDevice, &memAlloc, nullptr, &cudaTexture->m_DeviceMemory))

        CHECK_RESULT(vkBindImageMemory(device->m_LogicalDevice, cudaTexture->m_Image,
                                       cudaTexture->m_DeviceMemory, 0))

    }

    /*

    void cudaVkImportSemaphore() {
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};
        memset(&externalSemaphoreHandleDesc, 0,
               sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;

        externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
#else
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, cudaUpdateVkSemaphore);
#endif
        externalSemaphoreHandleDesc.flags = 0;

        checkCudaErrors(cudaImportExternalSemaphore(&cudaExtCudaUpdateVkSemaphore,
                                                    &externalSemaphoreHandleDesc));

        memset(&externalSemaphoreHandleDesc, 0,
               sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
        externalSemaphoreHandleDesc.type =
                IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
                                      : cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;;
        externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
                IsWindows8OrGreater()
                ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
                vkUpdateCudaSemaphore);
#else
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, vkUpdateCudaSemaphore);
#endif
        externalSemaphoreHandleDesc.flags = 0;
        checkCudaErrors(cudaImportExternalSemaphore(&cudaExtVkUpdateCudaSemaphore,
                                                    &externalSemaphoreHandleDesc));
        printf("CUDA Imported Vulkan semaphore\n");
    }

    void cudaVkImportImageMem() {
        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
        memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
        cudaExtMemHandleDesc.type =
                IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
                                      : cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
        cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(
                IsWindows8OrGreater()
                ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
        cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

    cudaExtMemHandleDesc.handle.fd =
        getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
        cudaExtMemHandleDesc.size = totalImageMemSize;

        checkCudaErrors(cudaImportExternalMemory(&cudaExtMemImageBuffer,
                                                 &cudaExtMemHandleDesc));

        cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;

        memset(&externalMemoryMipmappedArrayDesc, 0,
               sizeof(externalMemoryMipmappedArrayDesc));

        cudaExtent extent = make_cudaExtent(imageWidth, imageHeight, 0);
        cudaChannelFormatDesc formatDesc;
        formatDesc.x = 8;
        formatDesc.y = 8;
        formatDesc.z = 8;
        formatDesc.w = 8;
        formatDesc.f = cudaChannelFormatKindUnsigned;

        externalMemoryMipmappedArrayDesc.offset = 0;
        externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
        externalMemoryMipmappedArrayDesc.extent = extent;
        externalMemoryMipmappedArrayDesc.flags = 0;
        externalMemoryMipmappedArrayDesc.numLevels = mipLevels;

        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
                &cudaMipmappedImageArray, cudaExtMemImageBuffer,
                &externalMemoryMipmappedArrayDesc));

        checkCudaErrors(cudaMallocMipmappedArray(&cudaMipmappedImageArrayTemp,
                                                 &formatDesc, extent, mipLevels));
        checkCudaErrors(cudaMallocMipmappedArray(&cudaMipmappedImageArrayOrig,
                                                 &formatDesc, extent, mipLevels));

        for (int mipLevelIdx = 0; mipLevelIdx < mipLevels; mipLevelIdx++) {
            cudaArray_t cudaMipLevelArray, cudaMipLevelArrayTemp,
                    cudaMipLevelArrayOrig;
            cudaResourceDesc resourceDesc;

            checkCudaErrors(cudaGetMipmappedArrayLevel(
                    &cudaMipLevelArray, cudaMipmappedImageArray, mipLevelIdx));
            checkCudaErrors(cudaGetMipmappedArrayLevel(
                    &cudaMipLevelArrayTemp, cudaMipmappedImageArrayTemp, mipLevelIdx));
            checkCudaErrors(cudaGetMipmappedArrayLevel(
                    &cudaMipLevelArrayOrig, cudaMipmappedImageArrayOrig, mipLevelIdx));

            uint32_t width =
                    (imageWidth >> mipLevelIdx) ? (imageWidth >> mipLevelIdx) : 1;
            uint32_t height =
                    (imageHeight >> mipLevelIdx) ? (imageHeight >> mipLevelIdx) : 1;
            checkCudaErrors(cudaMemcpy2DArrayToArray(
                    cudaMipLevelArrayOrig, 0, 0, cudaMipLevelArray, 0, 0,
                    width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice));

            memset(&resourceDesc, 0, sizeof(resourceDesc));
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = cudaMipLevelArray;

            cudaSurfaceObject_t surfaceObject;
            checkCudaErrors(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));

            surfaceObjectList.push_back(surfaceObject);

            memset(&resourceDesc, 0, sizeof(resourceDesc));
            resourceDesc.resType = cudaResourceTypeArray;
            resourceDesc.res.array.array = cudaMipLevelArrayTemp;

            cudaSurfaceObject_t surfaceObjectTemp;
            checkCudaErrors(
                    cudaCreateSurfaceObject(&surfaceObjectTemp, &resourceDesc));
            surfaceObjectListTemp.push_back(surfaceObjectTemp);
        }

        cudaResourceDesc resDescr;
        memset(&resDescr, 0, sizeof(cudaResourceDesc));

        resDescr.resType = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = cudaMipmappedImageArrayOrig;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;

        texDescr.maxMipmapLevelClamp = float(mipLevels - 1);

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&textureObjMipMapInput, &resDescr,
                                                &texDescr, NULL));

        checkCudaErrors(cudaMalloc((void **) &d_surfaceObjectList,
                                   sizeof(cudaSurfaceObject_t) * mipLevels));
        checkCudaErrors(cudaMalloc((void **) &d_surfaceObjectListTemp,
                                   sizeof(cudaSurfaceObject_t) * mipLevels));

        checkCudaErrors(cudaMemcpy(d_surfaceObjectList, surfaceObjectList.data(),
                                   sizeof(cudaSurfaceObject_t) * mipLevels,
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
                d_surfaceObjectListTemp, surfaceObjectListTemp.data(),
                sizeof(cudaSurfaceObject_t) * mipLevels, cudaMemcpyHostToDevice));

        printf("CUDA Kernel Vulkan image buffer\n");
    }
     */

    void cudaUpdateVkImage() {
        /*
        cudaVkSemaphoreWait(cudaExtVkUpdateCudaSemaphore);

        int nthreads = 128;

        //Perform 2D box filter on image using CUDA
        d_boxfilter_rgba_x<<<imageHeight / nthreads, nthreads, 0, streamToRun>>>(
                d_surfaceObjectListTemp, textureObjMipMapInput, imageWidth, imageHeight,
                        mipLevels, filter_radius);

        d_boxfilter_rgba_y<<<imageWidth / nthreads, nthreads, 0, streamToRun>>>(
                d_surfaceObjectList, d_surfaceObjectListTemp, imageWidth, imageHeight,
                        mipLevels, filter_radius);

        varySigma();

        cudaVkSemaphoreSignal(cudaExtCudaUpdateVkSemaphore);
        */
    }

};


#endif //MULTISENSE_VIEWER_GAUSSIANSPLAT_H
