//
// Created by mgjer on 15/01/2024.
//

#define NOMINMAX
#define GLFW_INCLUDE_VULKAN
#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#define _USE_MATH_DEFINES
#endif

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#ifdef _WIN64
#include <vulkan/vulkan_win32.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <Viewer/Tools/helper_cuda.h>

#include <rasterize_points.h>
#include <opencv2/opencv.hpp>
#include <tinyply.h>

#include <Viewer/Tools/helper_cuda.h>
#include "Viewer/Cuda/Cuda_example.h"

#include <glm/trigonometric.hpp>

#include "Viewer/Core/Texture.h"
#include "Viewer/Tools/Logger.h"

void printTensor(const torch::Tensor& tensor, int64_t numRows = 1) {
    std::cout << "Shape/Size: " << tensor.sizes() << std::endl;
    std::cout << "Data Type: " << tensor.dtype() << std::endl;
    std::cout << "Device: " << tensor.device() << std::endl;
    if (numRows > 0) {
        int rowsToPrint = std::min(numRows, tensor.size(0));
        for (int i = 0; i < rowsToPrint; ++i) {
            // Print each row. Assuming tensor is 2D.
            for (int j = 0; j < tensor.size(1); ++j) {
                std::cout << tensor[i][j].item<float>() << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

inline std::vector<uint8_t> read_file_binary(const std::string& pathToFile) {
    std::ifstream file(pathToFile, std::ios::binary);
    std::vector<uint8_t> fileBufferBytes;

    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t sizeBytes = file.tellg();
        file.seekg(0, std::ios::beg);
        fileBufferBytes.resize(sizeBytes);
        if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
    }
    else throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
    return fileBufferBytes;
}

struct memory_buffer : public std::streambuf {
    char* p_start{nullptr};
    char* p_end{nullptr};
    size_t size;

    memory_buffer(char const* first_elem, size_t size)
        : p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size) {
        setg(p_start, p_start, p_end);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if (dir == std::ios_base::cur) gbump(static_cast<int>(off));
        else setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
        return gptr() - p_start;
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

struct memory_stream : virtual memory_buffer, public std::istream {
    memory_stream(char const* first_elem, size_t size)
        : memory_buffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this)) {
    }
};


// GaussianData class definition
class GaussianData {
public:
    torch::Tensor xyz;
    torch::Tensor rot;
    torch::Tensor scale;
    torch::Tensor opacity;
    torch::Tensor sh;

    torch::Tensor flat() const {
        return torch::cat({xyz, rot, scale, opacity, sh}, -1).contiguous();
    }

    int64_t length() const {
        return xyz.size(0);
    }

    int sh_dim() const {
        return sh.size(-1);
    }
};

torch::Tensor ConvertGlmMat4ToTensor(const glm::mat4& mat) {
    // Create a 4x4 tensor
    torch::Tensor tensor = torch::empty({4, 4}, torch::kFloat32);

    // Copy data from glm::mat4 to the tensor
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            tensor[i][j] = mat[i][j];
        }
    }
    return tensor;
}

torch::Tensor ConvertGlmVec3ToTensor(const glm::vec3& vec) {
    // Create a 1D tensor with 3 elements
    torch::Tensor tensor = torch::empty({3}, torch::kFloat32);

    // Copy data from glm::vec3 to the tensor
    tensor[0] = vec.x;
    tensor[1] = vec.y;
    tensor[2] = vec.z;

    return tensor;
}

GaussianData naive_gaussian() {
    auto gau_xyz = torch::tensor({
                                     0, 0, 0,
                                     1, 0, 0,
                                     0, 1, 0,
                                     0, 0, 1
                                 },
                                 torch::dtype(torch::kFloat32)).view({-1, 3});

    auto gau_rot = torch::tensor({
                                     1, 0, 0, 0,
                                     1, 0, 0, 0,
                                     1, 0, 0, 0,
                                     1, 0, 0, 0
                                 },
                                 torch::dtype(torch::kFloat32)).view({-1, 4});
    auto gau_s = torch::tensor({
                                   0.1, 0.1, 0.1,
                                   0.2, 0.03, 0.03,
                                   0.03, 0.2, 0.03,
                                   0.03, 0.03, 0.2
                               },
                               torch::dtype(torch::kFloat32)).view({-1, 3});
    auto gau_c = torch::tensor({
                                   1, 0, 1,
                                   1, 0, 0,
                                   0, 1, 0,
                                   0, 0, 1,
                               },
                               torch::dtype(torch::kFloat32)).view({-1, 3});
    gau_c = (gau_c - 0.5) / 0.28209;

    auto gau_a = torch::tensor({1, 1, 1, 1}, torch::dtype(torch::kFloat32)).view({-1, 1});

    return GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c);
}

GaussianData loadTinyPly(std::filesystem::path filePath, bool preloadIntoMemory = true) {
    std::cout << "........................................................................\n";
    std::cout << "Now Reading: " << filePath << std::endl;

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;
    torch::Tensor xyzTensor, rotationsTensor, scalesTensor, opacitiesTensor, shsTensor, reshapedTensor;
    try {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a
        // stream is a net win for parsing speed, about 40% faster.
        if (preloadIntoMemory) {
            byte_buffer = read_file_binary(filePath.string());
            file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        }

        if (!file_stream || file_stream->fail())
            throw std::runtime_error(
                "file_stream failed to open " + filePath.string());

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);

        std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        for (const auto& c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        for (const auto& c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

        for (const auto& e : file.get_elements()) {
            std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
            for (const auto& p : e.properties) {
                std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.
                    propertyType].str << ")";
                if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
                std::cout << std::endl;
            }
        }

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers.
        // See examples below on how to marry your own application-specific data structures with this one.
        std::shared_ptr<tinyply::PlyData> vertices, normals, f_dc, f_rest, opacities, scales, rotations;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties
        // like vertex position are hard-coded:
        try { vertices = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"}); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { f_dc = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        std::vector<std::string> propertyKeys;
        for (size_t i = 0; i < 45; ++i) {
            propertyKeys.push_back("f_rest_" + std::to_string(i));;
        }
        try { f_rest = file.request_properties_from_element("vertex", propertyKeys); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { opacities = file.request_properties_from_element("vertex", {"opacity"}); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have
        // arbitrary ply files, it is best to leave this 0.
        try { scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"}); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
        // are specifically in the file, which is unlikely);
        try { rotations = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}, 0); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


        if (vertices) std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
        if (normals) std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
        if (f_dc) std::cout << "\tRead " << f_dc->count << " total vertex f_dc " << std::endl;
        if (f_rest) std::cout << "\tRead " << f_rest->count << " total vertex f_rest " << std::endl;
        if (opacities) std::cout << "\tRead " << opacities->count << " total vertex opacities " << std::endl;
        if (scales) std::cout << "\tRead " << scales->count << " total scales " << std::endl;
        if (rotations) std::cout << "\tRead " << rotations->count << " total rotations) " << std::endl;

        try {
            file.read(*file_stream);
        }
        catch (const std::exception& e) {
            std::cerr << "tinyply exception (reading data): " << e.what() << std::endl;
        }

        // Example One: converting to your own application types
        {
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            if (vertices->t != tinyply::Type::FLOAT32) {
                std::cerr << "Data type is not float32" << std::endl;
            }
            int64_t count = vertices->count;
            xyzTensor = torch::zeros({count, 3}, torch::kFloat32);
            std::memcpy(xyzTensor.data_ptr(), vertices->buffer.get(), numVerticesBytes);

            rotationsTensor = torch::zeros({static_cast<int64_t>(rotations->count), 4}, torch::kFloat32);
            std::memcpy(rotationsTensor.data_ptr(), rotations->buffer.get(), rotations->buffer.size_bytes());
            torch::Tensor norms = rotationsTensor.norm(2, -1, true);
            printTensor(rotationsTensor, 10);
            rotationsTensor = rotationsTensor / norms;
            printTensor(rotationsTensor, 10);

            scalesTensor = torch::zeros({static_cast<int64_t>(scales->count), 3}, torch::kFloat32);
            std::memcpy(scalesTensor.data_ptr(), scales->buffer.get(), scales->buffer.size_bytes());

            opacitiesTensor = torch::zeros({static_cast<int64_t>(opacities->count), 1}, torch::kFloat32);
            std::memcpy(opacitiesTensor.data_ptr(), opacities->buffer.get(), opacities->buffer.size_bytes());
            printTensor(opacitiesTensor, 5);
            opacitiesTensor = torch::sigmoid(opacitiesTensor);
            printTensor(opacitiesTensor, 5);

            torch::Tensor dcFeatures = torch::zeros({static_cast<int64_t>(f_dc->count), 3}, torch::kFloat32);
            std::memcpy(dcFeatures.data_ptr(), f_dc->buffer.get(), f_dc->buffer.size_bytes());

            int maxShDegree = 3;
            torch::Tensor extraFeatures = torch::zeros({static_cast<int64_t>(f_rest->count), 45}, torch::kFloat32);
            std::memcpy(extraFeatures.data_ptr(), f_rest->buffer.get(), f_rest->buffer.size_bytes());
            extraFeatures = extraFeatures.reshape({
                extraFeatures.size(0), 3, static_cast<int64_t>((maxShDegree + 1) * (maxShDegree + 1) - 1)
            });
            extraFeatures = extraFeatures.transpose(1, 2);
            dcFeatures = dcFeatures.unsqueeze(1);
            dcFeatures = dcFeatures.transpose(1, 2);

            std::cout << "dcFeatures shape: " << dcFeatures.sizes() << std::endl;
            std::cout << "extraFeatures shape: " << extraFeatures.sizes() << std::endl;

            shsTensor = torch::concatenate({
                                               dcFeatures.reshape({-1, 3}),
                                               extraFeatures.reshape({dcFeatures.size(0), -1})
                                           }, -1);
            xyzTensor *= 10;

            printTensor(xyzTensor);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }


    return {xyzTensor, rotationsTensor, scalesTensor, opacitiesTensor, shsTensor};
}


CudaImplementation::CudaImplementation(VkInstance* instance, VkDevice device, const RasterSettings* settings,
                                       const std::filesystem::path& modelPath, uint32_t memSizeCuda,
                                       std::vector<TextureCuda>* textures
) {
    uint32_t type = sizeof(uint8_t);
    uint32_t channels = 4;
    uint32_t imageSize = settings->imageHeight * settings->imageWidth * channels * type;
    cudaExtMem.resize(textures->size());
    cudaMemPtr.resize(textures->size());
    cudaMipMappedArrays.resize(textures->size());
    cudaFirstLevels.resize(textures->size());

    PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetInstanceProcAddr(*instance, "vkGetMemoryWin32HandleKHR"));
    if (fpGetMemoryWin32HandleKHR == nullptr) {
        Log::Logger::getInstance()->error("Function not available");
    }

    for (size_t i = 0; i < cudaExtMem.size(); ++i) {
        void* handle;
        VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
        vkMemoryGetWin32HandleInfoKHR.sType =
            VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
        vkMemoryGetWin32HandleInfoKHR.memory = (*textures)[i].m_DeviceMemory;
        vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        if (fpGetMemoryWin32HandleKHR(device, &vkMemoryGetWin32HandleInfoKHR,
                                      &handle) !=
            VK_SUCCESS) {
            Log::Logger::getInstance()->error("vkGetMemoryWin32HandleKHR not available");
        }

        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
        memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
        cudaExtMemHandleDesc.size = memSizeCuda;
        cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        cudaExtMemHandleDesc.handle.win32.handle = handle;
        cudaExtMemHandleDesc.flags = 0;
        checkCudaErrors(cudaImportExternalMemory(&cudaExtMem[i], &cudaExtMemHandleDesc));


        cudaChannelFormatDesc formatDesc;
        memset(&formatDesc, 0, sizeof(formatDesc));
        formatDesc.x = 8;
        formatDesc.y = 8;
        formatDesc.z = 8;
        formatDesc.w = 8;
        formatDesc.f = cudaChannelFormatKindUnsigned;

        cudaExtent extent = make_cudaExtent(settings->imageWidth, settings->imageHeight, 0);

        unsigned int flags = 0;
        //flags |= cudaArrayColorAttachment;

        cudaExternalMemoryMipmappedArrayDesc desc = {};
        memset(&desc, 0, sizeof(desc));
        desc.offset = 0;
        desc.formatDesc = formatDesc;
        desc.extent = extent;
        desc.flags = flags;
        desc.numLevels = 1;

        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipMappedArrays[i], cudaExtMem[i], &desc));

        checkCudaErrors(cudaGetMipmappedArrayLevel(&cudaFirstLevels[i], cudaMipMappedArrays[i], 0));
        // 0 for the first level
    }


    torch::Device cudaDevice(torch::kCUDA);

    campos = ConvertGlmVec3ToTensor(settings->camPos).to(cudaDevice);
    bg = torch::tensor({0.0, 0.0, 0.0}, torch::dtype(torch::kFloat32)).to(cudaDevice);
    // Other parameters
    scale_modifier = settings->scaleModifier;
    tan_fovx = settings->tanFovX;
    tan_fovy = settings->tanFovY;
    image_height = settings->imageHeight;
    image_width = settings->imageWidth;
    degree = settings->shDegree;
    prefiltered = settings->prefilter;
    debug = settings->debug;
    std::filesystem::path pointCloudPath = "point_cloud/iteration_7000/point_cloud.ply";
    std::filesystem::path fullPath = modelPath / pointCloudPath;
    auto gaussianData = loadTinyPly(fullPath);
    //auto gaussianData = naive_gaussian();
    // Example usage
    if (!gaussianData.length()) {
        gaussianData = naive_gaussian();
    }
    means3D = gaussianData.xyz.to(cudaDevice) * 100;
    shs = gaussianData.sh.to(cudaDevice);
    opacity = gaussianData.opacity.to(cudaDevice);
    scales = gaussianData.scale.to(cudaDevice);
    rotations = gaussianData.rot.to(cudaDevice);
    cov3Dprecompute = torch::tensor({}).to(cudaDevice);
    colors = torch::tensor({}).to(cudaDevice);
    degree = 3;

    shs = shs.view({gaussianData.length(), -1, 3}).contiguous();
}

void CudaImplementation::updateGaussianData() {
}


void CudaImplementation::updateCameraPose(glm::mat4 view, glm::mat4 proj, glm::vec3 target) {
    torch::Device device(torch::kCUDA);
    glm::vec3 cameraPos = glm::inverse(view)[3];
    /*
    glm::vec3 cameraPos = glm::inverse(view)[3];
    //glm::vec3 cameraPos(3.0f, -3.0f, 0.0f);
    glm::vec3 dirNorm = glm::normalize(target - cameraPos);
    glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
    glm::vec3 right = glm::cross(worldUp, dirNorm);
    glm::vec3 cameraUp = glm::cross(dirNorm, right);
    glm::mat4 cameraTrans(
        right.x, right.y, right.z, 0,
        cameraUp.x, cameraUp.y, cameraUp.z, 0,
        -dirNorm.x, -dirNorm.y, -dirNorm.z, 0,
        cameraPos.x, cameraPos.y, cameraPos.z, 1
    );
    view = glm::inverse(cameraTrans);
    //std::cout << dirNorm.x << ", " << dirNorm.y << ", " << dirNorm.z << std::endl;

    //view[0][0] *= -1;
    //view[1][0] *= -1;
    //view[2][0] *= -1;
    //view[3][0] *= -1;
    ////
    //view[0][2] *= -1;
    //view[1][2] *= -1;
    //view[2][2] *= -1;
    //view[3][2] *= -1;

    float aspect = 1280.0f / 720.0f;
    float farPlane = 100;
    float nearPlane = 0.01;
    float focal_length = 1.0f / tan(glm::radians(60.0f) * 0.5f);
    float x = focal_length / aspect;
    float y = -focal_length;
    float A = -farPlane / (farPlane - nearPlane);
    float B = -farPlane * nearPlane / (farPlane - nearPlane);


    proj = glm::mat4(
        x, 0.0f, 0.0f, 0.0f,
        0.0f, y, 0.0f, 0.0f,
        0.0f, 0.0f, A, -1.0f,
        0.0f, 0.0f, B, 0.0f);
    */

    glm::mat4 projView = proj * view;
    viewmatrix = ConvertGlmMat4ToTensor(view).to(device);
    projmatrix = ConvertGlmMat4ToTensor(projView).to(device);
    //printTensor(viewmatrix, true);
    //printTensor(projmatrix, true);
    campos = ConvertGlmVec3ToTensor(cameraPos).to(torch::kFloat).to(device);
}

void CudaImplementation::updateSettings(const CudaImplementation::RasterSettings& settings) {
    scale_modifier = settings.scaleModifier;
}

void CudaImplementation::updateCameraIntrinsics(float hfox, float hfovy) {
    tan_fovx = hfox;
    tan_fovy = hfovy;
}


void CudaImplementation::draw(uint32_t i, void* streamToRun) {
    int rendered;
    torch::Tensor out_color, radii, geomBuffer, binningBuffer, imgBuffer;
    // Call the function
    try {
        std::tie(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer) = RasterizeGaussiansCUDA(
            bg, means3D, colors, opacity, scales, rotations,
            scale_modifier, cov3Dprecompute, viewmatrix, projmatrix, tan_fovx, tan_fovy,
            image_height, image_width, shs, degree, campos, prefiltered, debug, streamToRun
        );
    }
    catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        rendered = 0;
    }
    // Ensure the tensor is on the CPU and is a byte tensor
    if (rendered == 0) {
        return;
    }
    auto img = out_color.permute({1, 2, 0}); // Change [Channels, Height, Width] to [Height, Width, Channels]
    img = img.contiguous();
    auto alpha_channel = torch::ones({img.size(0), img.size(1), 1}, img.options());
    auto img_with_alpha = torch::cat({img, alpha_channel}, 2);
    img_with_alpha = img_with_alpha.contiguous();
    img_with_alpha *= 255;
    img_with_alpha = img_with_alpha.to(torch::kU8);
    auto img_with_alpha_ptr = img_with_alpha.data_ptr<uint8_t>();
    cudaMemcpy3DParms p{};
    memset(&p, 0x00, sizeof(cudaMemcpy3DParms));

    p.srcPtr = make_cudaPitchedPtr(img_with_alpha_ptr, image_width * 4, image_width, image_height);
    p.dstArray = cudaFirstLevels[i];
    p.extent = make_cudaExtent(image_width, image_height, 1); // depth is 1 for 2D
    p.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&p));
    /*
    try {
        if (img.device().is_cuda()) {
            img = img.to(torch::kCPU);
            // Make sure the tensor is contiguous and in the format [Height, Width, Channels]
            img = img.contiguous();
            //img = img * 255;
            //torch::Tensor img_uchar = img.to(torch::kU8);
            //printTensor(img_uchar, 0);

            auto channels = CV_32FC(img.size(2));
            cv::Mat mat(img.size(0), img.size(1), channels,
                        img.data_ptr<float>());
            cv::Mat img_flipped;
            cv::Mat img_flipped_x;
            cv::flip(mat, img_flipped, 0); // Flip the image vertically
            cv::cvtColor(img_flipped, img_flipped, cv::COLOR_BGR2RGB);
            // Display the image
            cv::imshow("Output Image", img_flipped);
            //cv::imshow("Output Image flipped x", img_flipped_x);
            cv::waitKey(1); // Wait for a key press (use 0 for infinite wait)
        }
    }
    catch (const torch::Error& e) {
        std::cerr << "Error during tensor device check or transfer: " << e.what() << std::endl;
    }
    */

    //checkCudaErrors(cudaMemcpy(cudaMipMappedArrays[i],img_with_alpha.data_ptr() , data_size, cudaMemcpyDeviceToDevice));
}
