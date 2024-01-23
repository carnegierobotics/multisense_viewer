//
// Created by mgjer on 15/01/2024.
//

#include <iostream>
#include <random>
#include <rasterize_points.h>

#include "Viewer/Cuda/Cuda_example.h"

#include <glm/ext/quaternion_transform.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>

#include "Viewer/Tools/Logger.h"
#include <Viewer/Core/Texture.h>

#include <torch/torch.h>
#include <Viewer/Tools/helper_cuda.h>

#include <opencv2/opencv.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void printTensor(const torch::Tensor& tensor, bool printContents = false) {
    std::cout << "Shape/Size: " << tensor.sizes() << std::endl;
    std::cout << "Data Type: " << tensor.dtype() << std::endl;
    std::cout << "Device: " << tensor.device() << std::endl;
    if (printContents)
        std::cout << "Data: " << tensor << std::endl;
}

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

    size_t length() const {
        return xyz.size(0);
    }

    int sh_dim() const {
        return sh.size(-1);
    }
};

struct PlyVertex {
    float x, y, z; // Position
    float nx, ny, nz; // Normal
    float f_dc[3]; // f_dc properties
    float f_rest[44]; // f_rest properties
    float opacity;
    float scale[3];
    float rot[4];
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
                                   0.03, 0.03, 0.03,
                                   0.2, 0.03, 0.03,
                                   0.03, 0.2, 0.03,
                                   0.03, 0.03, 0.2
                               },
                               torch::dtype(torch::kFloat32)).view({-1, 3});
    auto gau_c = torch::tensor({
                                   1, 0, 1,
                                   1, 0, 0,
                                   0, 1, 0,
                                   0, 0, 1
                               },
                               torch::dtype(torch::kFloat32)).view({-1, 3});
    gau_c = (gau_c - 0.5) / 0.28209;
    auto gau_a = torch::tensor({1, 1, 1, 1}, torch::dtype(torch::kFloat32)).view({-1, 1});

    return GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c);
}

GaussianData loadPly(std::filesystem::path filePath) {
    std::ifstream plyFile(filePath);
    if (!plyFile.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return {};
    }

    // Skip the header as we already know the format
    std::string line;
    while (std::getline(plyFile, line)) {
        if (line == "end_header") {
            break;
        }
    }

    int numberOfVertices = 1026508; // As per your file's header
    std::vector<PlyVertex> vertices(numberOfVertices);

    for (int i = 0; i < numberOfVertices; ++i) {
        plyFile.read(reinterpret_cast<char*>(&vertices[i]), sizeof(PlyVertex));
    }

    plyFile.close();

    // Stack the tensors
    std::vector<float> x_values, y_values, z_values, opacity_values;
    for (const auto& v : vertices) {
        x_values.push_back(v.x);
        y_values.push_back(v.y);
        z_values.push_back(v.z);
        opacity_values.push_back(v.opacity);
    }

    auto x_tensor = torch::from_blob(x_values.data(), {static_cast<int64>(vertices.size())}, torch::kFloat32);
    auto y_tensor = torch::from_blob(y_values.data(), {static_cast<int64>(vertices.size())}, torch::kFloat32);
    auto z_tensor = torch::from_blob(z_values.data(), {static_cast<int64>(vertices.size())}, torch::kFloat32);
    auto xyz_tensor = torch::stack({x_tensor, y_tensor, z_tensor}, 1);
    auto opacities_tensor = torch::from_blob(opacity_values.data(), {static_cast<int64>(vertices.size())},
                                             torch::kFloat32);
    auto opacities_tensor_reshaped = opacities_tensor.unsqueeze(-1);

    // Continue reading the file based on header information

    int64_t num_vertices = vertices.size();
    auto features_dc = torch::zeros({num_vertices, 3, 1}, torch::kFloat32);
    std::vector<float> f_dc_0_values, f_dc_1_values, f_dc_2_values;
    for (const auto& v : vertices) {
        f_dc_0_values.push_back(v.f_dc[0]);
        f_dc_1_values.push_back(v.f_dc[1]);
        f_dc_2_values.push_back(v.f_dc[2]);
    }

    auto f_dc_0_tensor = torch::tensor(f_dc_0_values, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(1);
    auto f_dc_1_tensor = torch::tensor(f_dc_1_values, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(1);
    auto f_dc_2_tensor = torch::tensor(f_dc_2_values, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(1);

    features_dc.index_put_({torch::indexing::Slice(), 0, torch::indexing::Slice()}, f_dc_0_tensor);
    features_dc.index_put_({torch::indexing::Slice(), 1, torch::indexing::Slice()}, f_dc_1_tensor);
    features_dc.index_put_({torch::indexing::Slice(), 2, torch::indexing::Slice()}, f_dc_2_tensor);


    std::vector<std::string> propertyNames; // Assume this is filled with your property names
    std::vector<std::string> scale_names; // Assume this is filled with your property names
    std::vector<std::string> rot_names; // Assume this is filled with your property names
    for (int i = 0; i < 45; ++i) {
        std::string str = "f_rest_" + std::to_string(i);
        propertyNames.push_back(str);
        scale_names.push_back("scale_" + std::to_string(i));
        rot_names.push_back("rot_" + std::to_string(i));
    }
    std::vector<std::string> extra_f_names;

    for (const auto& name : propertyNames) {
        if (name.rfind("f_rest_", 0) == 0) {
            // Check if the name starts with "f_rest_"
            extra_f_names.push_back(name);
        }
    }

    // Sort extra_f_names based on the integer suffix
    std::sort(extra_f_names.begin(), extra_f_names.end(), [](const std::string& a, const std::string& b) {
        int numA = std::stoi(a.substr(a.find_last_of('_') + 1));
        int numB = std::stoi(b.substr(b.find_last_of('_') + 1));
        return numA < numB;
    });

    int max_sh_degree = 3;
    int expected_count = 3 * std::pow(max_sh_degree + 1, 2) - 3;
    if (extra_f_names.size() != expected_count) {
        std::cerr << "Unexpected number of properties." << std::endl;
        return {}; // Or handle the error as needed
    }

    auto features_extra = torch::zeros({num_vertices, static_cast<int64_t>(extra_f_names.size())}, torch::kFloat32);

    for (int64_t idx = 0; idx < extra_f_names.size(); ++idx) {
        std::vector<float> property_values;
        for (const auto& v : vertices) {
            // Assuming 'getProperty' is a function to get the property value by name
            property_values.push_back(v.f_rest[idx]);
        }
        auto property_tensor = torch::from_blob(property_values.data(), {num_vertices}, torch::kFloat32);
        features_extra.index_put_({torch::indexing::Slice(), idx}, property_tensor);
    }

    features_extra = features_extra.view({num_vertices, 3, static_cast<int64_t>(std::pow(max_sh_degree + 1, 2) - 1)});
    features_extra = features_extra.transpose(1, 2);

    auto scales = torch::zeros({num_vertices, static_cast<int64_t>(scale_names.size())}, torch::kFloat32);

    for (int64_t idx = 0; idx < scale_names.size(); ++idx) {
        std::vector<float> scale_values;
        for (const auto& v : vertices) {
            scale_values.push_back(v.scale[idx]);
        }
        auto scale_tensor = torch::from_blob(scale_values.data(), {num_vertices}, torch::kFloat32);
        scales.index_put_({torch::indexing::Slice(), idx}, scale_tensor);
    }
    auto rotations = torch::zeros({num_vertices, static_cast<int64_t>(rot_names.size())}, torch::kFloat32);

    for (int64_t idx = 0; idx < rot_names.size(); ++idx) {
        std::vector<float> rot_vals;
        for (const auto& v : vertices) {
            rot_vals.push_back(v.rot[idx]);
        }
        auto rot_tensor = torch::from_blob(rot_vals.data(), {num_vertices}, torch::kFloat32);
        rotations.index_put_({torch::indexing::Slice(), idx}, rot_tensor);
    }

    auto rots_norm = rotations.norm(2, -1, true);
    rotations = rotations.div(rots_norm);
    scales = scales.exp();
    auto opacities = torch::sigmoid(opacities_tensor_reshaped);
    auto features_dc_reshaped = features_dc.view({-1, 3});

    // features_extra is already [1026508, 15, 3]

    auto features_dc_expanded = features_dc_reshaped.unsqueeze(-1).expand({-1, -1, 3});

    auto shs = torch::cat({features_dc_expanded, features_extra}, 1);
    return {xyz_tensor, rotations, scales, opacities, shs};
}

CudaImplementation::CudaImplementation(const RasterSettings* settings, std::vector<void*> handles) {
    this->handles = handles;

    cudaExtMem.resize(handles.size());
    cudaMemPtr.resize(handles.size());
    cudaMipMappedArrays.resize(handles.size());

    for (size_t i = 0; i < handles.size(); ++i) {
        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
        memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
        cudaExtMemHandleDesc.size = settings->imageHeight * settings->imageWidth * 4;
        cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        cudaExtMemHandleDesc.handle.win32.handle = handles[i];
        cudaExtMemHandleDesc.flags = 0;
        checkCudaErrors(cudaImportExternalMemory(&cudaExtMem[i], &cudaExtMemHandleDesc));

        // Step 3: CUDA memory copy
        /*
                cudaExternalMemoryBufferDesc bufferDesc{};
                memset(&bufferDesc, 0, sizeof(bufferDesc));
                bufferDesc.offset = 0;
                bufferDesc.size = cudaExtMemHandleDesc.size;
                checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&cudaMemPtr[i], cudaExtMem[i], &bufferDesc));
                */
        cudaExternalMemoryMipmappedArrayDesc desc = {};
        memset(&desc, 0, sizeof(desc));

        cudaChannelFormatDesc formatDesc;
        memset(&formatDesc, 0, sizeof(formatDesc));
        formatDesc.x = 8;
        formatDesc.y = 8;
        formatDesc.z = 8;
        formatDesc.w = 8;
        formatDesc.f = cudaChannelFormatKindUnsigned;

        cudaExtent extent = {0, 0, 0};
        extent.width = settings->imageWidth;
        extent.height = settings->imageHeight;
        extent.depth = 0;

        unsigned int flags = 0;
        flags |= cudaArrayLayered;
        flags |= cudaArrayColorAttachment;

        desc.offset = 0;
        desc.formatDesc = formatDesc;
        desc.extent = extent;
        desc.flags = 0;
        desc.numLevels = 1;

        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipMappedArrays[i], cudaExtMem[i], &desc));
    }


    torch::Device device(torch::kCUDA);

    //viewmatrix = ConvertGlmMat4ToTensor(settings->viewMat).to(device);
    //projmatrix = ConvertGlmMat4ToTensor(settings->projMat).to(device);
    campos = ConvertGlmVec3ToTensor(settings->camPos).to(device);
    bg = torch::tensor({0.0, 0.0, 0.0}, torch::dtype(torch::kFloat32)).to(device);
    // Other parameters
    scale_modifier = settings->scaleModifier;
    tan_fovx = settings->tanFovX;
    tan_fovy = settings->tanFovY;
    image_height = settings->imageHeight;
    image_width = settings->imageWidth;
    degree = settings->shDegree;
    prefiltered = settings->prefilter;
    debug = settings->debug;

    //auto gaussianData = loadPly("C:\\Users\\mgjer\\Downloads\\models\\room\\point_cloud\\iteration_7000\\point_cloud.ply");
    auto gaussianData = naive_gaussian();
    // Example usage
    means3D = gaussianData.xyz.to(device);
    shs = gaussianData.sh.to(device);
    opacity = gaussianData.opacity.to(device);
    scales = gaussianData.scale.to(device);
    rotations = gaussianData.rot.to(device);
    cov3D_precomp = torch::tensor({}).to(device);
    colors = torch::tensor({}).to(device);
    degree = 0;

    //printTensor(means3D);
    //printTensor(rotations);
}

void CudaImplementation::updateGaussianData() {
    /*
    auto gaussianData = naive_gaussian();
    torch::Device device(torch::kCUDA);

    means3D = gaussianData.xyz.to(device);
    shs = gaussianData.sh.to(device);
    opacity = gaussianData.opacity.to(device);
    scales = gaussianData.scale.to(device);
    rotations = gaussianData.rot.to(device);
    cov3D_precomp = torch::tensor({}).to(device);
    colors = torch::tensor({}).to(device);

    degree = static_cast<int>(std::round(std::sqrt(gaussianData.sh_dim()))) - 1;
    */
}

void CudaImplementation::updateCameraPose(glm::mat4 view, glm::mat4 proj, glm::vec3 pos) {
    torch::Device device(torch::kCUDA);

    auto cameraPos = glm::vec3(glm::inverse(view)[3]);
    // Inverting the first and third rows of the view matrix
    // Note: GLM is column-major, so we access columns via view[col][row]
    view = glm::lookAt(-cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    view[0][0] = -view[0][0];
    view[1][0] = -view[1][0];
    view[2][0] = -view[2][0];
    view[3][0] = -view[3][0];
    ////
    view[0][2] = -view[0][2];
    view[1][2] = -view[1][2];
    view[2][2] = -view[2][2];
    view[3][2] = -view[3][2];

    proj = glm::perspective(60.0f, 1024.0f / 1024.0f, 0.01f, 100.0f);
    glm::mat4 projView = proj * view;
    viewmatrix = ConvertGlmMat4ToTensor(view).to(device);
    projmatrix = ConvertGlmMat4ToTensor(projView).to(device);
    campos = ConvertGlmVec3ToTensor(cameraPos).to(torch::kFloat).to(device);
}

void CudaImplementation::updateSettings(const CudaImplementation::RasterSettings& settings) {
    scale_modifier = settings.scaleModifier;
}

void CudaImplementation::updateCameraIntrinsics(float hfox, float hfovy) {
    tan_fovx = hfox;
    tan_fovy = hfovy;
}


void CudaImplementation::draw(uint32_t i) {
    int rendered;
    torch::Tensor out_color, radii, geomBuffer, binningBuffer, imgBuffer;
    // Call the function
    std::tie(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer) = RasterizeGaussiansCUDA(
        bg, means3D, colors, opacity, scales, rotations,
        scale_modifier, cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy,
        image_height, image_width, shs, degree, campos, prefiltered, debug
    );
    // Ensure the tensor is on the CPU and is a byte tensor

    auto img = out_color.permute({1, 2, 0}); // Change [Channels, Height, Width] to [Height, Width, Channels]
    img = img.contiguous();
    auto alpha_channel = torch::ones({img.size(0), img.size(1), 1}, img.options());
    auto img_with_alpha = torch::cat({img, alpha_channel}, 2);
    img_with_alpha = img_with_alpha.contiguous();
    size_t data_size = img_with_alpha.numel(); // Assuming the tensor is of type torch::kFloat
    //printTensor(img_with_alpha);

    cudaArray_t levelArray;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&levelArray, cudaMipMappedArrays[i], 0)); // 0 for the first level

    cudaMemcpy3DParms p = {0};
    p.srcPtr = make_cudaPitchedPtr(img_with_alpha.data_ptr(), 1024 * 16, 1024, 1024);
    p.dstArray = levelArray;
    p.extent = make_cudaExtent(1024, 1024, 1); // depth is 1 for 2D
    p.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&p));
    /*
    cudaMemcpy3DParms p = {0};
    p.srcPtr   = make_cudaPitchedPtr(img_with_alpha.data_ptr(), 1024 * sizeof(float), 1024, 1024);
    p.dstArray = levelArray;
    p.extent   = make_cudaExtent(width, height, depth); // depth is 1 for 2D
    p.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&p));
    */

    try {
        if (img.device().is_cuda()) {
            img = img.to(torch::kCPU);
            // Make sure the tensor is contiguous and in the format [Height, Width, Channels]
            img = img.contiguous();

            cv::Mat mat(img.size(0), img.size(1), CV_32FC(img.size(2)), img.data_ptr<float>());

            // Display the image
            cv::imshow("Output Image", mat);
            cv::waitKey(1); // Wait for a key press (use 0 for infinite wait)
        }
    }
    catch (const torch::Error& e) {
        std::cerr << "Error during tensor device check or transfer: " << e.what() << std::endl;
    }


    //checkCudaErrors(cudaMemcpy(cudaMipMappedArrays[i],img_with_alpha.data_ptr() , data_size, cudaMemcpyDeviceToDevice));
}
