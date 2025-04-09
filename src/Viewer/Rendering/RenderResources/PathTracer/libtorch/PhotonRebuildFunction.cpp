//
// Created by magnus on 1/27/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
#include "stb_image_write.h"
#include <random>
#include <stb_image.h>

#include <glm/gtx/quaternion.hpp>
#include <OpenImageDenoise/oidn.hpp>
#include <tiffio.h> // Make sure to include libtiff's header

namespace VkRender::PathTracer {

    static void saveImageAsPng(std::filesystem::path &filename, uint32_t width, uint32_t height, float* image) {
        std::filesystem::path dir = filename.parent_path();

        // Create directory if it doesn't exist
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }


        std::vector<uint8_t> rgbDataPng(width * height * 3);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = (y * width + x);
                uint32_t rgbIndex = pixelIndex * 3;

                // Assuming image is in RGBA format with float values in range [0.0, 1.0]
                rgbDataPng[rgbIndex + 0] = static_cast<uint8_t>(image[pixelIndex] * 255.0f);
                // R
                rgbDataPng[rgbIndex + 1] = static_cast<uint8_t>(image[pixelIndex] * 255.0f);
                // G
                rgbDataPng[rgbIndex + 2] = static_cast<uint8_t>(image[pixelIndex] * 255.0f);
                // B
            }
        }


        // Write the image to a PNG file
        if (!stbi_write_png(filename.replace_extension(".png").string().c_str(), width, height, 3,
                            rgbDataPng.data(),
                            width * 3)) {
            throw std::runtime_error("Failed to write PNG file: " + filename.string());
        }

    }

    static void save_gradient_to_png(torch::Tensor gradient, const std::filesystem::path &filename) {
        std::filesystem::path dir = filename.parent_path();

        // Create directory if it doesn't exist
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        // Ensure the tensor is on CPU and in float32
        gradient = gradient.detach().cpu().to(torch::kFloat32);

        // Normalize to [0, 255]
        auto min = gradient.min().item<float>();
        auto max = gradient.max().item<float>();
        auto normalized = (gradient - min) / (max - min) * 255.0;

        // Convert to uint8
        auto uint8_tensor = normalized.to(torch::kUInt8);

        // Get raw pointer
        uint8_t *data = uint8_tensor.data_ptr<uint8_t>();

        // Get dimensions
        int width = gradient.size(1);
        int height = gradient.size(0);
        // Save as PNG using stb_image_write
        stbi_write_png(filename.c_str(), width, height, 1, data, width);


        std::filesystem::path filenamePath = filename;
        std::ofstream file(filenamePath.replace_extension(".pfm"), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error(
                "Failed to open file for writing: " + filenamePath.replace_extension(".pfm").string());
        }
        // Write the PFM header
        // "PF" indicates a color image. Use "Pf" for grayscale.
        file << "Pf\n" << width << " " << height << "\n-1.0\n";

        // PFM expects the data in binary format, row by row from top to bottom
        // Assuming your m_imageMemory is in RGBA format with floats

        // Allocate a temporary buffer for RGB data
        std::vector<float> rgbData(width * height);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = (y * width + x);
                uint32_t rgbIndex = (y * width + x);
                rgbData[rgbIndex + 0] = gradient.data_ptr<float>()[pixelIndex] * 255; // R
            }
        }

        // Write the RGB float data
        file.write(reinterpret_cast<const char *>(rgbData.data()), rgbData.size() * sizeof(float));

        if (!file) {
            throw std::runtime_error(
                "Failed to write PFM data to file: " + filenamePath.replace_extension(".pfm").string());
        }

        file.close();
    }


    static void saveTIFF(const std::filesystem::path &filename, uint32_t width,
                         uint32_t height, const float *image) {
        // Create the directory if it doesn't exist.
        std::filesystem::path dir = filename.parent_path();
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        // Open the TIFF file for writing.
        TIFF *tif = TIFFOpen(filename.string().c_str(), "w");
        if (!tif) {
            throw std::runtime_error("Unable to open TIFF file for writing.");
        }

        // Set TIFF fields.
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32); // 32-bit float
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP); // IEEE floating point
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1); // Single channel
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // Grayscale
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height); // Write the whole image in one strip.

        // Write the image row by row.
        // TIFF expects each scanline to be contiguous in memory.
        for (uint32_t row = 0; row < height; row++) {
            // The starting pointer of the row in the image vector.
            const float *rowData = &image[row * width];
            if (TIFFWriteScanline(tif, (tdata_t) rowData, row, 0) < 0) {
                TIFFClose(tif);
                throw std::runtime_error("Failed to write TIFF scanline.");
            }
        }

        TIFFClose(tif);
    }

    static void savePFM(const std::filesystem::path &filename, const std::vector<float> &image, uint32_t width,
                        uint32_t height) {
        std::filesystem::path dir = filename.parent_path();

        // Create directory if it doesn't exist
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }


        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file for writing.");
        }

        // Write the PFM header.
        // "PF" indicates a color image. Use "Pf" for grayscale.
        file << "PF\n" << width << " " << height << "\n-1.0\n";

        // PFM expects the data in binary format, row by row from top to bottom.
        // Here we assume that 'image' is a single-channel float image (size: width*height).
        // We create a temporary buffer for RGB data.
        std::vector<float> rgbData(width * height * 3);

        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t pixelIndex = y * width + x;
                uint32_t rgbIndex = pixelIndex * 3;
                // Duplicate the grayscale value across R, G, and B.
                rgbData[rgbIndex + 0] = image[pixelIndex];
                rgbData[rgbIndex + 1] = image[pixelIndex];
                rgbData[rgbIndex + 2] = image[pixelIndex];
            }
        }

        // PFM files expect the data to be written row by row from the top row to bottom.
        // Depending on how your image is stored (top-to-bottom or bottom-to-top),
        // you might need to flip the rows. Here we assume the 'image' vector is top-to-bottom.
        file.write(reinterpret_cast<const char *>(rgbData.data()), rgbData.size() * sizeof(float));
        file.close();
    }

    static void denoiseImage(float *singleChannelImage, uint32_t width, uint32_t height,
                             std::vector<float> &output) {
        // Initialize OIDN device and commit
        oidn::DeviceRef device = oidn::newDevice();
        device.commit();
        const uint32_t imageSize = width * height;

        // Allocate input and output buffers for OIDN
        oidn::BufferRef inputBuffer = device.newBuffer(imageSize * sizeof(float));
        oidn::BufferRef outputBuffer = device.newBuffer(imageSize * sizeof(float));

        // Copy input data to the device buffer
        std::memcpy(inputBuffer.getData(), singleChannelImage, imageSize * sizeof(float));

        // Create and configure the denoising filter
        oidn::FilterRef filter = device.newFilter("RT");
        filter.set("hdr", true);
        filter.setImage("color", inputBuffer, oidn::Format::Float, width, height);
        filter.setImage("output", outputBuffer, oidn::Format::Float, width, height);
        filter.commit();

        // Execute the filter
        filter.execute();

        // Check for errors from OIDN
        const char *errorMessage;
        if (device.getError(errorMessage) != oidn::Error::None) {
            std::cerr << "OIDN Error: " << errorMessage << std::endl;
            return;
        }

        // Retrieve the denoised image data
        output.resize(imageSize);
        std::memcpy(output.data(), outputBuffer.getData(), imageSize * sizeof(float));
    }

    torch::Tensor PhotonRebuildFunction::forward(torch::autograd::AutogradContext *ctx,
                                                 IterationInfo* iterationInfo, PhotonTracer *pathTracer,
                                                 torch::Tensor positions, torch::Tensor scales,
                                                 torch::Tensor normals, torch::Tensor emissions,
                                                 torch::Tensor colors,
                                                 torch::Tensor specular,
                                                 torch::Tensor diffuse,
                                                 torch::Tensor quadrics,
                                                 torch::Tensor quadricPositions,
                                                 torch::Tensor quadricRotations
    ) {
        // =================
        // 1) Save for backward any Tensors or scalar values you need
        //    to compute derivatives later. For example:
        ctx->save_for_backward({
            positions, scales, normals, emissions, colors, specular, diffuse, quadrics,
            quadricPositions, quadricRotations
        });
        ctx->saved_data["pathTracer"] = reinterpret_cast<int64_t>(pathTracer);

        // If you have non-tensor data you want in backward(), you can store
        // them as attributes:
        ctx->saved_data["IterationInfo"] = reinterpret_cast<int64_t>(iterationInfo); // example
        // or store the pointer as a raw pointer or shared pointer if you prefer
        // (but be careful with lifetimes).

        // =================
        // 2) Run your path tracer code that renders an image.

        // Example pseudo-code:

        pathTracer->update(iterationInfo->renderSettings);


        // For illustration:
        const PhotonTracer::PipelineSettings &photonTracerSettings = pathTracer->getPipelineSettings();
        int64_t height = photonTracerSettings.height;
        int64_t width = photonTracerSettings.width;
        float *rawImage = pathTracer->getImage();

        // Suppose the path tracer writes out to pathTracer->m_imageMemory,
        // with shape [height * width] or [height * width * channels].
        // We'll build a Torch tensor from that raw memory.


        // e.g. a float[height * width]  (gray) or float[height * width * 3]

        // Wrap it in a Torch tensor.
        // Note from_blob does not take ownership, so we typically clone().
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto output = torch::from_blob(rawImage, {height, width}, options).clone();


        // Return the rendered image
        return output;
    }

    static void saveAsPng(std::filesystem::path filePath, int width, int height, void *data) {
        std::filesystem::path dir = filePath.parent_path();

        // Create directory if it doesn't exist
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }



        // Save as PNusing stb_image_write
        stbi_write_png(filePath.c_str(), width, height, 3, data, width * 3);
    }

    static void saveLabelMaskAsPNG(const float *gradientImagePerObject,
                                   const std::filesystem::path &mseGradientImagePath, int width, int height) {
        // Create an RGB image buffer (3 channels per pixel)
        std::vector<unsigned char> colorImage(width * height * 3, 0);

        // Define a color palette for up to 10 classes.
        // Each class gets a unique RGB color.
        const std::array<std::array<unsigned char, 3>, 10> classColors = {
            {
                {255, 0, 0}, // Class 0: Red
                {0, 255, 0}, // Class 1: Green
                {0, 0, 255}, // Class 2: Blue
                {255, 255, 0}, // Class 3: Yellow
                {255, 0, 255}, // Class 4: Magenta
                {0, 255, 255}, // Class 5: Cyan
                {128, 0, 0}, // Class 6: Dark Red
                {0, 128, 0}, // Class 7: Dark Green
                {0, 0, 128}, // Class 8: Dark Blue
                {128, 128, 128} // Class 9: Gray
            }
        };

        // Process each pixel in the input image.
        for (int i = 0; i < width * height; ++i) {
            float label = gradientImagePerObject[i];
            // If no class is selected, label is FLT_MAX. Set to black.
            if (label > 100) {
                colorImage[i * 3 + 0] = 0;
                colorImage[i * 3 + 1] = 0;
                colorImage[i * 3 + 2] = 0;
            } else {
                // Convert the float label to an integer class index.
                int classIndex = static_cast<int>(label);
                if (classIndex >= 0 && classIndex < static_cast<int>(classColors.size())) {
                    colorImage[i * 3 + 0] = classColors[classIndex][0];
                    colorImage[i * 3 + 1] = classColors[classIndex][1];
                    colorImage[i * 3 + 2] = classColors[classIndex][2];
                } else {
                    // If the label is outside the expected range, default to black.
                    colorImage[i * 3 + 0] = 0;
                    colorImage[i * 3 + 1] = 0;
                    colorImage[i * 3 + 2] = 0;
                }
            }
        }

        saveAsPng(mseGradientImagePath, width, height, colorImage.data());
    }

    static void saveGradientAsPng(std::filesystem::path gradientImagePath, int width, int height, float *data) {
        // Compute a normalization factor (max absolute gradient value)
        float maxAbs = 0.0f;
        for (int i = 0; i < width * height; i++) {
            float v = data[i]; // grad[i] is the gradient value at pixel i
            if (fabs(v) > maxAbs)
                maxAbs = fabs(v);
        }
        if (maxAbs < 1e-6f)
            maxAbs = 1.0f; // Avoid division by zero if all gradients are nearly zero

        // Create an RGB image buffer (unsigned char per channel)
        std::vector<unsigned char> colorImage(width * height * 3, 0);

        for (int i = 0; i < width * height; i++) {
            // Normalize the gradient to [-1, +1]
            float v = data[i] / maxAbs;

            // We'll choose a simple linear mapping:
            // At v = -1: full blue: (0, 0, 255)
            // At v = 0: neutral gray: (128, 128, 128)
            // At v = +1: full red: (255, 0, 0)
            unsigned char r, g, b;
            if (v < 0.0f) {
                // Map negative values: as v goes from 0 to -1, interpolate from neutral to blue.
                float t = -v; // t in [0,1]
                r = static_cast<unsigned char>((1.0f - t) * 128.0f);
                g = static_cast<unsigned char>((1.0f - t) * 128.0f);
                b = static_cast<unsigned char>(t * 255.0f + (1.0f - t) * 128.0f);
            } else if (v > 0.0f) {
                // Map positive values: as v goes from 0 to 1, interpolate from neutral to red.
                float t = v; // t in [0,1]
                r = static_cast<unsigned char>(t * 255.0f + (1.0f - t) * 128.0f);
                g = static_cast<unsigned char>((1.0f - t) * 128.0f);
                b = static_cast<unsigned char>((1.0f - t) * 128.0f);
            } else {
                // For zero, use the neutral color.
                r = 128;
                g = 128;
                b = 128;
            }
            colorImage[i * 3 + 0] = r;
            colorImage[i * 3 + 1] = g;
            colorImage[i * 3 + 2] = b;
        }

        saveAsPng(gradientImagePath, width, height, colorImage.data());
    }



    void saveArrowFieldJson(std::filesystem::path filePath,
                            size_t photons,
                            const std::vector<glm::vec3> &origins,
                            const std::vector<glm::vec3> &directions)
    {

        std::filesystem::path dir = filePath.parent_path();

        // Create directory if it doesn't exist
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        nlohmann::json j;
        j["size"] = photons;

        // Use a small epsilon to decide if a vector is "zero"
        const float epsilon = std::numeric_limits<float>::epsilon();

        // Reserve space for the flattened arrays: 3 floats per arrow.
        std::vector<float> origins_flat;
        origins_flat.reserve(photons * 3);
        std::vector<float> directions_flat;
        directions_flat.reserve(photons * 3);

        // Iterate once over all photons (gradients)
        for (size_t i = 0; i < photons; ++i) {
            // Skip if either origin or direction is near zero.
            if (glm::length(origins[i]) < epsilon)
                continue;
            if ( glm::length(directions[i]) < epsilon)
                continue;

            // Append origin (3 floats)
            origins_flat.push_back(origins[i].x);
            origins_flat.push_back(origins[i].y);
            origins_flat.push_back(origins[i].z);

            // Append direction (3 floats)
            directions_flat.push_back(directions[i].x);
            directions_flat.push_back(directions[i].y);
            directions_flat.push_back(directions[i].z);
        }

        j["origins"] = origins_flat;
        j["directions"] = directions_flat;

        // Write the JSON to file
        std::ofstream outFile(filePath.string());
        if (!outFile) {
            throw std::runtime_error("Could not open " + filePath.string() + " for writing JSON");
        }
        outFile << j.dump(2) << std::endl;
        outFile.close();
        std::cout << "Saved arrow field to " << filePath.string() << std::endl;
    }
    torch::autograd::tensor_list PhotonRebuildFunction::backward(torch::autograd::AutogradContext *ctx,
                                                                 torch::autograd::tensor_list grad_outputs) {
        // Usually, the forward returned 1 tensor => grad_outputs.size() == 1
        // grad_outputs[0] is d(L)/d(output).

        auto dLoss_dRenderedImage = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto positions = saved[0];
        auto scales = saved[1];
        auto normals = saved[2];
        auto emissions = saved[3];
        auto colors = saved[4];
        auto specular = saved[5];
        auto diffuse = saved[6];
        auto quadrics = saved[7];
        auto quadricPositions = saved[8];
        auto quadricRotations = saved[9];

        // Retrieve the path tracer pointer
        auto pathTracerRaw = ctx->saved_data["pathTracer"].toInt();
        PhotonTracer *pathTracer = reinterpret_cast<PhotonTracer *>(pathTracerRaw);
        // Retrieve the path tracer pointer
        auto settingsPtr = ctx->saved_data["IterationInfo"].toInt();
        IterationInfo *iterationInfo = reinterpret_cast<IterationInfo *>(settingsPtr);
        std::string cameraName = iterationInfo->cameraName;
        std::filesystem::path mseGradientImagePath =
                "./debug/mse_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + ".png";
        save_gradient_to_png(dLoss_dRenderedImage, mseGradientImagePath);

        std::filesystem::path mseGradientImagePath2 =
        "./debug/mse_image/all/" + std::to_string(iterationInfo->iteration) + ".png";
        save_gradient_to_png(dLoss_dRenderedImage, mseGradientImagePath2);

        auto gradients = pathTracer->backward(iterationInfo->renderSettings);
        float* mseImage = dLoss_dRenderedImage.data_ptr<float>();
        float *image = pathTracer->getImage();
        auto &props = pathTracer->getPipelineSettings();
        int width = props.width;
        int height = props.height;

        // Save the gradient magnitude image as a PFM file.
        //saveTIFF(gradientImagePath, gradMag.data(), width, height);

        //saveGradientAsPng(gradientImagePathY, width, height, gradY.data());
        // Get the pointer to the loss gradient image (size: width*height)
        //float *dLoss_dI = dLoss_dRenderedImage.data_ptr<float>();

        float *gradientImagePerObject = gradients.gradientImagePerObject;



        auto posA = positions.accessor<float, 2>();
        auto gradientEmissivePositions = torch::zeros_like(positions);
        auto gradientQuadricPositions = torch::zeros_like(quadricPositions);
        auto gradPosA = gradientEmissivePositions.accessor<float, 2>();
        auto gradQuadPosA = gradientQuadricPositions.accessor<float, 2>();

        auto& settings = pathTracer->getPipelineSettings();

        std::vector<glm::vec3 > gradientPerEntity(gradientQuadricPositions.size(0), glm::vec3(0.0f));


        /*
        for (int i = 0; i < settings.width * settings.height; i++) {
            glm::mat3 grad = gradients.photonIDGradient[i];
            glm::vec3 gradient = {grad[0][0], grad[1][0], grad[2][0]};

            if (glm::any(glm::isnan(gradient)))
                continue;

            float mseLoss = mseImage[i];
            glm::vec3 L_mse_qc = -mseLoss * gradient;
            // Now, add the transformed gradient to the entity's gradient accumulator:
            gradientPerEntity[0] += L_mse_qc;
        }
        */

        std::vector<glm::vec3> L_mse_qc(pathTracer->getPipelineSettings().photonCount, glm::vec3(0.0f));
        std::vector<glm::vec3> L_Iuv_qc(pathTracer->getPipelineSettings().photonCount, glm::vec3(0.0f));
        std::vector<glm::vec3> L_mse_qc_origin(pathTracer->getPipelineSettings().photonCount, glm::vec3(0.0f));

        for (int i = 0; i < pathTracer->getPipelineSettings().photonCount; ++i) {
            glm::mat3 grad = gradients.photonIDGradient[i];
            glm::vec3 gradient = {grad[0][0], grad[1][0], grad[2][0]};
            glm::vec3 origin = {grad[0][1], grad[1][1], grad[2][1]};

            if  (glm::any(glm::isnan(gradient))){
                continue;
            }

            glm::vec2 gradCoords = gradients.gradientPixelCoordinates[i];

            int x = static_cast<int>(std::round(gradCoords.x));
            int y = static_cast<int>(std::round(gradCoords.y));

            size_t pixelIndex = x + y * width;

            L_Iuv_qc[i] =  gradient;
            L_mse_qc_origin[i] = origin;

            int entityID = gradientImagePerObject[pixelIndex];
            if (entityID >= gradientPerEntity.size())
                continue;

            float mseLoss = mseImage[pixelIndex];

            glm::vec3 finalGradient = mseLoss * gradient;


            L_mse_qc[i] =  mseLoss * gradient;


            // Now, add the transformed gradient to the entity's gradient accumulator:
            gradientPerEntity[entityID] += finalGradient;
        }
        if (iterationInfo->saveDebugInfo) {
        saveArrowFieldJson("debug/mse_grad_field/" + cameraName + "/" +  std::to_string(iterationInfo->iteration) +"_debug_arrow_field.json", pathTracer->getPipelineSettings().photonCount, L_mse_qc_origin, L_mse_qc);
        saveArrowFieldJson("debug/Iuv_grad_field/" + cameraName + "/" +  std::to_string(iterationInfo->iteration) +"_debug_arrow_field.json", pathTracer->getPipelineSettings().photonCount, L_mse_qc_origin, L_Iuv_qc);
        saveArrowFieldJson("debug/Iuv_grad_field/all/" +  std::to_string(iterationInfo->iteration) +"_debug_arrow_field.json", pathTracer->getPipelineSettings().photonCount, L_mse_qc_origin, L_Iuv_qc);
        saveArrowFieldJson("debug/mse_grad_field/all/" +  std::to_string(iterationInfo->iteration) +"_debug_arrow_field.json", pathTracer->getPipelineSettings().photonCount, L_mse_qc_origin, L_mse_qc);
        std::filesystem::path gradientImagePathX =
                "debug/grad_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + "_x.tiff";
        std::filesystem::path gradientImagePathY =
                "debug/grad_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + "_y.tiff";

        std::filesystem::path gradientImagePathXAll =
                "debug/grad_image/all/" + std::to_string(iterationInfo->iteration) + "_x.tiff";
        std::filesystem::path gradientImagePathYAll =
                "debug/grad_image/all/" + std::to_string(iterationInfo->iteration) + "_y.tiff";

        saveTIFF(gradientImagePathX, width, height, gradients.gradientImageHoriz);
        saveTIFF(gradientImagePathY, width, height, gradients.gradientImageVert);
        saveTIFF(gradientImagePathXAll, width, height, gradients.gradientImageHoriz);
        saveTIFF(gradientImagePathYAll, width, height, gradients.gradientImageVert);
        std::filesystem::path gradientPerPixelContributionPath =
        "debug/grad_id_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + ".png";
        saveLabelMaskAsPNG(gradientImagePerObject, gradientPerPixelContributionPath, width, height);
        std::filesystem::path gradientPerPixelContributionPathAll =
                "debug/grad_id_image/all/" + std::to_string(iterationInfo->iteration) + ".png";
        saveLabelMaskAsPNG(gradientImagePerObject, gradientPerPixelContributionPathAll, width, height);
        std::filesystem::path renderedImagePath =
               "debug/rendered_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + ".png";
        saveTIFF(renderedImagePath.replace_extension("tiff"), width, height, image);
        }
        /*
        std::vector<uint8_t> imageRGB8(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            glm::vec3 color = glm::clamp(mseWeightedImage[i], 0.0f, 1.0f);
            imageRGB8[i * 3 + 0] = static_cast<uint8_t>(color.r * 255.0f);
            imageRGB8[i * 3 + 1] = static_cast<uint8_t>(color.g * 255.0f);
            imageRGB8[i * 3 + 2] = static_cast<uint8_t>(color.b * 255.0f);
        }

        std::filesystem::path finalGradientImage =
                "debug/grad_mse/all/" + std::to_string(iterationInfo->iteration) + ".png";


        saveAsPng(finalGradientImage, width, height, imageRGB8.data());
            */
        // Save Screen Space Gradient:

        iterationInfo->gradients.entityGradients.resize( gradientQuadricPositions.size(0));
        iterationInfo->gradients.screenSpaceGradients.resize( gradientQuadricPositions.size(0));
        for (int i = 0; i < gradientQuadricPositions.size(0); ++i) {
            float grad_x = gradientPerEntity[i].x ;
            float grad_y = gradientPerEntity[i].y ;
            float grad_z = gradientPerEntity[i].z ;

            if (std::isnan(grad_x) || std::isnan(grad_y) || std::isnan(grad_z)) {
                std::cout << "NAN warning: Gradient: (" << grad_x << ", " << grad_y << ", " << grad_z << ")" << std::endl;
                continue;

            }

            gradQuadPosA[i][0] = grad_x;
            gradQuadPosA[i][1] = grad_y;
            gradQuadPosA[i][2] = grad_z;

            std::cout << "Final Gradient: (" << grad_x << ", " << grad_y << ", " << grad_z << ")" << std::endl;

            iterationInfo->gradients.entityGradients[i] = gradientPerEntity[i];
        }


        // Return them in the same order as forward inputs
        return {
            torch::Tensor(), // wrt settings (not a Tensor)
            torch::Tensor(), // wrt pathTracer (not a Tensor)
            gradientEmissivePositions, // wrt positions
            torch::Tensor(), // wrt scales
            torch::Tensor(), // wrt normals
            torch::Tensor(), // emission
            torch::Tensor(), // colors
            torch::Tensor(), // specular
            torch::Tensor(), // diffuse
            torch::Tensor(), // gradQuadApperance
            gradientQuadricPositions, // gradQUadPos
            torch::Tensor() // gradQUadPos
        };
    }
}
