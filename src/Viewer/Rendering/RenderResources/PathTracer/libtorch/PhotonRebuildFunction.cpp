//
// Created by magnus on 1/27/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
#include "stb_image_write.h"
#include <random>
#include <glm/gtx/quaternion.hpp>
#include <OpenImageDenoise/oidn.hpp>
#include <tiffio.h> // Make sure to include libtiff's header

namespace VkRender::PathTracer {
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


    static void saveTIFF(const std::filesystem::path &filename, const float *image, uint32_t width,
                         uint32_t height) {
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
                                                 IterationInfo &iterationInfo, PhotonTracer *pathTracer,
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
        ctx->saved_data["IterationInfo"] = reinterpret_cast<int64_t>(&iterationInfo); // example
        // or store the pointer as a raw pointer or shared pointer if you prefer
        // (but be careful with lifetimes).

        // =================
        // 2) Run your path tracer code that renders an image.

        // Example pseudo-code:

        pathTracer->update(iterationInfo.renderSettings);


        // For illustration:
        const PhotonTracer::PipelineSettings &photonTracerSettings = pathTracer->getPipelineSettings();
        int64_t height = photonTracerSettings.height;
        int64_t width = photonTracerSettings.width;
        float *rawImage = pathTracer->getImage();

        std::vector<float> denoisedImage;
        if (iterationInfo.denoise) {
            denoiseImage(rawImage, width, height, denoisedImage);
            rawImage = denoisedImage.data();
        }


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


    static void applyBoxBlur(const float *input, int width, int height, int kernelSize, std::vector<float> &output) {
        // Ensure kernelSize is odd.
        assert(kernelSize % 2 == 1);
        output.resize(width * height, 0.0f);
        int half = kernelSize / 2;

        // Loop over each pixel in the image.
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                int count = 0;
                // Iterate over the kernel window.
                for (int ky = -half; ky <= half; ++ky) {
                    int iy = y + ky;
                    if (iy < 0 || iy >= height)
                        continue;
                    for (int kx = -half; kx <= half; ++kx) {
                        int ix = x + kx;
                        if (ix < 0 || ix >= width)
                            continue;
                        sum += input[iy * width + ix];
                        ++count;
                    }
                }
                // Average over the valid pixels.
                output[y * width + x] = sum / static_cast<float>(count);
            }
        }
    }

    // Applies the Scharr filter to compute gradients.
    // 'image' is a pointer to the image data of size (width * height).
    // 'width' and 'height' are the dimensions of the image.
    // The function outputs gradient images gradX and gradY.
    static void applyScharrFilter(const float *image, int width, int height,
                                  std::vector<float> &gradX, std::vector<float> &gradY) {
        // Resize output vectors to hold the gradient images.
        gradX.resize(width * height, 0.0f);
        gradY.resize(width * height, 0.0f);

        // Scharr operator kernels for x and y derivatives.
        const float scharrX[3][3] = {
            {3, 0, -3},
            {10, 0, -10},
            {3, 0, -3}
        };

        const float scharrY[3][3] = {
            {3, 10, 3},
            {0, 0, 0},
            {-3, -10, -3}
        };

        // Loop over the image pixels, skipping the boundary pixels.
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float gx = 0.0f;
                float gy = 0.0f;
                // Convolve with the Scharr kernel.
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ix = x + kx;
                        int iy = y + ky;
                        float pixel = image[iy * width + ix];
                        gx += pixel * scharrX[ky + 1][kx + 1];
                        gy += pixel * scharrY[ky + 1][kx + 1];
                    }
                }
                gradX[y * width + x] = gx;
                gradY[y * width + x] = gy;
            }
        }

        int kernelSize = 9;
        std::vector<float> blurredGradX, blurredGradY;
        applyBoxBlur(gradX.data(), width, height, kernelSize, blurredGradX);
        applyBoxBlur(gradY.data(), width, height, kernelSize, blurredGradY);

        gradX = blurredGradX;
        gradY = blurredGradY;
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
        pathTracer->m_backwardInfo.gradientImage = dLoss_dRenderedImage.data_ptr<float>();
        auto gradients = pathTracer->backward(iterationInfo->renderSettings);

        // d_I(u,v) / d_(u,v)
        float *image = pathTracer->getImage();
        auto &props = pathTracer->getPipelineSettings();
        int width = props.width;
        int height = props.height;
        std::vector<float> gradX, gradY;
        applyScharrFilter(image, width, height, gradX, gradY);
        // Optionally, combine gradX and gradY to compute gradient magnitude:
        //std::vector<float> gradMag(width * height, 0.0f);
        //for (int i = 0; i < width * height; i++) {
        //    gradMag[i] = std::sqrt(gradX[i] * gradX[i] + gradY[i] * gradY[i]);
        //}
        std::filesystem::path gradientImagePathX =
                "debug/grad_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + "_x.png";
        std::filesystem::path gradientImagePathY =
                "debug/grad_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + "_y.png";
        // Save the gradient magnitude image as a PFM file.
        //saveTIFF(gradientImagePath, gradMag.data(), width, height);

        saveGradientAsPng(gradientImagePathX, width, height, gradX.data());
        saveGradientAsPng(gradientImagePathY, width, height, gradY.data());
        // Get the pointer to the loss gradient image (size: width*height)
        float *dLoss_dI = dLoss_dRenderedImage.data_ptr<float>();

        float *gradientImagePerObject = gradients.gradientImagePerObject;

        std::filesystem::path gradientPerPixelContributionPath =
                "debug/grad_id_image/" + cameraName + "/" + std::to_string(iterationInfo->iteration) + ".png";
        // Save the gradient magnitude image as a PFM file.
        saveLabelMaskAsPNG(gradientImagePerObject, gradientPerPixelContributionPath, width, height);

        auto posA = positions.accessor<float, 2>();
        auto gradientEmissivePositions = torch::zeros_like(positions);
        auto gradientQuadricPositions = torch::zeros_like(quadricPositions);
        auto gradPosA = gradientEmissivePositions.accessor<float, 2>();
        auto gradQuadPosA = gradientQuadricPositions.accessor<float, 2>();

        std::vector<glm::vec3> collectedGradients(pathTracer->getPipelineSettings().photonCount);
        glm::vec3 summedGradient = glm::vec3(0.0f);


        int numEntities = gradientQuadricPositions.size(0);
        struct EntityGradient {
            // Assume there are 'width*height' pixels.
            std::vector<glm::vec3> pixelGradientSum;
            std::vector<int> pixelGradientCount;
        };
        std::vector<EntityGradient> entityGradients(numEntities);
        // Loop over each photon.
        for (int i = 0; i < pathTracer->getPipelineSettings().photonCount; ++i) {
            glm::mat3 grad = gradients.photonIDGradient[i];
            glm::vec2 gradCoords = gradients.gradientPixelCoordinates[i];

            glm::vec3 dU_dPos = {grad[0][0], grad[1][0], grad[2][0]};
            glm::vec3 dV_dPos = {grad[0][1], grad[1][1], grad[2][1]};

            // Bilinear interpolation could be applied here instead of rounding,
            // but for simplicity we show rounding:
            int x = std::round(gradCoords.x);
            int y = std::round(gradCoords.y);
            size_t pixelIndex = x + y * width;

            // Determine which entity this pixel belongs to.
            int entityID = gradientImagePerObject[pixelIndex];
            if (entityID < 0 || entityID >= numEntities)
                continue; // Skip if invalid.

            if (entityGradients[entityID].pixelGradientSum.empty()) {
                entityGradients[entityID].pixelGradientSum = std::vector<glm::vec3>(width * height, glm::vec3(0.0f));
                entityGradients[entityID].pixelGradientCount = std::vector<int>(width * height, 0);
            }

            // Retrieve loss and image gradients for this pixel.
            float mseLoss = dLoss_dI[pixelIndex];
            float u_grad = gradX[pixelIndex];
            float v_grad = gradY[pixelIndex];

            glm::vec3 final_grad_pos_x = mseLoss * u_grad * dU_dPos;
            glm::vec3 final_grad_pos_y = mseLoss * v_grad * dV_dPos;
            glm::vec3 finalGradient = final_grad_pos_x + final_grad_pos_y;

            // Accumulate per-pixel.
            entityGradients[entityID].pixelGradientSum[pixelIndex] += finalGradient;
            entityGradients[entityID].pixelGradientCount[pixelIndex] += 1;
        }

        // Now, compute per-pixel averaged gradients.
        for (int entityIDX = 0; auto &entityGradient: entityGradients) {
            std::vector<glm::vec3> pixelGradientAvg(width * height, glm::vec3(0.0f));
            for (size_t i = 0; i < entityGradient.pixelGradientSum.size(); ++i) {
                if (entityGradient.pixelGradientCount[i] > 0)
                    pixelGradientAvg[i] = entityGradient.pixelGradientSum[i] / static_cast<float>(entityGradient.
                                              pixelGradientCount[i]);
                else
                    pixelGradientAvg[i] = glm::vec3(0.0f);
            }

            // Finally, combine per-pixel gradients into the final gradient over the scene parameter.
            // This depends on how your loss is defined; for an MSE loss defined as an average, you would
            // sum (or average) over all pixels.
            glm::vec3 finalGradScene(0.0f);
            int totalCount = 0;
            for (size_t i = 0; i < pixelGradientAvg.size(); ++i) {
                finalGradScene += pixelGradientAvg[i];
                totalCount++;
            }
            finalGradScene /= totalCount;

            gradQuadPosA[entityIDX][0] = finalGradScene.x;
            gradQuadPosA[entityIDX][1] = finalGradScene.y;
            gradQuadPosA[entityIDX][2] = finalGradScene.z;
            entityIDX++;
        }


        //finalGradScene /= static_cast<float>(totalCount);
        // or sum, if your loss derivative already includes a 1/N factor.
        //summedGradient.y = 0;


        /*
        for (int i = 0; i < gradientQuadricPositions.size(0); ++i) {
            gradQuadPosA[i][0] = gradientPerEntity[i].x / numGradientsSummed;
            gradQuadPosA[i][1] = gradientPerEntity[i].y / numGradientsSummed;
            gradQuadPosA[i][2] = gradientPerEntity[i].z / numGradientsSummed;
        }
        */

        /*
        for (int i = 0; i < gradientQuadricPositions.size(0); ++i) {
            // Allocate vectors to hold per-pixel contributions for u and v.
            std::vector<float> combinedU(width * height, 0.0f);
            std::vector<float> combinedV(width * height, 0.0f);
            // For each pixel, multiply the loss gradient with the image gradient
            for (int idx = 0; idx < width * height; idx++) {
                if (static_cast<int>(gradientImagePerObject[idx]) == i) {
                    combinedU[idx] = dLoss_dI[idx] * gradX[idx]; // contribution for u direction
                    combinedV[idx] = dLoss_dI[idx] * gradY[idx]; // contribution for v direction
                }
            }
            // Sum over all pixels to aggregate to a single scalar for each coordinate.
            float sumU = std::accumulate(combinedU.begin(), combinedU.end(), 0.0f);
            float sumV = std::accumulate(combinedV.begin(), combinedV.end(), 0.0f);

            glm::mat3 grad = gradients.photonIDGradient[i];
            glm::vec2 gradCoords = gradients.gradientPixelCoordinates[i];

            if (glm::any(glm::isnan(grad[0])) || glm::any(glm::isnan(grad[1])) || glm::any(glm::isnan(grad[2]))) {
                Log::Logger::getInstance()->error("Error: NaN detected in gradient matrix!");
            }

            glm::vec3 dU_dPos = {grad[0][0], grad[1][0], grad[2][0]};
            glm::vec3 dV_dPos = {grad[0][1], grad[1][1], grad[2][1]};
            // Finally, combine the contributions:
            // dL/dpos = (sumU) * d(u)/d(pos) + (sumV) * d(v)/d(pos)
            glm::vec3 final_grad_pos_x = sumU * dU_dPos;
            glm::vec3 final_grad_pos_y = sumV * dV_dPos;
            glm::vec3 finalGradient = final_grad_pos_x + final_grad_pos_y;
            float x = finalGradient.x;
            float y = finalGradient.y;
            float z = finalGradient.z;

        }
        */

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
