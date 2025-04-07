//
// Created by magnus on 12/27/24.
//

#ifndef LightTracerKernel_H
#define LightTracerKernel_H

#include <glm/gtc/matrix_inverse.hpp>

#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracerKernelCommon.h"

namespace VkRender::PathTracer {
    class LightTracerKernel {
    public:
        LightTracerKernel(GPUData gpuData,
                          GPUDataOutput *gpuDataOutput,
                          PCG32 *rng)
            : m_gpuData(gpuData), m_gpuDataOutput(gpuDataOutput), m_rng(rng) {
            m_cameraTransform = m_gpuData.cameraTransform;
            m_camera = m_gpuData.pinholeCamera;
        }

        void operator()(sycl::item<1> item) const {
            size_t photonID = item.get_linear_id();
            if (photonID >= m_gpuData.renderInformation->totalPhotons) {
                return;
            }
            // Each thread traces one photon.
            traceOnePhoton(photonID);
        }

    private:
        GPUData m_gpuData{};
        GPUDataOutput *m_gpuDataOutput = nullptr;

        PCG32 *m_rng = nullptr;
        TransformComponent *m_cameraTransform = nullptr;
        PinholeCamera *m_camera = nullptr;

        // ---------------------------------------------------------
        // Single Photon Trace (Multi-Bounce)
        // ---------------------------------------------------------
        void traceOnePhoton(size_t photonID) const {
            // 1) Pick an emissive triangle and sample a random point on it
            size_t gaussianID = sampleRandomEmissiveGaussian(photonID);
            glm::vec3 emitPosLocal(0.0f), emitNormalLocal(0.0f);
            float photonFlux = 0.0f;
            sampleEmissiveDirectionAndPower(gaussianID, photonID, emitPosLocal, emitNormalLocal,
                                            photonFlux);

            // 2) Sample emission direction
            glm::vec3 rayDir = sampleCosineWeightedHemisphere(emitNormalLocal, photonID);
            glm::vec3 rayOrigin = emitPosLocal;

            //rayOrigin = glm::vec3(0.0f, 0.0f, 5.0f);
            //rayDir = glm::normalize(glm::vec3(0.05f, 0.00f, -1.0f));

            float apertureDiameter = (m_camera->parameters().focalLength / m_camera->parameters().fNumber) / 1000;
            float apertureRadius = 0.0f;

            glm::mat4 entityTransform = m_cameraTransform->getTransform();
            glm::vec3 cameraPlaneNormalWorld =
                    glm::normalize(glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));
            m_gpuDataOutput[photonID].emissionOrigin = rayOrigin;
            m_gpuDataOutput[photonID].emissionDirection = rayDir;
            m_gpuDataOutput[photonID].gaussianID = gaussianID;

            glm::vec3 directLightDir(0.0f);
            float camera_t = 0.0f;
            glm::vec3 apertureHitPoint(0.0f);
            glm::vec3 cameraHitPointLocal(0.0f);
            float scalePowerDirectLighting = 0.00000001 * photonFlux;
            glm::vec2 pixelCoordHit = glm::vec2(0.0f);
            if (castContributionRay(rayOrigin, cameraPlaneNormalWorld, apertureRadius, photonID,
                                    scalePowerDirectLighting
                                    , directLightDir, apertureHitPoint, cameraHitPointLocal, pixelCoordHit, camera_t
            )) {
                m_gpuDataOutput[photonID].directLightingDir = directLightDir;
                m_gpuDataOutput[photonID].emissionDirectionLength = camera_t;
                m_gpuDataOutput[photonID].apertureHitPoint = apertureHitPoint;
                m_gpuDataOutput[photonID].cameraHitPointLocal = cameraHitPointLocal;
                m_gpuDataOutput[photonID].hitCamera = true;
                m_gpuDataOutput[photonID].emissionDirection = directLightDir;
            }

            // 3) Multi-bounce loop
            for (uint32_t bounce = 0; bounce < m_gpuData.renderInformation->numBounces; ++bounce) {
                // A) Intersect with the scene
                float closest_t = FLT_MAX;
                size_t hitEntity = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);
                float betaContribution = 0.0f;

                GPUDataOutput::QuadraticInfo quadraticInfo{};
                // check intersection with geometry
                bool hit = geometryIntersectionQuadric(gaussianID, rayOrigin, rayDir, hitEntity, closest_t,
                                                       hitPointWorld,
                                                       hitNormalWorld, betaContribution, quadraticInfo);

                // If we hit some geometry then calculate the bounce
                if (hit) {
                    // Fetch material parameters
                    const QuadricInputAssembly &hitGaussianEntity = m_gpuData.quadricInputAssembly[hitEntity];
                    float color = hitGaussianEntity.color.x;
                    float specular = hitGaussianEntity.specular; // Specular coefficient
                    float shininess = hitGaussianEntity.phongExponent;
                    float diffuse = hitGaussianEntity.diffuse; // Diffuse coefficient
                    float contributionRayContribution = 0.0f;
                    float brdfFactor = 1.0f;


                    // ----------------------------------
                    // NON-METALLIC branch (diffuse > 0)
                    // ----------------------------------
                    // 1) Diffuse contribution
                    // Compute cosTheta for the diffuse term

                    // 2) Specular contribution
                    // TODO also calculate the contribution if I am sampling directly towards the camera instead of just reflecting along the surface normal
                    // Contribution Dir

                    {
                        glm::vec3 a(0.0f);
                        glm::vec3 contributionRayDir = sampleDirectionTowardAperture(
                            hitPointWorld,
                            m_cameraTransform->getPosition(), // center of aperture
                            cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
                            a,
                            apertureRadius,
                            photonID
                        );

                        float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                        cosTheta = glm::max(0.0f, cosTheta); // Clamp to 0 to prevent negative contributions
                        float diffuseContribution = cosTheta * color / M_PIf;


                        glm::vec3 delta = -rayDir + contributionRayDir;
                        glm::vec3 halfVector = delta / glm::length(delta);
                        float cosAlpha = glm::dot(hitNormalWorld, halfVector);
                        float cosAlphaMax = std::max(cosAlpha, 0.0f);
                        float specularContribution = std::pow(cosAlphaMax, shininess) / M_PIf;

                        float sumForWeights = diffuse + specular;
                        if (sumForWeights > 0.0f) {
                            float diffuseWeight = diffuse / sumForWeights;
                            float specularWeight = specular / sumForWeights;
                            // Weighted sum
                            contributionRayContribution = diffuseWeight * diffuseContribution
                                                          + specularWeight * specularContribution;
                        }
                    }

                    // Sample new random outgoing direction
                    glm::vec3 newDir = sampleCosineWeightedHemisphere(hitNormalWorld, photonID); {
                        float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                        cosTheta = glm::max(0.0f, cosTheta); // Clamp to 0 to prevent negative contributions
                        float diffuseContribution = cosTheta * color / M_PIf;


                        glm::vec3 delta = -rayDir + newDir;
                        glm::vec3 halfVector = delta / glm::length(delta);
                        float cosAlpha = glm::dot(hitNormalWorld, halfVector);
                        float cosAlphaMax = std::max(cosAlpha, 0.0f);
                        float specularContribution = std::pow(cosAlphaMax, shininess) / M_PIf;

                        float sumForWeights = diffuse + specular;
                        if (sumForWeights > 0.0f) {
                            float diffuseWeight = diffuse / sumForWeights;
                            float specularWeight = specular / sumForWeights;
                            // Weighted sum
                            brdfFactor = diffuseWeight * diffuseContribution
                                         + specularWeight * specularContribution;
                        }
                    }

                    float contributionFlux = photonFlux;
                    if (m_gpuData.renderInformation->applyBetaWeight)
                        contributionFlux *= betaContribution;


                    // Sample new direction (Lambertian reflection)
                    glm::vec3 newRayOrigin = hitPointWorld + hitNormalWorld * 1e-3f;
                    // Offset to prevent self-intersection
                    float offsetDist = glm::length(newRayOrigin - hitPointWorld);

                    glm::vec3 newDirectLightDir(0.0f);
                    float newCamera_t = 0.0f;
                    glm::vec3 newApertureHitPoint(0.0f);
                    glm::vec3 newCameraHitPointLocal(0.0f);
                    glm::vec2 pixelCoordinates(0.0f);
                    if (castContributionRay(hitPointWorld, cameraPlaneNormalWorld, apertureRadius, photonID,
                                            contributionFlux
                                            , newDirectLightDir, newApertureHitPoint, newCameraHitPointLocal,
                                            pixelCoordinates,
                                            newCamera_t
                    )) {
                        m_gpuDataOutput[photonID].bounce[bounce].hitCamera = true;
                        m_gpuDataOutput[photonID].bounce[bounce].apertureDirection = newDirectLightDir;
                        m_gpuDataOutput[photonID].bounce[bounce].emissionDirectionLength = closest_t;
                        m_gpuDataOutput[photonID].bounce[bounce].cameraDirectionLength = newCamera_t;
                        m_gpuDataOutput[photonID].bounce[bounce].apertureHitPoint = newApertureHitPoint;
                        m_gpuDataOutput[photonID].bounce[bounce].cameraHitPointLocal = newCameraHitPointLocal;
                        m_gpuDataOutput[photonID].bounce[bounce].pixelCoordinate = pixelCoordinates;
                        m_gpuDataOutput[photonID].bounce[bounce].quadInfo = quadraticInfo;
                    }

                    m_gpuDataOutput[photonID].bounce[bounce].hitPointWorld = hitPointWorld;
                    m_gpuDataOutput[photonID].bounce[bounce].hitNormalWorld = hitNormalWorld;
                    m_gpuDataOutput[photonID].bounce[bounce].outGoingOrigin = newRayOrigin;
                    m_gpuDataOutput[photonID].bounce[bounce].outGoingDirection = newRayOrigin;
                    m_gpuDataOutput[photonID].bounce[bounce].quadricID = hitEntity;

                    // Finally, scale the photonFlux (or outgoing radiance) by total contribution

                    photonFlux *= brdfFactor;

                    // Russian Roulette termination
                    float rrProb = photonFlux;
                    float minProbability = 0.2f; // 20%
                    float maxProbability = 0.9f; // 90%
                    rrProb = glm::clamp(rrProb, minProbability, maxProbability);
                    float rnd = m_rng[photonID].nextFloat();
                    if (rnd > rrProb) {
                        return; // Photon terminated i.e. absorbed by the last surface
                    }
                    photonFlux = photonFlux / rrProb;

                    //glm::vec3 newDir = sampleRandomDirection(photonID);
                    rayOrigin = newRayOrigin; // Offset to prevent self-intersection
                    rayDir = glm::normalize(newDir);
                } else {
                    return;
                }
            } // end for bounces

            // If we exit here, we used up all bounces w/o hitting sensor
        }

        bool castContributionRay(const glm::vec3 &directLightingOrigin, const glm::vec3 &cameraPlaneNormalWorld,
                                 float apertureRadius, size_t photonID, float photonFlux,
                                 glm::vec3 &directLightDir,
                                 glm::vec3 &apertureHitPoint,
                                 glm::vec3 &cameraHitPointLocal,
                                 glm::vec2 &pixelCoordinates,
                                 float &camera_t
        ) const {
            // Calculate direct lighting

            directLightDir = sampleDirectionTowardAperture(
                directLightingOrigin,
                m_cameraTransform->getPosition(), // center of aperture
                cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
                apertureHitPoint,
                apertureRadius,
                photonID
            );

            // Early exist if we are hitting the camera plane from behind, this happens if the aperture direction and camera plane normal are parallell or within that quadrant
            float direction = glm::dot(directLightDir, cameraPlaneNormalWorld);
            if (direction >= 0.0f) {
                return false;
            }
            // Check if contribution ray intersects geometry

            // Create contribution Rays and trace towards the camera
            // Trace our contribution ray
            glm::vec3 camHit(0.0f);
            float incidentAngle = 0.0f;
            bool cameraHit = checkCameraPlaneIntersection(directLightingOrigin, directLightDir, camHit,
                                                          camera_t, incidentAngle);
            if (cameraHit) {
                float closest_t = FLT_MAX;
                size_t hitEntity = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);
                size_t emissiveEntityID = 0;
                float betaContribution = 0.0f;
                // check intersection with geometry
                GPUDataOutput::QuadraticInfo quadraticInfo(0.0f);
                bool hit = geometryIntersectionQuadric(emissiveEntityID, directLightingOrigin, directLightDir,
                                                       hitEntity,
                                                       closest_t, hitPointWorld, hitNormalWorld, betaContribution,
                                                       quadraticInfo, true);

                float tGeom = hit ? glm::length(hitPointWorld - m_cameraTransform->getPosition()) : FLT_MAX;
                float tAperture = glm::length(directLightingOrigin - m_cameraTransform->getPosition());
                //float tGeom = hit ? closest_t : FLT_MAX;
                if (tAperture < tGeom) {
                    glm::vec3 cameraHitPointWorld = directLightingOrigin + directLightDir * camera_t;

                    glm::mat4 worldToCamera = glm::inverse(m_cameraTransform->getTransform());
                    glm::vec4 hitPointCam4 = worldToCamera * glm::vec4(cameraHitPointWorld, 1.0f);
                    cameraHitPointLocal = hitPointCam4 / hitPointCam4.w;

                    if (accumulateOnSensor(photonID, cameraHitPointLocal, photonFlux, pixelCoordinates)) {
                        return true;
                    }
                }
            }
            return false;
        }


        // Main function: BVH traversal version of geometryIntersectionQuadric.
        // Instead of iterating over all quadrics, we traverse the BVH stored in m_gpuData.bvhNodes.
        bool geometryIntersectionQuadric(
            size_t gaussianID,
            const glm::vec3 &rayOrigin,
            const glm::vec3 &rayDir,
            size_t &hitEntity,
            float &closest_t,
            glm::vec3 &hitPointWorld,
            glm::vec3 &hitNormalWorld,
            float &betaContribution,
            GPUDataOutput::QuadraticInfo &quadraticInfo,
            bool isContributionRay = false
        ) const {
            // Set up initial values.
            float tMinGlobal = std::numeric_limits<float>::max();
            bool hitFound = false;
            size_t bestQuadricIndex = 0;
            glm::vec3 bestHitPoint(0.0f), bestHitNormal(0.0f);
            float bestBeta = 0.0f;
            // Set up an iterative traversal stack.
            const int MAX_STACK_SIZE = 64;
            int stack[MAX_STACK_SIZE];
            int stackPtr = 0;
            // Push the BVH root index (assumed 0) onto the stack.
            stack[stackPtr++] = m_gpuData.numBVHNodes - 1;

            // Traverse the BVH iteratively.
            while (stackPtr > 0) {
                int currentIndex = stack[--stackPtr];
                const BVHNode &node = m_gpuData.bvhNodes[currentIndex];

                // Test ray against node's bounding box.
                if (!rayAABBIntersect(rayOrigin, rayDir, node.bboxMin, node.bboxMax, tMinGlobal))
                    continue;

                if (node.isLeaf) {
                    // Leaf node: perform the detailed quadric intersection test.
                    float tCandidate = std::numeric_limits<float>::max();
                    glm::vec3 localHitPoint(0.0f), localHitNormal(0.0f);
                    float beta = 0.0f;
                    const QuadricInputAssembly &quadric = m_gpuData.quadricInputAssembly[node.quadricIndex];
                    if (isContributionRay) {
                        if (checkContributionCollision(rayOrigin, rayDir, quadric, localHitPoint)) {
                            hitFound = true;
                            bestHitPoint = localHitPoint;
                        }
                    } else {
                        if (intersectQuadricLeaf(rayOrigin, rayDir, quadric, tCandidate, localHitPoint, localHitNormal,
                                                 beta, quadraticInfo)) {
                            if (tCandidate < tMinGlobal) {
                                tMinGlobal = tCandidate;
                                bestQuadricIndex = node.quadricIndex;
                                bestHitPoint = localHitPoint;
                                bestHitNormal = localHitNormal;
                                bestBeta = beta;
                                hitFound = true;
                            }
                        }
                    }
                } else {
                    // Internal node: push its child nodes onto the stack.
                    if (stackPtr + 2 < MAX_STACK_SIZE) {
                        stack[stackPtr++] = node.leftChild;
                        stack[stackPtr++] = node.rightChild;
                    }
                }
            }

            // If a hit was found, update the output parameters.
            if (hitFound) {
                hitEntity = bestQuadricIndex;
                closest_t = tMinGlobal;
                hitPointWorld = bestHitPoint;
                hitNormalWorld = bestHitNormal;
                betaContribution = bestBeta;
                return true;
            }
            return false;
        }

        bool geometryIntersection2DGS(
            size_t gaussianID,
            const glm::vec3 &rayOrigin,
            const glm::vec3 &rayDir,
            size_t &hitPointIdx,
            float &closest_t,
            glm::vec3 &hitPointWorld,
            glm::vec3 &hitNormalWorld
        ) const {
            bool hit = false;
            float epsilon = 1e-6f;

            for (uint32_t i = 0; i < m_gpuData.numGaussians; ++i) {
                if (i == gaussianID)
                    continue;

                const GaussianInputAssembly &gp = m_gpuData.gaussianInputAssembly[i];
                glm::vec3 N = gp.normal; // plane normal
                float denom = glm::dot(N, rayDir);

                if (fabs(denom) < epsilon) {
                    continue; // nearly parallel => no valid intersection
                }

                float t = glm::dot((gp.position - rayOrigin), N) / denom;
                if (t < epsilon) {
                    continue; // intersection behind the origin
                }

                glm::vec3 p = rayOrigin + t * rayDir;
                float dist = glm::distance(rayOrigin, p);
                if (dist >= closest_t) {
                    continue; // not closer than current intersection
                }

                // Build local tangent basis for the plane
                glm::vec3 tAxis, bAxis;
                buildTangentBasis(N, tAxis, bAxis);

                // Vector in plane coords
                glm::vec3 vPlane = p - gp.position;
                float u = glm::dot(vPlane, tAxis);
                float v = glm::dot(vPlane, bAxis);

                // Check elliptical boundary:
                float sigmaU = gp.scale.x;
                float sigmaV = gp.scale.y;
                float ellipseParam = (u * u) / (sigmaU * sigmaU)
                                     + (v * v) / (sigmaV * sigmaV);

                if (ellipseParam > 1.0f) {
                    // Outside the ellipse => ignore this intersection
                    continue;
                }


                // If we get here, we have a valid intersection
                closest_t = dist;
                hitPointIdx = i;
                hitPointWorld = p;
                hitNormalWorld = N;
                hit = true;
            }

            return hit;
        }


        glm::vec3 sampleDirectionTowardAperture(
            const glm::vec3 &lightPos,
            const glm::vec3 &apertureCenter,
            const glm::vec3 &apertureNormal,
            glm::vec3 &apertureHitpoint,
            float apertureRadius,
            uint64_t photonID) const {
            // pick random point on the lens
            apertureHitpoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
            // direction from light to lens point
            glm::vec3 dir = apertureHitpoint - lightPos;
            return normalize(dir);
        }

        glm::vec3 samplePointOnDisk(size_t photonID,
                                    const glm::vec3 &center,
                                    const glm::vec3 &normal,
                                    float radius) const {
            // Or use any 2D disk sampling approach (e.g., concentric disk sampling).
            // We'll do a simple naive approach:
            float r = radius * sqrt(m_rng[photonID].nextFloat());
            float theta = 2.f * M_PI * m_rng[photonID].nextFloat();


            glm::vec3 refVec = sampleRandomDirection(photonID);
            // Ensure refVec is not parallel or nearly parallel to normal
            if (abs(dot(normal, refVec)) > 0.999f) {
                refVec = glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)); // Use a fixed backup vector
            }
            // Construct orthonormal basis for the disk plane
            glm::vec3 u = normalize(cross(normal, refVec));
            glm::vec3 v = cross(normal, u);

            float dx = r * cos(theta);
            float dy = r * sin(theta);

            glm::vec3 offset = dx * u + dy * v;
            return center + offset;
        }

        bool checkCameraPlaneIntersection(
            const glm::vec3 &rayOriginWorld,
            const glm::vec3 &rayDirWorld,
            glm::vec3 &hitPointCam, // out: intersection in camera space
            float &tIntersect, // out: parameter t
            float &contributionScore // out: parameter contributionScore
        ) const {
            // 1) Transform to camera space

            glm::mat4 entityTransform = m_cameraTransform->getTransform();
            glm::mat3 entityRotation = glm::mat3(entityTransform);
            // Camera plane normal in world space
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(entityRotation * glm::vec3(0.0f, 0.0f, -1.0f));

            glm::vec4 cameraPlanePointWorld4 = entityTransform * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

            glm::vec3 cameraPlanePointWorld = glm::vec3(cameraPlanePointWorld4 / cameraPlanePointWorld4.w);
            // Ray-plane intersection

            // Ray-plane intersection calculation
            float denom = glm::dot(cameraPlaneNormalWorld, rayDirWorld);
            if (std::abs(denom) < 1e-6) {
                return false; // Ray is parallel to the plane
            }
            glm::vec3 p0l0 = cameraPlanePointWorld - rayOriginWorld;
            float t = glm::dot(p0l0, cameraPlaneNormalWorld) / denom;
            if (t < 1e-6) {
                return false; // Intersection is behind the ray origin
            }
            glm::vec3 intersectionPoint = rayOriginWorld + t * rayDirWorld;

            float det = glm::determinant(entityTransform);
            if (fabs(det) < 1e-6f) {
                return false;
            };

            glm::mat4 view = glm::inverse(entityTransform);
            glm::vec4 intersectionPointCamera = view * glm::vec4(intersectionPoint, 1.0f);
            glm::vec3 intersectionCamSpace = glm::vec3(intersectionPointCamera) / intersectionPointCamera.w;

            // Sensor plane bounds in camera space
            float halfW = (m_camera->parameters().width * 0.5f) / m_camera->parameters().fx;
            float halfH = (m_camera->parameters().height * 0.5f) / m_camera->parameters().fy;

            // Check bounds
            if (intersectionCamSpace.x < -halfW || intersectionCamSpace.x > halfW ||
                intersectionCamSpace.y < -halfH || intersectionCamSpace.y > halfH) {
                return false; // Outside sensor bounds
            }

            float cosAngle = std::abs(glm::dot(cameraPlaneNormalWorld, rayDirWorld)); // Ensure positive cosine
            contributionScore = cosAngle; // Higher cosine means closer to perpendicular


            // If we reach here, the ray intersects the camera plane within bounds
            hitPointCam = intersectionPoint; // Intersection point in camera space
            tIntersect = t; // Distance along the ray to the intersection
            return true;
        }


        // ---------------------------------------------------------------------
        //  accumulateOnSensor
        // ---------------------------------------------------------------------
        bool accumulateOnSensor(size_t photonID, const glm::vec3 &hitPointCam, float photonFlux,
                                glm::vec2 &pixelCoordinatesOut) const {
            // 2. Project to the image plane using pinhole intrinsics:
            // Important: Z_cam should be > 0 for a point in front of the camera.
            //

            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float X = hitPointCam.x;
            float Y = hitPointCam.y;
            float Z = hitPointCam.z;
            float xPixel = (fx * X / Z) + cx;
            float yPixel = (fy * Y / Z) + cy;

            pixelCoordinatesOut.x = xPixel;
            pixelCoordinatesOut.y = yPixel;

            // 5. Retrieve image dimensions.
            const size_t imageWidth = m_camera->parameters().width;
            const size_t imageHeight = m_camera->parameters().height;

            if (xPixel < 0 && xPixel > static_cast<int>(imageWidth) &&
                yPixel < 0 && yPixel > static_cast<int>(imageHeight)) {
                return false;
            }
            // 2. Determine the neighboring pixels.
            // Compute the lower-left (floor) pixel coordinates and the fractional offsets.
            int x0 = static_cast<int>(std::floor(xPixel));
            int y0 = static_cast<int>(std::floor(yPixel));
            float dx = xPixel - x0;
            float dy = yPixel - y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // 3. Compute the bilinear weights for the 4 pixels.
            float w00 = (1.0f - dx) * (1.0f - dy); // weight for pixel at (x0, y0)
            float w10 = dx * (1.0f - dy); // weight for pixel at (x1, y0)
            float w01 = (1.0f - dx) * dy; // weight for pixel at (x0, y1)
            float w11 = dx * dy; // weight for pixel at (x1, y1)

            // 4. Pre-correct the photon flux with gamma adjustment.
            float correctedFlux = std::pow(photonFlux, 1.0f / m_gpuData.renderInformation->gamma);


            bool hitSensor = false;
            // Helper lambda to add the weighted flux contribution to a given pixel.
            auto addFluxToPixel = [&](int px, int py, float weight) {
                // Only update if the pixel is inside the image bounds.
                if (px >= 0 && px < static_cast<int>(imageWidth) &&
                    py >= 0 && py < static_cast<int>(imageHeight)) {
                    size_t pixelIndex = static_cast<size_t>(py) * imageWidth + static_cast<size_t>(px);
                    float fluxToAdd = weight * correctedFlux;

                    // Use atomic operations to safely update the pixel value.
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            imageMemoryAtomic(m_gpuData.imageMemory[pixelIndex]);
                    // Use atomic operations to safely update the pixel value.
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            imageMemoryCounterAtomic(m_gpuData.imageMemoryCounter[pixelIndex]);

                    imageMemoryCounterAtomic.fetch_add(1.0f);

                    // Optionally, prevent saturation by clamping the pixel value to 1.0f.
                    //float currentValue = imageMemoryAtomic.load();
                    //float newValue = std::min(1.0f, currentValue + fluxToAdd);
                    //fluxToAdd = newValue - currentValue; // Adjust flux to the remaining margin.
                    imageMemoryAtomic.fetch_add(fluxToAdd);

                    hitSensor = true;
                    // 7. Atomically update the photon count.
                    /*
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            photonsAccumulatedAtomic(m_gpuData.renderInformation->photonsAccumulated);
                    photonsAccumulatedAtomic.fetch_add(static_cast<uint64_t>(1));
                    */
                }
            };
            // 6. Distribute the corrected flux into the four neighboring pixels.
            addFluxToPixel(x0, y0, w00);
            addFluxToPixel(x1, y0, w10);
            addFluxToPixel(x0, y1, w01);
            addFluxToPixel(x1, y1, w11);

            return hitSensor;
        }

        // ---------------------------------------------------------------------
        //  Helper: sample an emissive gaussian object
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveGaussian(size_t photonID) const {
            std::array<size_t, 10> emissiveIndices{}; // max 10 emissive light sources supported
            size_t count = 0;
            for (size_t entityIdx = 0; entityIdx < m_gpuData.numGaussians; ++entityIdx) {
                if (m_gpuData.gaussianInputAssembly[entityIdx].emission > 0.f) {
                    if (count < emissiveIndices.size()) {
                        // ensure we don't write out-of-bounds
                        emissiveIndices[count++] = entityIdx;
                    }
                }
            }

            size_t emissiveIndex = 0;
            if (count == 0) {
                // No emissive gaussian found; handle this error case as needed.
                emissiveIndex = 0;
                return 0;
            }

            size_t randomChoice = m_rng[photonID].nextUInt() % count;
            emissiveIndex = emissiveIndices[randomChoice];
            return emissiveIndex;
        }

        // ---------------------------------------------------------------------
        //  sampleEmissiveDirectionAndPower
        // ---------------------------------------------------------------------
        void sampleEmissiveDirectionAndPower(size_t emissiveEntityIdx,
                                             size_t photonID,
                                             glm::vec3 &outPos,
                                             glm::vec3 &outNormal,
                                             float &emissionPower) const {
            const GaussianInputAssembly &gaussian = m_gpuData.gaussianInputAssembly[emissiveEntityIdx];
            // ------------------------------------------------------------------
            // 1. Prepare the normal, find two tangent vectors for the plane.
            // ------------------------------------------------------------------
            glm::vec3 n = glm::normalize(gaussian.normal);
            glm::vec3 t1(0.0f);
            glm::vec3 t2(0.0f);
            buildTangentBasis(n, t1, t2);
            // 2. Repeatedly draw samples from a standard normal, then scale
            //    them by (sigma_x, sigma_y), until they fall inside the ellipse.

            float x = 0.0f, y = 0.0f; // final offsets in local 2D coords

            // (a) Generate two uniform randoms in [0,1)
            float u1 = m_rng[photonID].nextFloat();
            float u2 = m_rng[photonID].nextFloat();

            // (b) Box-Muller transform for standard normal
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * M_PIf * u2;
            float z0 = r * cosf(theta); // ~ N(0,1)
            float z1 = r * sinf(theta); // ~ N(0,1)

            // (c) Scale by anisotropic stddev (sigma_x, sigma_y)
            x = z0 * gaussian.scale.x;
            y = z1 * gaussian.scale.y;

            // ------------------------------------------------------------------
            // 3. Offset the center by (x, y) in the plane spanned by (t1, t2).
            // ------------------------------------------------------------------
            glm::vec3 offset = x * t1 + y * t2;
            outPos = gaussian.position;

            // Normal remains the same as the Gaussian's normal
            outNormal = n;

            // Emission power (or flux) from your stored value
            // ------------------------------------------------------------------
            // 4. Compute Emission Power per Sample
            // ------------------------------------------------------------------
            // Total emission power from the Gaussian
            float P_total = gaussian.emission;

            // Compute the Gaussian PDF at the sampled (x, y)
            float sigma_x = gaussian.scale.x;
            float sigma_y = gaussian.scale.y;

            // Gaussian PDF (unnormalized since we are within the ellipse)
            float gaussianPDF = (1.0f / (2.0f * M_PIf * sigma_x * sigma_y)) *
                                expf(-0.5f * ((x * x) / (sigma_x * sigma_x) + (y * y) / (sigma_y * sigma_y)));

            // Area of the ellipse
            float ellipseArea = M_PIf * sigma_x * sigma_y;

            // Since we are using rejection sampling, the samples are uniformly distributed over the ellipse
            // Probability density of the uniform distribution over the ellipse
            float uniformPDF = 1.0f / ellipseArea;

            // Weight for the sample based on the ratio of Gaussian PDF to uniform PDF
            // This ensures that emissionPower reflects the importance of the sample
            float weight = gaussianPDF / uniformPDF; // = (1 / (2πσxσy)) * exp(...) / (1 / πσxσy) = 0.5 * exp(...)

            // Emission power per sample
            // Distribute P_total across samples based on weight
            emissionPower = P_total * weight;
        }


        // ---------------------------------------------------------------------
        //  randomUnitVector using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 randomUnitVector(size_t photonID) const {
            float theta = m_rng[photonID].nextFloat() * 2.0f * M_PI; // [0, 2π)
            float z = m_rng[photonID].nextFloat() * 2.0f - 1.0f; // [-1, 1)
            float r = sqrtf(1.0f - z * z); // Radius at z

            float x = r * cosf(theta);
            float y = r * sinf(theta);

            return glm::vec3(x, y, z); // Already normalized
        }


        // Constructs an orthonormal basis (T, B, N) given a normal N.
        static void buildTangentBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B) {
            // Any vector not collinear with N will do for "temp"
            glm::vec3 temp = (fabs(N.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);

            T = glm::normalize(glm::cross(temp, N));
            B = glm::cross(N, T);
            // Now T, B, N is an orthonormal basis
        }

        glm::vec3 sampleCosineWeightedHemisphere(
            const glm::vec3 &normal,
            size_t photonID) const {
            // Step 1: Generate two uniform random numbers in [0,1]
            float u1 = m_rng[photonID].nextFloat();
            float u2 = m_rng[photonID].nextFloat();

            // Step 2: Convert to polar coordinates for cosine-weighted distribution
            float r = std::sqrt(u1);
            float phi = 2.0f * M_PIf * u2;

            // Step 3: Compute the sample in local coordinates (where z is up)
            float x = r * std::cos(phi);
            float y = r * std::sin(phi);
            float z = std::sqrt(1.0f - u1);

            // Step 4: Build a local orthonormal basis around 'normal'
            glm::vec3 t(0.0f), b(0.0f);
            buildTangentBasis(normal, t, b);

            // Step 5: Transform the local sample to world space
            glm::vec3 sampleWorld = x * t + y * b + z * normal;
            return glm::normalize(sampleWorld);
        }


        // ---------------------------------------------------------------------
        //  sampleRandomHemisphere (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomHemisphere(const glm::vec3 &normal, size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            if (glm::dot(r, normal) < 0.f) {
                r = -r;
            }
            return glm::normalize(r);
        }

        // ---------------------------------------------------------------------
        //  sampleRandomDirection (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomDirection(size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            return glm::normalize(r);
        }
    };
}


#endif //LightTracerKernel
