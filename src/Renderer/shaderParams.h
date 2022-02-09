//
// Created by magnus on 12/8/21.
//

#ifndef AR_ENGINE_SHADERPARAMS_H
#define AR_ENGINE_SHADERPARAMS_H


#include <glm/vec4.hpp>
#include <glm/ext/matrix_float4x4.hpp>

struct FragShaderParams {
    glm::vec4 lightColor;
    glm::vec4 objectColor;
    glm::vec4 lightPos;
    glm::vec4 viewPos;
};


/*
struct ShaderValuesParams {
    glm::vec4 lightDir{};
    float exposure = 10.5f;
    float gamma = 2.2f;
    float prefilteredCubeMipLevels = 10.0f;
    float scaleIBLAmbient = 1.0f;
    float debugViewInputs = 0;
    float debugViewEquation = 0;
};
*/
struct UBOMatrix {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
};



#endif //AR_ENGINE_SHADERPARAMS_H
