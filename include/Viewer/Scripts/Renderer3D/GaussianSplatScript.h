//
// Created by magnus on 10/2/23.
//

#ifndef GAUSSIAN_SPLAT_SCRIPT
#define GAUSSIAN_SPLAT_SCRIPT

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/Core/RenderResource.h"


class GaussianSplatScript : public VkRender::Base, public VkRender::RegisteredInFactory<GaussianSplatScript> {
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    GaussianSplatScript() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }

    /** @brief Static method to create instance of this class, returns a unique ptr of GaussianSplatScript **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<GaussianSplatScript>(); }

    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "GaussianSplatScript"; }

    /** @brief Setup function called one during script creating prepare **/
    void setup() override;

    /** @brief update function called once per frame **/
    void update() override;

    /** @brief destroy function called before script deletion **/
    void onDestroy() override;

    /** @brief set if this script should be drawn or not. */
    void setDrawMethod(VkRender::CRL_SCRIPT_DRAW_METHOD _drawMethod) override { this->drawMethod = _drawMethod; }

    /** @brief draw function called once per frame **/
    void draw(CommandBuffer *commandBuffer, uint32_t i, bool b) override;

    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    VkRender::ScriptTypeFlags getType() override { return type; }

    VkRender::CRL_SCRIPT_DRAW_METHOD getDrawMethod() override { return drawMethod; }

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    VkRender::ScriptTypeFlags type = VkRender::CRL_SCRIPT_TYPE_DISABLED;
    VkRender::CRL_SCRIPT_DRAW_METHOD drawMethod = VkRender::CRL_SCRIPT_DONT_DRAW;

    VkRender::UBOMatrix mvpMat{};
    std::vector<Buffer> uniformBuffers;
    std::vector<Texture2D> textures;

    std::unique_ptr<RenderResource::Mesh> mesh;
    std::unique_ptr<RenderResource::Pipeline> pipeline;

    char filePathDialog[1024] = "C:\\Users\\mgjer\\Downloads\\models\\stump";
    std::future<std::string> plyFileFolder;
    std::vector<void*> handles;
    bool renderGaussians = true;

    float scaleModifier = 1.0f;
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 up = glm::vec3(0.0f, -1.0f, 0.0f);
    uint32_t cudaRequestedMemorySize = 0;

    //std::unique_ptr<GaussianSplat> splat;

    void createCudaVkImage(uint32_t width, uint32_t height);
};


#endif //GAUSSIAN_SPLAT_SCRIPT
