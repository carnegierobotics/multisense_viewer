//
// Created by magnus on 6/27/22.
//


#include <execution>
#include "DecodeVideo.h"


void DecodeVideo::setup(CameraConnection *camHandle) {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, CrlImage);
    // Don't draw it before we create the texture in update()

    model->draw = false;
    this->camHandle = camHandle;


}


void DecodeVideo::update() {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    if (model->draw){
        crl::multisense::image::Header stream;
        camHandle->camPtr->getCameraStream(&stream);
        model->setColorTexture(stream);

        free((void *) stream.imageDataP);
    }

    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(-1.3, 0.4, -5));

    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;
}

void DecodeVideo::prepareTextureAfterDecode() {
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;
    vertexShaderFileName = "myScene/spv/quad.vert";
    fragmentShaderFileName = "myScene/spv/quad.frag";

    auto inf = camHandle->camPtr->getCameraInfo();

    width = inf.imgConf.width();
    height = inf.imgConf.height();

    model->prepareTextureImage(width, height, CrlColorImageYUV420);
    auto *imgData = new ImageData(((float) width / (float) height), 1);


    // Load shaders
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    // Create quad and store it locally on the GPU
    model->createMeshDeviceLocal((ArEngine::Vertex *) imgData->quad.vertices,
                                 imgData->quad.vertexCount, imgData->quad.indices, imgData->quad.indexCount);

    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

}

void DecodeVideo::onUIUpdate(GuiObjectHandles uiHandle) {
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        if (dev.streams.find(AR_PREVIEW_VIRTUAL) == dev.streams.end())
            continue;

        src = dev.streams.find(AR_PREVIEW_VIRTUAL)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_VIRTUAL)->second.playbackStatus;


    }

    if (playbackSate == AR_PREVIEW_PLAYING) {
        for (auto &dev: *uiHandle.devices) {
            if (dev.cameraName == "Virtual Camera" && !model->draw) {

                camHandle->camPtr->start(src + ".mp4", " ");
                prepareTextureAfterDecode();
                model->draw = true;

            }
        }

    } else if (playbackSate == AR_PREVIEW_STOPPED && model->draw == true) {
        model->draw = false;
        camHandle->camPtr->stop("");

    }

}


void DecodeVideo::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw)
        CRLCameraModels::draw(commandBuffer, i, model);
}

/*

int DecodeVideo::childProcessDecode() {
// thread declaration
    int N = 5;

    // Declaration of attribute......
    pthread_attr_t attr;

    // semaphore initialization
    sem_init(&notEmpty, 0, 0);
    sem_init(&notFull, 0, N);

    // pthread_attr_t initialization
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,
                                PTHREAD_CREATE_JOINABLE);

    // Creation of process
    r1 = pthread_create(&producer, &attr, DecodeVideo::decode, this);
    if (r1) {
        std::cout <<
                  "Error in creating thread" << std::endl;
        exit(-1);
    }
    return EXIT_SUCCESS;
}
/*
void *DecodeVideo::decode(void *arg) {
    auto *instance = (DecodeVideo *) arg;
    while (instance->runDecodeThread) {


        AVFormatContext *ctx_format = nullptr;
        AVCodecContext *ctx_codec = nullptr;
        AVCodec *codec = nullptr;
        AVFrame *frame = av_frame_alloc();
        int stream_idx;
        SwsContext *ctx_sws = nullptr;
        std::string fileName = Utils::getTexturePath() + "Video/pixels.mp4";
        AVStream *vid_stream = nullptr;
        AVPacket *pkt = av_packet_alloc();

        av_register_all();

        if (int ret = avformat_open_input(&ctx_format, fileName.c_str(), nullptr, nullptr) != 0) {
            std::cout << 1 << std::endl;

        }
        if (avformat_find_stream_info(ctx_format, nullptr) < 0) {
            std::cout << 2 << std::endl;

        }
        av_dump_format(ctx_format, 0, fileName.c_str(), false);

        for (int i = 0; i < ctx_format->nb_streams; i++)
            if (ctx_format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                stream_idx = i;
                vid_stream = ctx_format->streams[i];
                break;
            }
        if (vid_stream == nullptr) {
            std::cout << 4 << std::endl;
        }

        codec = avcodec_find_decoder(vid_stream->codecpar->codec_id);
        if (!codec) {
            fprintf(stderr, "codec not found\n");
            exit(1);
        }
        ctx_codec = avcodec_alloc_context3(codec);

        if (avcodec_parameters_to_context(ctx_codec, vid_stream->codecpar) < 0)
            std::cout << 512;
        if (avcodec_open2(ctx_codec, codec, nullptr) < 0) {
            std::cout << 5;
        }

        instance->width = ctx_codec->width;
        instance->height = ctx_codec->height;

        //av_new_packet(pkt, pic_size);
        int index = 0;

        while (av_read_frame(ctx_format, pkt) >= 0 && instance->runDecodeThread) {
            if (pkt->stream_index == stream_idx) {
                int ret = avcodec_send_packet(ctx_codec, pkt);
                if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    std::cout << "avcodec_send_packet: " << ret << std::endl;
                    break;
                }
                while (ret >= 0 && instance->runDecodeThread) {
                    ret = avcodec_receive_frame(ctx_codec, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        //std::cout << "avcodec_receive_frame: " << ret << std::endl;
                        break;
                    }

                    sem_wait(&instance->notFull);
                    usleep(33);
                    std::cout <<
                              "Producer produces item.Items Present = "
                              << ++instance->items << std::endl;
                    //std::cout << "frame: " << ctx_codec->frame_number << std::endl;
                    instance->videoFrame[index] = *frame;
                    instance->bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, ctx_codec->width,
                                                                    ctx_codec->height, 1);
                    index++;
                    if (index == 5)
                        index = 0;
                    sem_post(&instance->notEmpty);

                    //saveFrameYUV420P(frame, frame->width, frame->height, ctx_codec->frame_number);

                }
            }
            av_packet_unref(pkt);
        }


        avformat_close_input(&ctx_format);
        av_packet_unref(pkt);
        avcodec_free_context(&ctx_codec);
        avformat_free_context(ctx_format);
    }

    pthread_exit((void *) (intptr_t) EXIT_SUCCESS);
}
 */

void DecodeVideo::saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame) {
    FILE *pFile;
    char szFilename[32];
    int y;

    // Open file
    sprintf(szFilename, "frame%d.yuv", iFrame);
    pFile = fopen(szFilename, "wb");
    if (pFile == nullptr)
        return;

    // Write pixel data
    fwrite(pFrame->data[0], 1, width * height, pFile);
    fwrite(pFrame->data[1], 1, width * height / 4, pFile);
    fwrite(pFrame->data[2], 1, width * height / 4, pFile);

    // Close file
    fclose(pFile);
}