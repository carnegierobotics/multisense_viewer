//
// Created by magnus on 2/21/22.
//

#include "CRLVirtualCamera.h"

extern "C" {
#include<libavutil/avutil.h>
#include<libavutil/imgutils.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>

}

#ifdef WIN32
#else

#include<bits/stdc++.h>
#include<pthread.h>
#include<semaphore.h>

#endif

#include <MultiSense/src/tools/Utils.h>
#include <thread>
#include <MultiSense/src/tools/Logger.h>


bool CRLVirtualCamera::connect(const std::string &ip) {

    return true;
}

void CRLVirtualCamera::start(std::string string, std::string dataSourceStr) {


}

void CRLVirtualCamera::start(std::string src, StreamIndex parent) {

    auto it = parentKeyMap.find(parent);
    if (it != parentKeyMap.end()) {
        Log::Logger::getInstance()->info("Tried to start decode stream with an already used parent key");
        return;
    }

    parentKeyMap[parent] = (key % MAX_STREAMS);


    container[parentKeyMap[parent]].videoFrame = new AVFrame[20]; // TODO Optimized approach is to use a sliding window to store frames and discard drawn
    container[parentKeyMap[parent]].pauseThread = false;
    container[parentKeyMap[parent]].videoName = src;
    getVideoMetadata(parentKeyMap[parent]);
    Log::Logger::getInstance()->info("Started virtual stream. Parent {} got Key {}", (uint32_t) parent,
                                     (key % MAX_STREAMS));
    if (!container[parentKeyMap[parent]].runDecodeThread) {
        container[parentKeyMap[parent]].runDecodeThread = true;
        childProcessDecode(parentKeyMap[parent]);
    }

    updateCameraInfo(parentKeyMap[parent]);
    key++;
}


void CRLVirtualCamera::stop(std::string dataSourceStr) {


}

void CRLVirtualCamera::stop(StreamIndex parent) {
    auto it = parentKeyMap.find(parent);
    if (it == parentKeyMap.end()) {
        Log::Logger::getInstance()->info(
                "Tried to stop decode stream with a parent key that was not used to start the stream");
        return;
    }

    uint32_t idx = parentKeyMap[parent];

    container[idx].pauseThread = true;
    container[idx].runDecodeThread = false;
    delete[] container[idx].videoFrame;
    container[idx].lastFrame = 1;
    container[idx].idx = 0;
    container[idx].idxFrame = 0;
    container[idx].frameIDs = std::array<uint32_t, 20>{};

    sem_post(&container[idx].notFull);

    Log::Logger::getInstance()->info("Joining producer thread for video decoder. Parent {} using key {}",
                                     (uint32_t) parent, parentKeyMap[parent]);
    pthread_join(container[idx].producer, nullptr);

    size_t ret = parentKeyMap.erase(parent);

    Log::Logger::getInstance()->info("Erased {}", ret);
}

void CRLVirtualCamera::updateCameraInfo() {
    // Just populating it with some hardcoded data
    // - DevInfo
    /*
    info.devInfo.name = "CRL Virtual Camera";
    info.devInfo.imagerName = "Virtual";
    info.devInfo.serialNumber = "25.8069758011"; // Root of all evil
    // - getImageCalibration
    info.netConfig.ipv4Address = "Knock knock";

     */
}


void CRLVirtualCamera::updateCameraInfo(uint32_t index) {
    // Just populating it with some hardcoded data
    // - DevInfo

    info.devInfo.name = "CRL Virtual Camera";
    info.devInfo.imagerName = "Virtual";
    info.devInfo.serialNumber = "25.8069758011"; // Root of all evil
    info.netConfig.ipv4Address = "Knock knock";
    info.imgConf.setWidth(container[index].width);
    info.imgConf.setHeight(container[index].height);

}

void CRLVirtualCamera::update() {


}

void CRLVirtualCamera::preparePointCloud(uint32_t width, uint32_t height) {
    //this->width = width;
    //this->height = height;

    float fx = width / 2;
    float fy = height / 2;
    float cx = width / 2;
    float cy = height / 2;

    kInverseMatrix =
            glm::mat4(
                    glm::vec4(1 / fx, 0, -(cx * fx) / (fx * fy), 0),
                    glm::vec4(0, 1 / fy, -(cy) / fy, 0),
                    glm::vec4(0, 0, 1, 0),
                    glm::vec4(0, 0, 0, 1));

    kInverseMatrix = glm::transpose(kInverseMatrix);

    info.kInverseMatrix = kInverseMatrix;
}


bool CRLVirtualCamera::getCameraStream(ArEngine::MP4Frame *frame, StreamIndex parent) {
    uint32_t index = parentKeyMap[parent];

    container[index].pauseThread = false;
    assert(frame != nullptr);

#ifdef WIN32 // TODO USE MACROS INSTEAD AND DEFINE MACROS DEPENDING ON PLATFORM
    if (!decoded) {
        SetEvent(notFull);
        return false;
    }

    /*
    DWORD dwWaitResult = WaitForSingleObject(
        notEmpty,   // handle to semaphore // zero-second time-out interval
        INFINITE);
        */
    WaitForSingleObject(notEmpty, INFINITE);

#else
    sem_wait(&container[index].notEmpty);
#endif

/*
    frame->plane0Size = videoFrame[0].linesize[0] * videoFrame[0].height;
    frame->plane1Size = videoFrame[0].linesize[1] * videoFrame[0].height;
    frame->plane2Size = videoFrame[0].linesize[2] * videoFrame[0].height;
 */
    bool found = false;
    for (int i = 0; i < container[index].frameIDs.size(); ++i) {
        uint32_t val = container[index].frameIDs[i];
        uint32_t searchVal = container[index].lastFrame % 20;
        if (val == searchVal) {
            container[index].idx = i;
            container[index].lastFrame++;
            found = true;
        }
    }
    if (!found) {
        sem_post(&container[index].notFull);
        return false;
    }

    // TODO Please optimize. Painful to see this
    uint32_t w = container[index].videoFrame[container[index].idx].width;
    if (w != container[index].width) {
        sem_post(&container[index].notFull);
        return false;
    }

    frame->plane0Size = container[index].videoFrame[container[index].idx].width *
                        container[index].videoFrame[container[index].idx].height;
    frame->plane1Size = (container[index].videoFrame[container[index].idx].width *
                         container[index].videoFrame[container[index].idx].height) / 4;
    frame->plane2Size = (container[index].videoFrame[container[index].idx].width *
                         container[index].videoFrame[container[index].idx].height) / 4;

    frame->plane0 = malloc(frame->plane0Size + 10);
    frame->plane1 = malloc(frame->plane1Size + 10);
    frame->plane2 = malloc(frame->plane2Size + 10);

    memcpy(frame->plane0, container[index].videoFrame[container[index].idx].data[0], frame->plane0Size);
    memcpy(frame->plane1, container[index].videoFrame[container[index].idx].data[1], frame->plane1Size);
    memcpy(frame->plane2, container[index].videoFrame[container[index].idx].data[2], frame->plane2Size);


#ifdef WIN32
    /*    if (!ReleaseSemaphore(
        notFull,  // handle to semaphore
        1,            // increase count by one
        NULL))       // not interested in previous count
    {
        printf("ReleaseSemaphore error: %d\n", GetLastError());
    }
    */
    SetEvent(notFull);

#else
    sem_post(&container[index].notFull);
#endif
    return true;

}


void CRLVirtualCamera::getCameraStream(crl::multisense::image::Header *stream) {
    /*
    pauseThread = false;
    assert(stream != nullptr);

    //sem_wait(&notEmpty);
    auto *str = stream;

    auto *yuv420pBuffer = (uint8_t *) malloc(bufferSize);
    str->imageDataP = malloc(bufferSize);

    if (videoFrame->coded_picture_number > 593) // TODO REMOVE MAGIC NUMBER
        return;

    uint32_t size = videoFrame[0].width * videoFrame[0].height;
    memcpy(yuv420pBuffer, videoFrame[0].data[0], size);
    memcpy(yuv420pBuffer + size, videoFrame[0].data[1], (videoFrame[0].width * videoFrame[0].height) / 2);

    uint32_t totalSize = size + (videoFrame[0].width * videoFrame[0].height) / 2;

    memcpy((void *) str->imageDataP, yuv420pBuffer, totalSize);

    free(yuv420pBuffer);

    str->imageLength = (uint32_t) bufferSize;
    str->frameId = frameIndex;
    str->width = videoFrame[0].width;
    str->height = videoFrame[0].height;

    frameIndex++;
    if (frameIndex == 5)
        frameIndex = 0;

    //sem_post(&notFull);

     */
}


// TODO COMPLETE IMPLEMENTATION
bool CRLVirtualCamera::getCameraStream(ArEngine::YUVTexture *tex) {


}

int CRLVirtualCamera::childProcessDecode(uint32_t index) {
// thread declaration
    int N = 20;
#ifdef WIN32
    /*
    notEmpty = CreateSemaphore(
        NULL,           // default security attributes
        0,  // initial count
        1,  // maximum count
        NULL);          // unnamed semaphore

    notFull = CreateSemaphore(
        NULL,           // default security attributes
        0,  // initial count
        N,  // maximum count
        NULL);          // unnamed semaphore

        producer = new std::thread(CRLVirtualCamera::decode, this);
        */

    notEmpty = CreateEvent(NULL, FALSE, FALSE, NULL);
    notFull = CreateEvent(NULL, FALSE, FALSE, NULL);
    SetEvent(notFull);

    producer = new std::thread(CRLVirtualCamera::decode, this);

#else
    // Declaration of attribute......
    pthread_attr_t attr;

    // semaphore initialization
    int err = -1;
    err = sem_init(&container[index].notEmpty, 0, 0);
    if (err != 0) {
        Log::Logger::getInstance()->error("Failed to initialize producer notEmpty semaphore");
    }

    err = sem_init(&container[index].notFull, 0, N);
    if (err != 0) {
        Log::Logger::getInstance()->error("Failed to initialize producer notFull semaphore");
    }
    // pthread_attr_t initialization
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,
                                PTHREAD_CREATE_JOINABLE);

    args[index].ctx = this;
    args[index].index = index;
    // Creation of process
    int r1 = pthread_create(&container[index].producer, &attr, CRLVirtualCamera::DecodeContainer::decode, &args[index]);
    if (r1) {
        std::cout <<
                  "Error in creating thread" << std::endl;
        exit(-1);
    }
#endif
    return EXIT_SUCCESS;
}

void CRLVirtualCamera::getVideoMetadata(uint32_t i) {
    AVFormatContext *ctx_format = nullptr;
    AVCodecContext *ctx_codec = nullptr;
    const AVCodec *codec = nullptr;
    AVFrame *frame = av_frame_alloc();
    int stream_idx;
    SwsContext *ctx_sws = nullptr;
    std::string fileName = Utils::getTexturePath() + "Video/" + container[i].videoName;
    AVStream *vid_stream = nullptr;
    AVPacket *pkt = av_packet_alloc();

    //av_register_all();

    if (int ret = avformat_open_input(&ctx_format, fileName.c_str(), nullptr, nullptr) != 0) {
        std::cout << 1 << std::endl;
        Log::Logger::getInstance()->error("Error in opening video input: {}", fileName.c_str());

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

    container[i].width = ctx_codec->width;
    container[i].height = ctx_codec->height;

    avformat_close_input(&ctx_format);
    av_packet_unref(pkt);
    avcodec_free_context(&ctx_codec);
    avformat_free_context(ctx_format);
}

void *CRLVirtualCamera::DecodeContainer::decode(void *arg) {
    auto *args = (DecodeThreadArgs *) arg;
    CRLVirtualCamera *instance = args->ctx;
    uint32_t idx = args->index;
    while (instance->container[idx].runDecodeThread) {

        // If paused or we haven't specificed any video
        if (instance->container[idx].pauseThread || instance->container[idx].videoName == "None") {
            std::chrono::milliseconds ten_ms(10);
            std::this_thread::sleep_for(ten_ms);
            continue;
        }

        AVFormatContext *ctx_format = nullptr;
        AVCodecContext *ctx_codec = nullptr;
        const AVCodec *codec = nullptr;
        AVFrame *frame = av_frame_alloc();
        int stream_idx;
        SwsContext *ctx_sws = nullptr;
        std::string fileName = Utils::getTexturePath() + "Video/" + instance->container[idx].videoName;
        AVStream *vid_stream = nullptr;
        AVPacket *pkt = av_packet_alloc();

        //av_register_all();

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


        //av_new_packet(pkt, pic_size);

        while (av_read_frame(ctx_format, pkt) >= 0 && instance->container[idx].runDecodeThread) {
            if (pkt->stream_index == stream_idx) {
                int ret = avcodec_send_packet(ctx_codec, pkt);
                if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    std::cout << "avcodec_send_packet: " << ret << std::endl;
                    break;
                }
                while (ret >= 0 && instance->container[idx].runDecodeThread) {
                    ret = avcodec_receive_frame(ctx_codec, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        //std::cout << "avcodec_receive_frame: " << ret << std::endl;
                        break;
                    }

#ifdef WIN32 // TODO USE MACROS INSTEAD AND DEFINE MACROS DEPENDING ON PLATFORM
                    DWORD dwWaitResult;
                    WaitForSingleObject(instance->notFull,INFINITE);
#else
                    sem_wait(&instance->container[idx].notFull);
#endif
                    uint32_t sequenceNumber = frame->coded_picture_number;

                    uint32_t indexSlot = sequenceNumber % 20;

                    instance->container[idx].videoFrame[indexSlot] = *frame;
                    instance->container[idx].bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, ctx_codec->width,
                                                                                   ctx_codec->height, 1);
                    instance->container[idx].frameIDs[instance->container[idx].idxFrame % 20] = indexSlot;


                    instance->container[idx].idxFrame++;
#ifdef WIN32
                    /*
                    if (!ReleaseSemaphore(
                        instance->notEmpty,  // handle to semaphore
                        1,            // increase count by one
                        NULL))       // not interested in previous count
                    {
                        printf("ReleaseSemaphore error: %d\n", GetLastError());
                    }
                    */
                    SetEvent(instance->notEmpty);

#else
                    sem_post(&instance->container[idx].notEmpty);
#endif
                    //instance->saveFrameYUV420P(frame, frame->width, frame->height, ctx_codec->frame_number);


                }
            }
            av_packet_unref(pkt);
        }


        avformat_close_input(&ctx_format);
        av_packet_unref(pkt);
        avcodec_free_context(&ctx_codec);
        avformat_free_context(ctx_format);
    }

    //pthread_exit((void *) (intptr_t) EXIT_SUCCESS);
}

void CRLVirtualCamera::saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame) {
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

CRLBaseInterface::CameraInfo CRLVirtualCamera::getCameraInfo() {
    return info;
}

