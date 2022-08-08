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

    pauseThread = false;
    videoName = string;
    getVideoMetadata();

    if (!runDecodeThread) {
        runDecodeThread = true;
        childProcessDecode();
    }

    updateCameraInfo();
}

void CRLVirtualCamera::stop(std::string dataSourceStr) {
    pauseThread = true;

}

void CRLVirtualCamera::updateCameraInfo() {
    // Just populating it with some hardcoded data
    // - DevInfo

    cameraInfo.devInfo.name = "CRL Virtual Camera";
    cameraInfo.devInfo.imagerName = "Virtual";
    cameraInfo.devInfo.serialNumber = "25.8069758011"; // Root of all evil
    // - getImageCalibration
    cameraInfo.netConfig.ipv4Address = "Knock knock";

    cameraInfo.imgConf.setWidth(width);
    cameraInfo.imgConf.setHeight(height);

}

void CRLVirtualCamera::update() {


}

void CRLVirtualCamera::preparePointCloud(uint32_t width, uint32_t height) {

}


bool CRLVirtualCamera::getCameraStream(ArEngine::MP4Frame *frame) {
    pauseThread = false;
    assert(frame != nullptr);

#ifdef WIN32
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
    sem_wait(&notEmpty);
#endif

/*
    frame->plane0Size = videoFrame[0].linesize[0] * videoFrame[0].height;
    frame->plane1Size = videoFrame[0].linesize[1] * videoFrame[0].height;
    frame->plane2Size = videoFrame[0].linesize[2] * videoFrame[0].height;
 */

    frame->plane0Size = videoFrame[0].width * videoFrame[0].height;
    frame->plane1Size = (videoFrame[0].width * videoFrame[0].height) / 4;
    frame->plane2Size = (videoFrame[0].width * videoFrame[0].height) / 4;

    frame->plane0 = malloc(frame->plane0Size + 10);
    frame->plane1 = malloc(frame->plane1Size + 10);
    frame->plane2 = malloc(frame->plane2Size + 10);

    memcpy(frame->plane0, videoFrame[0].data[0], frame->plane0Size);
    memcpy(frame->plane1, videoFrame[0].data[1], frame->plane1Size);
    memcpy(frame->plane2, videoFrame[0].data[2], frame->plane2Size);

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
    sem_post(&notFull);
#endif
    return true;

}


void CRLVirtualCamera::getCameraStream(crl::multisense::image::Header *stream) {
    pauseThread = false;
    assert(stream != nullptr);

    //sem_wait(&notEmpty);
    std::cout << "Consumer consumes item. Items Present = " << --items << std::endl;
    auto *str = stream;

    auto *yuv420pBuffer = (uint8_t *) malloc(bufferSize);
    str->imageDataP = malloc(bufferSize);

    if (videoFrame->coded_picture_number > 593)
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

}


int CRLVirtualCamera::childProcessDecode() {
// thread declaration
    int N = 1;
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
    sem_init(&notEmpty, 0, 0);
    sem_init(&notFull, 0, N);

    // pthread_attr_t initialization
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,
                                PTHREAD_CREATE_JOINABLE);

    // Creation of process
    r1 = pthread_create(&producer, &attr, CRLVirtualCamera::decode, this);
    if (r1) {
        std::cout <<
                  "Error in creating thread" << std::endl;
        exit(-1);
    }
#endif
    return EXIT_SUCCESS;
}

void CRLVirtualCamera::getVideoMetadata() {
    AVFormatContext *ctx_format = nullptr;
    AVCodecContext *ctx_codec = nullptr;
    const AVCodec *codec = nullptr;
    AVFrame *frame = av_frame_alloc();
    int stream_idx;
    SwsContext *ctx_sws = nullptr;
    std::string fileName = Utils::getTexturePath() + "Video/" + videoName;
    AVStream *vid_stream = nullptr;
    AVPacket *pkt = av_packet_alloc();

    //av_register_all();

    if (int ret = avformat_open_input(&ctx_format, fileName.c_str(), nullptr, nullptr) != 0) {
        std::cout << 1 << std::endl;
        Log::Logger::getInstance()->error("Error in opening video input: %s", fileName.c_str());

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

    width = ctx_codec->width;
    height = ctx_codec->height;

    avformat_close_input(&ctx_format);
    av_packet_unref(pkt);
    avcodec_free_context(&ctx_codec);
    avformat_free_context(ctx_format);
}

void *CRLVirtualCamera::decode(void *arg) {
    auto *instance = (CRLVirtualCamera *) arg;
    while (instance->runDecodeThread) {

        // If paused or we haven't specificed any video
        if (instance->pauseThread, instance->videoName == "None") {
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
        std::string fileName = Utils::getTexturePath() + "Video/" + instance->videoName;
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

                    // TODO REMOVE THIS STOPS AFTER 593 frames
                    if (frame->coded_picture_number > 593)
                        continue;
#ifdef WIN32
                    DWORD dwWaitResult;
                    WaitForSingleObject(instance->notFull,INFINITE);
#else
                    sem_wait(&instance->notFull);
#endif
                    //std::cout << "frame: " << ctx_codec->frame_number << std::endl;
                    instance->videoFrame[0] = *frame;
                    instance->decoded = true;

                    instance->bufferSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, ctx_codec->width,
                                                                    ctx_codec->height, 1);
                    index++;
                    if (index == 5)
                        index = 0;
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
                    sem_post(&instance->notEmpty);
#endif
                    //instance->saveFrameYUV420P(frame, frame->width, frame->height, ctx_codec->frame_number);

                }
            }
            av_packet_unref(pkt);
        }

        instance->decoded = false;

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