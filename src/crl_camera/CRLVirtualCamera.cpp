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

#include<bits/stdc++.h>
#include<pthread.h>
#include<semaphore.h>
#include <MultiSense/src/tools/Utils.h>
#include <thread>

bool CRLVirtualCamera::connect(const std::string &ip) {

    online = true;
    return true;
}

void CRLVirtualCamera::start(std::string string, std::string dataSourceStr) {

    runDecodeThread = true;
    childProcessDecode();

}

void CRLVirtualCamera::stop(std::string dataSourceStr) {
    void *status;
    pthread_join(producer, &status);
    printf("Decoder thread exited with status %ld\n", (intptr_t) status);
}

void CRLVirtualCamera::updateCameraInfo() {
    // Just populating it with some hardcoded data
    // - DevInfo

    cameraInfo.devInfo.name = "CRL Virtual Camera";
    cameraInfo.devInfo.imagerName = "Virtual";
    cameraInfo.devInfo.serialNumber = "25.8069758011"; // Root of all evil
    // - getImageCalibration
    cameraInfo.netConfig.ipv4Address = "Knock knock";

}

void CRLVirtualCamera::update() {



}

void CRLVirtualCamera::preparePointCloud(uint32_t width, uint32_t height) {

}

void CRLVirtualCamera::getCameraStream(std::string stringSrc, crl::multisense::image::Header **stream,
                                       crl::multisense::image::Header **stream2) {

    assert(stream != nullptr);

    sem_wait(&notEmpty);
    std::cout << "Consumer consumes item. Items Present = " << --items << std::endl;

    auto * str = *stream;
    str->imageLength = bufferSize;
    str->frameId = frameIndex;
    str->imageDataP = (void *) videoFrame[frameIndex].data;
    str->width = videoFrame[frameIndex].width;
    str->height = videoFrame[frameIndex].height;

    frameIndex++;
    if (frameIndex == 5)
        frameIndex = 0;

    sem_post(&notFull);

}


int CRLVirtualCamera::childProcessDecode() {
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
    r1 = pthread_create(&producer, &attr, CRLVirtualCamera::decode, this);
    if (r1) {
        std::cout <<
                  "Error in creating thread" << std::endl;
        exit(-1);
    }
    return EXIT_SUCCESS;
}

void *CRLVirtualCamera::decode(void *arg) {
    auto *instance = (CRLVirtualCamera *) arg;
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