#define __STDC_CONSTANT_MACROS
extern "C" {
#include<libavutil/avutil.h>
#include<libavutil/imgutils.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include <iostream>
#include "opencv2/opencv.hpp"

void yuv2rgb(uint8_t yValue, uint8_t uValue, uint8_t vValue,
                       uint8_t *r, uint8_t *g, uint8_t *b) {
    int rTmp = yValue + (1.370705 * (vValue-128));
    // or fast integer computing with a small approximation
    // rTmp = yValue + (351*(vValue-128))>>8;
    int gTmp = yValue - (0.698001 * (vValue-128)) - (0.337633 * (uValue-128));
    // gTmp = yValue - (179*(vValue-128) + 86*(uValue-128))>>8;
    int bTmp = yValue + (1.732446 * (uValue-128));
    // bTmp = yValue + (443*(uValue-128))>>8;
    *r = rTmp;
    *g = gTmp;
    *b = bTmp;
}

void saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame)
{
    FILE *pFile;
    char szFilename[32];
    int  y;

    // Open file
    sprintf(szFilename, "frame%d.yuv", iFrame);
    pFile=fopen(szFilename, "wb");
    if(pFile==NULL)
        return;

    // Write pixel data
    fwrite(pFrame->data[0], 1, width*height, pFile);
    fwrite(pFrame->data[1], 1, width*height/4, pFile);
    fwrite(pFrame->data[2], 1, width*height/4, pFile);

    // Close file
    fclose(pFile);
}

static void rgb_save(void* data, unsigned char *buf1, unsigned char *buf2, unsigned char *buf3, int Ysize, int UVSize, int xsize, int ysize){

}


int main(int argc, char **argv) {
    AVFormatContext* ctx_format = nullptr;
    AVCodecContext* ctx_codec = nullptr;
    AVCodec* codec = nullptr;
    AVFrame* frame = av_frame_alloc();
    int stream_idx;
    SwsContext* ctx_sws = nullptr;
    const char* fin = "pixels.mpg";
    AVStream *vid_stream = nullptr;
    AVPacket* pkt = av_packet_alloc();

    av_register_all();

    if (int ret = avformat_open_input(&ctx_format, fin, nullptr, nullptr) != 0) {
        std::cout << 1 << std::endl;
        return ret;
    }
    if (avformat_find_stream_info(ctx_format, nullptr) < 0) {
        std::cout << 2 << std::endl;
        return -1; // Couldn't find stream information
    }
    av_dump_format(ctx_format, 0, fin, false);

    for (int i = 0; i < ctx_format->nb_streams; i++)
        if (ctx_format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            stream_idx = i;
            vid_stream = ctx_format->streams[i];
            break;
        }
    if (vid_stream == nullptr) {
        std::cout << 4 << std::endl;
        return -1;
    }

    codec = avcodec_find_decoder(vid_stream->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }
    ctx_codec = avcodec_alloc_context3(codec);

    if(avcodec_parameters_to_context(ctx_codec, vid_stream->codecpar)<0)
        std::cout << 512;
    if (avcodec_open2(ctx_codec, codec, nullptr)<0) {
        std::cout << 5;
        return -1;
    }

    //av_new_packet(pkt, pic_size);

    while(av_read_frame(ctx_format, pkt) >= 0){
        if (pkt->stream_index == stream_idx) {
            int ret = avcodec_send_packet(ctx_codec, pkt);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                std::cout << "avcodec_send_packet: " << ret << std::endl;
                break;
            }
            while (ret  >= 0) {
                ret = avcodec_receive_frame(ctx_codec, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    //std::cout << "avcodec_receive_frame: " << ret << std::endl;
                    break;
                }
                std::cout << "frame: " << ctx_codec->frame_number << std::endl;
                //rgb_save(frame->data, frame->data[0], frame->data[1], frame->data[2], frame->linesize[0], frame->linesize[1] frame->width, frame->height);
                saveFrameYUV420P(frame, frame->width, frame->height, ctx_codec->frame_number);
            }
        }
        av_packet_unref(pkt);
    }


    avformat_close_input(&ctx_format);
    av_packet_unref(pkt);
    avcodec_free_context(&ctx_codec);
    avformat_free_context(ctx_format);
    return 0;
}