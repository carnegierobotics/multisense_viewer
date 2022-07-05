//
// Created by magnus on 6/27/22.
//

/*
 * Copyright (c) 2001 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file
 * video decoding with libavcodec API example
 *
 * @example decode_video.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include <libavcodec/avcodec.h>

}

#include "opencv2/opencv.hpp"

#define INBUF_SIZE 4096

static void rgb_save(unsigned char *buf1, unsigned char *buf2, unsigned char *buf3, int wrap, int xsize, int ysize,
                     char *filename) {
    cv::Mat img(2160, 3840, CV_8UC3);
    cv::Mat chans[3];

//split the channels in order to manipulate them
    cv::split(img, chans);

//by default opencv put channels in BGR order , so in your situation you want to copy the first channel which is blue. Set green and red channels elements to zero.
    chans[2].data = buf3;
    chans[1].data = buf2;
    chans[0].data = buf1;

//then merge them back
    cv::merge(chans, 3, img);

//display
    cv::Mat out;
    cv::resize(img, out, cv::Size(0, 0), 0.25, 0.25);
    cv::imshow("img", out);
    cv::waitKey(0);

}

static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename) {
    FILE *f;
    int i;

    f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}

static void decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt,
                   const char *filename) {
    char buf[1024];
    int ret;

    if (pkt->stream_index == 0) {
        int ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            std::cout << "avcodec_send_packet: " << ret << std::endl;
        }
        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                //std::cout << "avcodec_receive_frame: " << ret << std::endl;
                break;
            }

            printf("saving frame %3d\n", dec_ctx->frame_number);
            fflush(stdout);

            /* the picture is allocated by the decoder. no need to
               free it */
            snprintf(buf, sizeof(buf), "%s-%d", filename, dec_ctx->frame_number);
            //pgm_save(frame->data[0], frame->linesize[0],frame->width, frame->height, buf);
            rgb_save(frame->data[0], frame->data[1], frame->data[2], frame->linesize[0], frame->width, frame->height,
                     buf);

        }
    }
}

int main(int argc, char **argv) {
    const char *filename, *outfilename;
    const AVCodec *codec;
    AVCodecParserContext *parser;
    AVCodecContext *c = NULL;
    FILE *f;
    AVFrame *frame;
    uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    uint8_t *data;
    size_t data_size;
    int ret;
    int eof;
    AVPacket *pkt;

    filename = "pixels.mpg";
    outfilename = "./out/outfile";

    pkt = av_packet_alloc();
    if (!pkt)
        exit(1);

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams) */
    memset(inbuf + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    /* find the MPEG-1 video decoder AV_CODEC_ID_MPEG4 AV_CODEC_ID_H264 */
    codec = avcodec_find_decoder(AV_CODEC_ID_MPEG2VIDEO);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */
    //c->width = 3840; c->height = 2160;


    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    do {
        /* read raw data from the input file */
        data_size = fread(inbuf, 1, INBUF_SIZE, f);
        if (ferror(f))
            break;
        eof = !data_size;

        /* use the parser to split the data into frames */
        data = inbuf;
        while (data_size > 0 || eof) {
            ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                                   data, (int) data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data += ret;
            data_size -= ret;

            if (pkt->size)
                decode(c, frame, pkt, outfilename);
            else if (eof)
                break;
        }
    } while (!eof);

    /* flush the decoder */
    decode(c, frame, NULL, outfilename);

    fclose(f);

    av_parser_close(parser);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    return 0;
}