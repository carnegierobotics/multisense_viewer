//
// Created by magnus on 6/13/23.
//

#ifndef MULTISENSE_VIEWER_YOLOV7_H
#define MULTISENSE_VIEWER_YOLOV7_H

#include <fstream>
#include <opencv2/dnn/dnn.hpp>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct YoloV7Config
{
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    std::string modelPath;
    int inpHeight;
    int inpWidth;
};

class YOLOV7 {
public:
    int inpWidth;
    int inpHeight;
    std::vector<std::string> class_names;
    int num_class;

    float confThreshold;
    float nmsThreshold;
    cv::dnn::Net net;


    YOLOV7(YoloV7Config config) {
        confThreshold = config.confThreshold;
        nmsThreshold = config.nmsThreshold;

        net = cv::dnn::readNet(config.modelPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

        std::ifstream ifs(Utils::getAssetsPath() / "ML/yolov7.yaml");
        std::string line;
        while (getline(ifs, line)) class_names.push_back(line);
        num_class = class_names.size();
        inpHeight = config.inpHeight;
        inpWidth = config.inpWidth;
    }

    void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat &frame,
                          int classid)   // Draw the predicted bounding box
    {
        //Draw a rectangle displaying the bounding box
        rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);
        //Get the label for the class name and its confidence
        std::string label = cv::format("%.2f", conf);
        label = this->class_names[classid] + ":" + label;

        //Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = cv::max(top, labelSize.height);
        //rectangle(frame, cv::Point(left, top - int(1.5 * labelSize.height)), cv::Point(left + int(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), FILLED);
        putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
    }

    void detect(cv::Mat &frame) {
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight),
                                              cv::Scalar(0, 0, 0), true, false);
        this->net.setInput(blob);
        std::vector<cv::Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
        int num_proposal = outs[0].size[0];
        int nout = outs[0].size[1];
        if (outs[0].dims > 2) {
            num_proposal = outs[0].size[1];
            nout = outs[0].size[2];
            outs[0] = outs[0].reshape(0, num_proposal);
        }
        /////generate proposals
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        float ratioh = static_cast<float>( frame.rows / this->inpHeight), ratiow = static_cast<float>(frame.cols / this->inpWidth);
        int n = 0, row_ind = 0; ///cx,cy,w,h,box_score,class_score
        float *pdata = reinterpret_cast<float *>(outs[0].data);
        for (n = 0; n < num_proposal; n++)   ///ÌØÕ÷Í¼³ß¶È
        {
            float box_score = pdata[4];
            if (box_score > this->confThreshold) {
                cv::Mat scores = outs[0].row(row_ind).colRange(5, nout);
                cv::Point classIdPoint;
                double max_class_socre;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                max_class_socre *= box_score;
                if (max_class_socre > this->confThreshold) {
                    const int class_idx = classIdPoint.x;
                    float cx = pdata[0] * ratiow;  ///cx
                    float cy = pdata[1] * ratioh;   ///cy
                    float w = pdata[2] * ratiow;   ///w
                    float h = pdata[3] * ratioh;  ///h

                    int left = int(cx - 0.5f * w);
                    int top = int(cy - 0.5f * h);

                    confidences.push_back(static_cast<float> (max_class_socre));
                    boxes.push_back(cv::Rect(left, top, static_cast<int>(w), static_cast<int>(h)));
                    classIds.push_back(class_idx);
                }
            }
            row_ind++;
            pdata += nout;
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            this->drawPred(confidences[idx], box.x, box.y,
                           box.x + box.width, box.y + box.height, frame, classIds[idx]);
        }
        /*
        */
    }
};

#endif //MULTISENSE_VIEWER_YOLOV7_H
