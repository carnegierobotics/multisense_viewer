#include "Feature.h"

#if USE_CUDA
static void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}
#endif

void deleteUnmatchFeatures(std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1,
                           std::vector<uchar> &status) {
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        cv::Point2f pt = points1.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points0.erase(points0.begin() + (i - indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f> &points) {
//uses FAST as for feature dection, modify parameters as necessary
    std::vector<cv::KeyPoint> keypoints;
    int fast_threshold = 10;
    bool nonmaxSuppression = true;
    cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}


void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points3,
                                 std::vector<cv::Point2f> &points0_return, std::vector<uchar> &status0,
                                 std::vector<uchar> &status1, std::vector<uchar> &status2, std::vector<uchar> &status3,
                                 std::vector<int> &ages) {
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    for (int i = 0; i < ages.size(); ++i) {
        ages[i] += 1;
    }

    int indexCorrection = 0;
    for (int i = 0; i < status3.size(); i++) {
        cv::Point2f pt0 = points0.at(i - indexCorrection);
        cv::Point2f pt3 = points3.at(i - indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i - indexCorrection);

        if ((status3.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) ||
            (status0.at(i) == 0) || (pt0.x < 0) || (pt0.y < 0)) {
            if ((pt0.x < 0) || (pt0.y < 0) || (pt3.x < 0) || (pt3.y < 0)) {
                status3.at(i) = 0;
            }
            points0.erase(points0.begin() + (i - indexCorrection));
            points3.erase(points3.begin() + (i - indexCorrection));
            points0_return.erase(points0_return.begin() + (i - indexCorrection));
            ages.erase(ages.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void opticalFlow(const cv::Mat &img_l_0, const cv::Mat &img_l_1, std::vector<cv::Point2f> *points_l_0,
                 std::vector<cv::Point2f> *points_l_1, std::vector<uchar> *status) {
    std::vector<float> err;
    cv::Size winSize = cv::Size(15, 15);
    cv::TermCriteria termcrit = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01);
    calcOpticalFlowPyrLK(img_l_0, img_l_1, *points_l_0, *points_l_1, *status, err, winSize, 4, termcrit, 0, 0.001);

}

void appendNewFeatures(const cv::Mat &image, FeatureSet *current_features) {
    std::vector<cv::Point2f> points_new;
    featureDetectionFast(image, points_new);
    current_features->points.insert(current_features->points.end(), points_new.begin(), points_new.end());
    std::vector<int> ages_new(points_new.size(), 0);
    current_features->ages.insert(current_features->ages.end(), ages_new.begin(), ages_new.end());
}


void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points1,
                                 std::vector<cv::Point2f> &points2, std::vector<cv::Point2f> &points3,
                                 std::vector<cv::Point2f> &points0_return,
                                 std::vector<uchar> &status0, std::vector<uchar> &status1,
                                 std::vector<uchar> &status2, std::vector<uchar> &status3,
                                 std::vector<int> &ages) {
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    for (int i = 0; i < ages.size(); ++i) {
        ages[i] += 1;
    }

    int indexCorrection = 0;
    for (int i = 0; i < status3.size(); i++) {
        cv::Point2f pt0 = points0.at(i - indexCorrection);
        cv::Point2f pt1 = points1.at(i - indexCorrection);
        cv::Point2f pt2 = points2.at(i - indexCorrection);
        cv::Point2f pt3 = points3.at(i - indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i - indexCorrection);

        if ((status3.at(i) == 0) || (pt3.x < 0) || (pt3.y < 0) ||
            (status2.at(i) == 0) || (pt2.x < 0) || (pt2.y < 0) ||
            (status1.at(i) == 0) || (pt1.x < 0) || (pt1.y < 0) ||
            (status0.at(i) == 0) || (pt0.x < 0) || (pt0.y < 0)) {
            if ((pt0.x < 0) || (pt0.y < 0) || (pt1.x < 0) || (pt1.y < 0) || (pt2.x < 0) || (pt2.y < 0) || (pt3.x < 0) ||
                (pt3.y < 0)) {
                status3.at(i) = 0;
            }
            points0.erase(points0.begin() + (i - indexCorrection));
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            points3.erase(points3.begin() + (i - indexCorrection));
            points0_return.erase(points0_return.begin() + (i - indexCorrection));

            ages.erase(ages.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }
}

void
checkValidMatch(std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &points_return, std::vector<bool> &status,
                int threshold) {
    int offset;
    for (int i = 0; i < points.size(); i++) {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        // std::cout << offset << ", ";

        if (offset > threshold) {
            status.push_back(false);
        } else {
            status.push_back(true);
        }
    }
}

void removeInvalidPoints(std::vector<cv::Point2f> &points, const std::vector<bool> &status) {
    int index = 0;
    for (int i = 0; i < status.size(); i++) {
        if (status[i] == false) {
            points.erase(points.begin() + index);
        } else {
            index++;
        }
    }
}


void trackingFrame2Frame(cv::Mat &projMatrl, cv::Mat &projMatrr,
                         std::vector<cv::Point2f> &pointsLeft_t0,
                         std::vector<cv::Point2f> &pointsLeft_t1,
                         cv::Mat &points3D_t0,
                         cv::Mat &rotation,
                         cv::Mat &translation,
                         bool mono_rotation) {

    // Calculate frame to frame transformation

    // -----------------------------------------------------------
    // Rotation(R) estimation using Nister's Five Points Algorithm
    // -----------------------------------------------------------
    double focal = projMatrl.at<float>(0, 0);
    cv::Point2d principle_point(projMatrl.at<float>(0, 2), projMatrl.at<float>(1, 2));

    //recovering the pose and the essential cv::matrix
    cv::Mat E, mask;
    cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
    if (mono_rotation) {
        E = cv::findEssentialMat(pointsLeft_t0, pointsLeft_t1, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, pointsLeft_t0, pointsLeft_t1, rotation, translation_mono, focal, principle_point, mask);
        // std::cout << "recoverPose rotation: " << rotation << std::endl;
    }
    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) <<
            projMatrl.at<float>(0, 0), projMatrl.at<float>(0,1), projMatrl.at<float>(0, 2),
            projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
            projMatrl.at<float>(2, 0), projMatrl.at<float>(2, 1), projMatrl.at<float>(2, 2));
    // projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

    int iterationsCount = 1000;        // number of Ransac iterations.
    float reprojectionError = .5;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.999;          // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags = cv::SOLVEPNP_ITERATIVE;


    cv::Mat inliers;

    cv::solvePnPRansac(points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, rvec, translation,
                       useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                       inliers, flags);

    std::cout << "translation: " << translation.t() << std::endl;

    if (!mono_rotation) {
        cv::Rodrigues(rvec, rotation);
    }

}



void displayTracking(cv::Mat &imageLeft_t1,
                     std::vector<cv::Point2f> &pointsLeft_t0,
                     std::vector<cv::Point2f> &pointsLeft_t1) {
    // -----------------------------------------
    // Display feature racking
    // -----------------------------------------
    int radius = 2;
    cv::Mat vis;

    cv::cvtColor(imageLeft_t1, vis, cv::COLOR_GRAY2BGR, 3);


    for (int i = 0; i < pointsLeft_t0.size(); i++) {
        cv::circle(vis, cv::Point(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(0, 255, 0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++) {
        cv::circle(vis, cv::Point(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius, CV_RGB(255, 0, 0));
    }

    for (int i = 0; i < pointsLeft_t1.size(); i++) {
        cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(0, 255, 0));
    }

    cv::imshow("vis ", vis);
}

bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());

    return  norm(I, shouldBeIdentity) < 1e-6;

}

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{

    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);

}


void integrateOdometryStereo(int frame_i, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, const cv::Mat& translation_stereo)
{

    // std::cout << "rotation" << rotation << std::endl;
    // std::cout << "translation_stereo" << translation_stereo << std::endl;


    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

    double scale = sqrt((translation_stereo.at<double>(0))*(translation_stereo.at<double>(0))
                        + (translation_stereo.at<double>(1))*(translation_stereo.at<double>(1))
                        + (translation_stereo.at<double>(2))*(translation_stereo.at<double>(2)));

    rigid_body_transformation = rigid_body_transformation.inv();
    //frame_pose = frame_pose * rigid_body_transformation;

    // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) && (translation_stereo.at<double>(2) > translation_stereo.at<double>(1)))
    if (scale > 0.01)
    {
        // std::cout << "Rpose" << Rpose << std::endl;

        frame_pose = frame_pose * rigid_body_transformation;

    }
    else
    {
        //std::cout << "[WARNING] scale below 0.01, or incorrect translation" << std::endl;
    }
}


void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose)
{
    // draw estimated trajectory
    int x = int(pose.at<double>(0)) * 10 + 600;
    int y = int(pose.at<double>(2)) * 10 + 300;

    circle(trajectory, cv::Point(x, y) , 4, CV_RGB(255,0,0), 2);

    // rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    // sprintf(text, "FPS: %02f", fps);
    // putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    cv::imshow( "Trajectory", trajectory );
}


void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;
    int buckets_number = buckets_nums_height * buckets_nums_width;

    std::vector<Bucket> Buckets;

    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
        {
            Buckets.push_back(Bucket(features_per_bucket));
        }
    }

    // bucket all current features into buckets by their location
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
        buckets_nums_height_idx = current_features.points[i].y/bucket_size;
        buckets_nums_width_idx = current_features.points[i].x/bucket_size;
        buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
        Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i]);

    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
        for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
        {
            buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
            Buckets[buckets_idx].get_features(current_features);
        }
    }

    //std::cout << "current features number after bucketing: " << current_features.size() << std::endl;

}