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

void deleteUnmatchFeatures(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1, std::vector<uchar>& status)
{
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points1.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))   
        {
              if((pt.x<0)||(pt.y<0))    
              {
                status.at(i) = 0;
              }
              points0.erase (points0.begin() + (i - indexCorrection));
              points1.erase (points1.begin() + (i - indexCorrection));
              indexCorrection++;
        }
     }
}

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points)  
{   
//uses FAST as for feature dection, modify parameters as necessary
  std::vector<cv::KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}


void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f> &points0, std::vector<cv::Point2f> &points3,
                                 std::vector<cv::Point2f> &points0_return, std::vector<uchar> &status0,
                                 std::vector<uchar> &status1, std::vector<uchar> &status2, std::vector<uchar> &status3,
                                 std::vector<int> &ages) {
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  for (int i = 0; i < ages.size(); ++i)
  {
     ages[i] += 1;
  }

  int indexCorrection = 0;
  for( int i=0; i<status3.size(); i++)
     {  cv::Point2f pt0 = points0.at(i- indexCorrection);
        cv::Point2f pt3 = points3.at(i- indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i- indexCorrection);
        
        if ((status3.at(i) == 0)||(pt3.x<0)||(pt3.y<0)||
            (status0.at(i) == 0)||(pt0.x<0)||(pt0.y<0))   
        {
          if((pt0.x<0)||(pt0.y<0)||(pt3.x<0)||(pt3.y<0))
          {
            status3.at(i) = 0;
          }
          points0.erase (points0.begin() + (i - indexCorrection));
          points3.erase (points3.begin() + (i - indexCorrection));
          points0_return.erase (points0_return.begin() + (i - indexCorrection));
          ages.erase (ages.begin() + (i - indexCorrection));
          indexCorrection++;
        }
     }  
}

void opticalFlow(const cv::Mat &img_l_0, const cv::Mat &img_l_1, std::vector<cv::Point2f> *points_l_0,
                 std::vector<cv::Point2f> *points_l_1,   std::vector<uchar>* status) {
  
  //this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  cv::Size winSize=cv::Size(15,15);
  cv::TermCriteria termcrit=cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);


  clock_t tic = clock();
  calcOpticalFlowPyrLK(img_l_0, img_l_1, *points_l_0, *points_l_1 , *status, err, winSize, 2, termcrit);
  clock_t toc = clock();
  std::cerr << "calcOpticalFlowPyrLK time: " << float(toc - tic)/CLOCKS_PER_SEC*1000 << "ms" << std::endl;

}

void appendNewFeatures(const cv::Mat &image, FeatureSet *current_features)
{
    std::vector<cv::Point2f>  points_new;
    featureDetectionFast(image, points_new);
    current_features->points.insert(current_features->points.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    current_features->ages.insert(current_features->ages.end(), ages_new.begin(), ages_new.end());
}

