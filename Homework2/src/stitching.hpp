#ifndef __STITCHING_HPP__
#define __STITCHING_HPP__


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/stitching/detail/blenders.hpp"

enum BlendType {
    NO_BLEND = 0,
    LINEAR_BLEND = 1,
    MY_MULTIBAND_BLEND = 2,
    OPENCV_MULTIBAND_BLEND = 3,
    ALPHA_BLEND = 4
};

class MyStitcher {
public:
    /*
     * Constructor.
     * @param filenames: the filenames of the images
     * @param output_inner: whether to output the inner image
     */
    MyStitcher(std::vector<cv::String> filenames,
               int blend_type = LINEAR_BLEND,
               bool output_inner = false);

    /*
     * Destructor.
     */
    ~MyStitcher();

    /*
     * Stitch the images.
     */
    void stitch();

    /*
     * Get the stitched image.
     
     * @return: the stitched image
     */
    cv::Mat getStitchedImg();

private:
    bool output_inner_;
    int blend_type_;
    cv::Mat dst_;
    cv::Ptr<cv::SIFT> sift_;
    std::vector<cv::String> filenames_;
    std::vector<cv::Mat> images_;
    std::vector<cv::Mat> images_gray_;
    std::vector<std::vector<cv::KeyPoint>> images_keypoints_;
    std::vector<cv::Mat> images_descriptors_;

    /*
     * Step 0: Prepare the images.
     * Read the images from the filenames, and
     * prepare the images for stitching.
     */
    void prepareImages();

    /*
     * Step 1: Detect the features.
     * Use SIFT to detect keypoints and extract descriptors
     */
    void detectFeatures();

    /*
     * Step 2: Feature Matching.
     * Choose the best match image.
     * Prepare the srcB from the dst.
     * @param srcA: the source image A
     * @param srcB: the source image B
     * @param keypointsA: the keypoints of image A
     * @param keypointsB: the keypoints of image B
     * @param best_matches: the best matches of image A and image B
     */
    void featureMatching(cv::Mat &srcA, cv::Mat &srcB, 
                         std::vector<cv::KeyPoint> &keypointsA,
                         std::vector<cv::KeyPoint> &keypointsB,
                         std::vector<cv::DMatch> &best_matches);

    /*
     * Step 3: Homography Estimation.
     * Estimate the homography matrix.
     * @param srcA: the source image A
     * @param srcB: the source image B
     * @param keypointsA: the keypoints of image A
     * @param keypointsB: the keypoints of image B
     * @param best_matches: the best matches of image A and image B
     * @param H: the homography matrix
     */
    void homographyEstimation(cv::Mat &srcA, cv::Mat &srcB,
                              std::vector<cv::KeyPoint> &keypointsA,
                              std::vector<cv::KeyPoint> &keypointsB,
                              std::vector<cv::DMatch> &best_matches,
                              cv::Mat &H);

    /*
     * Step 4: Image Warping
     * Warp the image A to match the image B
     * @param srcA: the source image A
     * @param srcB: the source image B
     * @param H: the homography matrix
     * @param base: the base image
     * @param warped: the warped image
     */
    void imageWarping(cv::Mat &srcA, cv::Mat &srcB, cv::Mat &H,
                      cv::Mat &base, cv::Mat &warped);

    /*
     * Step 5: Image Blending
     * Blend the warped image to the base image
     * @param base: the base image
     * @param warped: the warped image
     */
    void imageBlending(cv::Mat &base, cv::Mat &warped);
};
#endif