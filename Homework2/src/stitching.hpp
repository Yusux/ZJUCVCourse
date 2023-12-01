#ifndef __STITCHING_HPP__
#define __STITCHING_HPP__


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"

enum BlendType {
    NO_BLEND = 0,
    LINEAR_BLEND = 1,
    ALPHA_BLEND = 2,
    MY_MULTIBAND_BLEND = 3,
    OPENCV_MULTIBAND_BLEND = 4
};

enum SeamFinderType {
    NO = 0,
    VORONOI = 1,
    DP_COLOR = 2,
    DP_COLOR_GRAD = 3,
    GC_COLOR = 4,
    GC_COLOR_GRAD = 5
};

class MyStitcher {
public:
    /*
     * Constructor.
     * @param filenames: the filenames of the images
     * @param blend_type: the type of the blender
     * @param seam_finder_type: the type of the seam finder
     * @param output_inner: whether to output the inner image
     */
    MyStitcher(std::vector<cv::String> filenames,
               int blend_type = LINEAR_BLEND,
               int seam_finder_type = DP_COLOR,
               bool output_inner = false);

    /*
     * Destructor.
     */
    ~MyStitcher();

    /*
     * Stitch the images.
     * @param dst: the stitched image
     */
    void stitch(cv::Mat &dst);

private:
    bool output_inner_;
    int blend_type_;
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    cv::Ptr<cv::detail::SeamFinder> seam_finder_;
    cv::Mat dst_;
    std::vector<cv::String> filenames_;
    std::vector<cv::Mat> images_;
    std::vector<cv::Mat> images_gray_;
    std::vector<std::vector<cv::KeyPoint>> images_keypoints_;
    std::vector<cv::Mat> images_descriptors_;

    /*
     * Find the seam of the images.
     * @param images: the images
     * @param corners: the corners of the images
     * @param masks: the masks of the images
     */
    void seamFind(std::vector<cv::Mat> &images,
                  std::vector<cv::Point> &corners,
                  std::vector<cv::Mat> &masks);

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
     * @param idx: the index of the warped image,
     *            used to output the inner image
     */
    void imageBlending(cv::Mat &base, cv::Mat &warped, int idx);
};
#endif