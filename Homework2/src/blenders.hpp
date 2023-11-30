#ifndef __BLENDERS_HPP__
#define __BLENDERS_HPP__

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/blenders.hpp"

// calculate the linear blend of the base and warped image
// based on the distance of the boundary
/*
 * Calculate the linear blend of the base and warped image
 * based on the distance of the boundary, using
 * dst = alpha * warped + (1 - alpha) * base
 * @param base: the base image
 * @param warped: the warped image
 * @param dst: the dst image
 */
void linearBlend(cv::Mat &base, cv::Mat &warped, cv::Mat &dst);

/*
 * Sub-function of alpha blending
 * to calculate the border of the mask
 * reference: https://stackoverflow.com/a/22324790
 * @param mask: the mask
 * @return: the border of the mask
 */
cv::Mat border(cv::Mat mask);

/*
 * Function to compute the alpha blending of two images
 * based on the distance of the boundary
 * reference: https://stackoverflow.com/a/22324790
 * @param image1: the first image
 * @param mask1: the mask of the first image
 * @param image2: the second image
 * @param mask2: the mask of the second image
 * @return: the blended image
 */
cv::Mat computeAlphaBlending(cv::Mat image1, cv::Mat mask1, cv::Mat image2, cv::Mat mask2);


/*
 * Self implemented multi-band blender class
 */
class MyMultiBandBlender {
public:
    /*
     * Constructor of the multi-band blender
     * @param images: the images to blend
     * @param num_bands: the number of bands, -1 means automatically calculate the number of bands
     */
    MyMultiBandBlender(std::vector<cv::Mat> images, int num_bands = -1);

    /*
     * Destructor of the multi-band blender
     */
    ~MyMultiBandBlender();

    /*
     * Blend the two images
     * @param dst: the dst image
     */
    void blend(cv::Mat &dst);

private:
    int num_bands_;
    cv::Mat mask_;
    cv::Mat dst_;
    std::vector<cv::Mat> images_;
    std::vector<cv::Mat> masks_;
    std::vector<cv::Mat> pyr_gaussian_;
    std::vector<std::vector<cv::Mat>> pyr_laplace_;

    /*
     * Use the args get from the constructor
     * to prepare the image pyramids, including
     * the gaussian pyramid and the laplace pyramid
     */
    void prepare();

    /*
     * Calculate the gaussian pyramid of the image
     * @param img: the image
     * @param pyr_gaussian: the gaussian pyramid of the image
     */
    void calGaussianPyramid(cv::Mat img, std::vector<cv::Mat> &pyr_gaussian);

    /*
     * Calculate the laplace pyramid of the image
     * @param img: the image
     * @param pyr_laplace: the laplace pyramid of the image
     */
    void calLaplacePyramid(cv::Mat img, std::vector<cv::Mat> &pyr_laplace);

    /*
     * Blend the laplace pyramids of the two images
     * @param pyr_laplace_base: the laplace pyramid of the base image
     * @param pyr_laplace_add: the laplace pyramid of the warped image
     * @param pyr_gaussian: the gaussian pyramid of the mask
     * @param pyr_laplace_dst: the laplace pyramid of the dst image
     */
    void blendPyramids(std::vector<cv::Mat> &pyr_laplace_base,
                       std::vector<cv::Mat> &pyr_laplace_add,
                       std::vector<cv::Mat> &pyr_gaussian,
                       std::vector<cv::Mat> &pyr_laplace_dst);

    /*
     * Reconstruct the image from the laplace pyramid
     * @param pyr_laplace: the laplace pyramid
     * @param dst: the dst image
     */
    void reconstruct(std::vector<cv::Mat> &pyr_laplace, cv::Mat &dst);
};

#endif