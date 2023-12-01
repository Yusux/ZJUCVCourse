#include "utils.hpp"
#include "blenders.hpp"
#include <iostream>
#include <omp.h>

using namespace cv;

// calculate the linear blend of the base and warped image
// based on the distance of the boundary
void linearBlend(Mat &base, Mat &warped, Mat &dst) {
    Mat base_mask, warped_mask, overlap_mask;
    // create mask for the base and warped image
    getMask(base, base_mask);
    getMask(warped, warped_mask, 1, 2);

    // prepare the dst image
    dst = Mat::zeros(base.rows, base.cols, CV_8UC3);
    base.copyTo(dst, base_mask);
    warped.copyTo(dst, warped_mask);

    // erode the base_mask for 2 pixels and dilate for 1 pixel
    dilate(base_mask, base_mask, Mat(), Point(-1, -1), 1);
    erode(base_mask, base_mask, Mat(), Point(-1, -1), 2);
    // get the overlap mask
    bitwise_and(base_mask, warped_mask, overlap_mask);

    // find the approx center of the base mask,
    // warped mask and overlap mask
    Rect base_rect, warped_rect, overlap_rect;
    getMinEnclosingRect(base_mask, base_rect);
    getMinEnclosingRect(warped_mask, warped_rect);
    getMinEnclosingRect(overlap_mask, overlap_rect);
    // get the center of the rects
    Point base_mask_center = (base_rect.tl() + base_rect.br()) / 2;
    Point warped_mask_center = (warped_rect.tl() + warped_rect.br()) / 2;

    // use the calculated alpha in [0, 1] to represent the radio
    // of the distance of the boundary to the bottom_left of the base mask
    // to paint the dst image
#pragma omp parallel for
    for (int y = overlap_rect.y; y < overlap_rect.y + overlap_rect.height; y++) {
        // find the start and end of the overlap mask
        int start = -1;
        int end = -1;
        for (int x = overlap_rect.x; x < overlap_rect.x + overlap_rect.width; x++) {
            if (overlap_mask.at<uchar>(y, x) == 255) {
                if (start == -1) {
                    start = x;
                }
                end = x;
            }
        }

        // if there is no overlap in this row
        if (start == -1) {
            continue;
        }

        // use the distance in x axis to the bottom_left of
        // the warped mask to determine the alpha
        // to determine the initial alpha and the distance_delta
        float distance_delta;
        float alpha;
        if (warped_mask_center.x < base_mask_center.x) {
            // warped_mask_center is in the left of base_mask_center
            // the alpha should be larger
            alpha = 1.0f;
            // and the distance_delta should be negative
            distance_delta = -1.0f / (end - start);
        } else {
            // the end is closer to the bottom_left
            // the alpha should be smaller
            alpha = 0.0f;
            // and the distance_delta should be positive
            distance_delta = 1.0f / (end - start);
        }

        // paint the dst image in this row
        for (int i = start; i <= end; i++) {
            dst.at<Vec3b>(y, i) = alpha * warped.at<Vec3b>(y, i) + (1 - alpha) * base.at<Vec3b>(y, i);
            // iter the alpha
            alpha += distance_delta;
        }
    }
}

Mat border(Mat mask) {
    Mat gx;
    Mat gy;

    Sobel(mask, gx, CV_32F, 1,0,3);
    Sobel(mask, gy, CV_32F, 0,1,3);

    Mat border;
    magnitude(gx, gy, border);

    return border > 100;
}

Mat computeAlphaBlending(Mat image1, Mat mask1, Mat image2, Mat mask2) {
    // find regions where no mask is set
    // compute the region where no mask is set at all
    // to use those color values unblended
    Mat overlap_mask = mask1 | mask2;
    Mat noMask = 255 - overlap_mask;
    // ------------------------------------------

    // create an image with equal alpha values:
    Mat rawAlpha = Mat(noMask.rows, noMask.cols, CV_32FC1);
    rawAlpha = 1.0f;

    // invert the border, so that border values are 0
    // which is needed for the distance transform
    Mat border1 = 255-border(mask1);
    Mat border2 = 255-border(mask2);

    // compute the distance to the object center
    Mat dist1;
    // L2 normal
    distanceTransform(border1, dist1, DIST_L2, 3);

    // scale distances to values between 0 and 1
    double min, max;
    Point minLoc, maxLoc;

    // find min/max vals
    minMaxLoc(dist1, &min, &max, &minLoc, &maxLoc, mask1&(dist1>0));    // find min values > 0
    dist1 = dist1 * 1.0 / max;  // values between 0 and 1 since min val should alwaysbe 0

    // same for the 2nd image
    Mat dist2;
    distanceTransform(border2, dist2, DIST_L2, 3);
    minMaxLoc(dist2, &min, &max, &minLoc, &maxLoc, mask2&(dist2>0));    // find min values > 0
    dist2 = dist2 * 1.0 / max;  // values between 0 and 1

    // mask the distance values to reduce information to masked regions
    Mat dist1Masked;
    rawAlpha.copyTo(dist1Masked, noMask);   // where no mask is set, blend with equal values
    dist1.copyTo(dist1Masked, mask1);
    rawAlpha.copyTo(dist1Masked, mask1&(255-mask2));

    Mat dist2Masked;
    rawAlpha.copyTo(dist2Masked, noMask);   // where no mask is set, blend with equal values
    dist2.copyTo(dist2Masked, mask2);
    rawAlpha.copyTo(dist2Masked, mask2&(255-mask1));

    // divide by the sum of both weights to get
    // a linear combination of both image's pixel values
    Mat blendMaskSum = dist1Masked + dist2Masked;

    // convert the images to float to multiply with the weight
    Mat im1Float;
    image1.convertTo(im1Float, dist1Masked.type());

    // the splitting is just used here to use .mul later... which needs same number of channels
    std::vector<Mat> channels1;
    split(im1Float, channels1);
    // multiply pixel value with the quality weights for image 1
    Mat im1AlphaB = dist1Masked.mul(channels1[0]);
    Mat im1AlphaG = dist1Masked.mul(channels1[1]);
    Mat im1AlphaR = dist1Masked.mul(channels1[2]);

    std::vector<Mat> alpha1;
    alpha1.push_back(im1AlphaB);
    alpha1.push_back(im1AlphaG);
    alpha1.push_back(im1AlphaR);
    Mat im1Alpha;
    merge(alpha1, im1Alpha);

    Mat im2Float;
    image2.convertTo(im2Float, dist2Masked.type());

    std::vector<Mat> channels2;
    split(im2Float, channels2);
    // multiply pixel value with the quality weights for image 2
    Mat im2AlphaB = dist2Masked.mul(channels2[0]);
    Mat im2AlphaG = dist2Masked.mul(channels2[1]);
    Mat im2AlphaR = dist2Masked.mul(channels2[2]);

    std::vector<Mat> alpha2;
    alpha2.push_back(im2AlphaB);
    alpha2.push_back(im2AlphaG);
    alpha2.push_back(im2AlphaR);
    Mat im2Alpha;
    merge(alpha2, im2Alpha);

    // now sum both weighted images and divide by the sum of the weights (linear combination)
    Mat imBlendedB = (im1AlphaB + im2AlphaB)/blendMaskSum;
    Mat imBlendedG = (im1AlphaG + im2AlphaG)/blendMaskSum;
    Mat imBlendedR = (im1AlphaR + im2AlphaR)/blendMaskSum;
    std::vector<Mat> channelsBlended;
    channelsBlended.push_back(imBlendedB);
    channelsBlended.push_back(imBlendedG);
    channelsBlended.push_back(imBlendedR);

    // merge back to 3 channel image
    Mat merged;
    merge(channelsBlended, merged);

    // convert to 8UC3
    Mat merged8U;
    merged.convertTo(merged8U, CV_8UC3);

    return merged8U;
}

MyMultiBandBlender::MyMultiBandBlender(std::vector<Mat> images,
                                       std::vector<Mat> masks,
                                       bool output_inner,
                                       int num_bands) {
    CV_Assert(images.size() == 2 && masks.size() == 2);

    // get the intersection of the two images
    Mat mask0, mask1, overlap_mask;
    getMask(images[0], mask0);
    getMask(images[1], mask1, 1, 2);
    // get the intersection of the two masks
    bitwise_and(mask0, mask1, overlap_mask);
    Rect rect;
    getMinEnclosingRect(overlap_mask, rect);

    // copy the masks
    masks_.resize(masks.size());
#pragma omp parallel for
    for (int i = 0; i < masks.size(); i++) {
        masks[i].convertTo(masks_[i], CV_32FC1, 1./255.);
    }

    // copy the images
    images_.resize(images.size());
#pragma omp parallel for
    for (int i = 0; i < images.size(); i++) {
        images[i].convertTo(images_[i], CV_32FC3);
    }

    // whether to output inner image
    output_inner_ = output_inner;

    // get the number of bands
    if (num_bands == -1) {
        // calculate the number of bands
        float min_size = std::fmin(std::min(rect.width, rect.height), std::sqrt(rect.area())/24.0f);
        num_bands_ = (int)std::ceil(std::log(min_size) / std::log(2)) - 1;
    } else {
        // use the given number of bands
        num_bands_ = num_bands;
    }

    // prepare the image pyramids
    prepare();
}

MyMultiBandBlender::~MyMultiBandBlender() {
}

void MyMultiBandBlender::prepare() {
    // calculate the mask pyramid
    pyr_gaussian_.resize(masks_.size());
#pragma omp parallel for
    for (int i = 0; i < masks_.size(); i++) {
        // calculate the gaussian pyramid for each mask
        calGaussianPyramid(masks_[i], pyr_gaussian_[i]);
    }

    // calculate the laplace pyramid for each image
    pyr_laplace_.resize(images_.size());
#pragma omp parallel for
    for (int i = 0; i < images_.size(); i++) {
        // calculate the laplace pyramid for each image
        calLaplacePyramid(images_[i], pyr_laplace_[i]);
    }
}

void MyMultiBandBlender::calGaussianPyramid(Mat img, std::vector<Mat> &pyr_gaussian) {
    // create the gaussian pyramid
    pyr_gaussian.resize(num_bands_ + 1);
    pyr_gaussian[0] = img;
    for (int i = 0; i < num_bands_; i++) {
        // Downsample the image
        pyrDown(pyr_gaussian[i], pyr_gaussian[i+1]);
    }

    // output all images in the gaussian pyramid
    if (output_inner_) {
        for (int i = 0; i < num_bands_ + 1; i++) {
            imwrite("inner/blenders/gaussian_" + std::to_string(i) + ".png", pyr_gaussian[i]*255);
        }
    }
}

void MyMultiBandBlender::calLaplacePyramid(Mat img, std::vector<Mat> &pyr_laplace) {
    pyr_laplace.resize(num_bands_+1);
    pyr_laplace[0] = img;
    for (int i = 0; i < num_bands_; i++) {
        pyrDown(pyr_laplace[i], pyr_laplace[i+1]);
    }
    Mat tmp;
    for (int i = 0; i < num_bands_; i++) {
        pyrUp(pyr_laplace[i+1], tmp, pyr_laplace[i].size());
        subtract(pyr_laplace[i], tmp, pyr_laplace[i]);
    }

    // output all images in the laplace pyramid
    if (output_inner_) {
        for (int i = 0; i < num_bands_ + 1; i++) {
            imwrite("inner/blenders/laplace_" + std::to_string(i) + ".png", pyr_laplace[i]);
        }
    }
}

void MyMultiBandBlender::blendPyramids(std::vector<Mat> &pyr_laplace_base,
                                       std::vector<Mat> &pyr_laplace_add,
                                       std::vector<Mat> &pyr_gaussian_base,
                                       std::vector<Mat> &pyr_gaussian_add,
                                       std::vector<Mat> &pyr_laplace_dst) {
    // create the laplace pyramid for the dst image
    pyr_laplace_dst.resize(num_bands_+1);
#pragma omp parallel for
    for (int i = 0; i <= num_bands_; i++) {
        pyr_laplace_dst[i] = Mat::zeros(pyr_laplace_base[i].size(), pyr_laplace_base[i].type());
    }

    for (int i = 0; i <= num_bands_; i++) {
        // blend the laplace pyramid
#pragma omp parallel for
        for (int y = 0; y < pyr_laplace_base[i].rows; y++) {
            Point3_<float> *row_dst = pyr_laplace_dst[i].ptr<Point3_<float>>(y);
            Point3_<float> *row_base = pyr_laplace_base[i].ptr<Point3_<float>>(y);
            Point3_<float> *row_add = pyr_laplace_add[i].ptr<Point3_<float>>(y);
            float *row_weight_base = pyr_gaussian_base[i].ptr<float>(y);
            float *row_weight_add = pyr_gaussian_add[i].ptr<float>(y);
            for (int x = 0; x < pyr_laplace_base[i].cols; x++) {
                float alpha = row_weight_add[x] / (row_weight_base[x] + row_weight_add[x]);
                row_dst[x].x += alpha * row_add[x].x + (1 - alpha) * row_base[x].x;
                row_dst[x].y += alpha * row_add[x].y + (1 - alpha) * row_base[x].y;
                row_dst[x].z += alpha * row_add[x].z + (1 - alpha) * row_base[x].z;
            }
        }
    }

    if (output_inner_) {
        for (int i = 0; i < num_bands_ + 1; i++) {
            imwrite("inner/blenders/laplace_dst_" + std::to_string(i) + ".png", pyr_laplace_dst[i]);
        }
    }
}

void MyMultiBandBlender::reconstruct(std::vector<Mat> &pyr_laplace, Mat &dst) {
    // reconstruct the image from the laplace pyramid
    Mat tmp;
    for (int i = num_bands_; i > 0; i--) {
        pyrUp(pyr_laplace[i], tmp, pyr_laplace[i-1].size());
        add(pyr_laplace[i-1], tmp, pyr_laplace[i-1]);
    }
    dst = pyr_laplace[0];
}

void MyMultiBandBlender::blend(Mat &dst) {
    // blend the images
    std::vector<Mat> pyr_laplace_dst;
    blendPyramids(pyr_laplace_[0], pyr_laplace_[1], pyr_gaussian_[0], pyr_gaussian_[1], pyr_laplace_dst);
    
    // reconstruct the image
    reconstruct(pyr_laplace_dst, dst_);

    // output reconstructed
    if (output_inner_) {
        imwrite("inner/blenders/reconstructed.png", dst_);
    }

    // convert the dst image to CV_8UC3
    dst_.convertTo(dst, CV_8UC3);
}