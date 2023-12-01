#include "stitching.hpp"
#include "blenders.hpp"
#include "utils.hpp"
#include <iostream>
#include <omp.h>
#include <ctime>
#include <iomanip>

using namespace cv;

MyStitcher::MyStitcher(std::vector<cv::String> filenames,
                       int blend_type,
                       int seam_finder_type,
                       bool output_inner) {
    CV_Assert(filenames.size() >= 2);
    CV_Assert(blend_type >= 0 && blend_type <= 4);
    CV_Assert(seam_finder_type >= 0 && seam_finder_type <= 5);

    output_inner_ = output_inner;
    blend_type_ = blend_type;
    filenames_ = filenames;

    // initialize sift
    sift_ = SIFT::create();

    // initialize matcher
    matcher_ = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    // initalize seam finder
    if (seam_finder_type == 0) {
        seam_finder_ = makePtr<detail::NoSeamFinder>();
    } else if (seam_finder_type == 1) {
        seam_finder_ = makePtr<detail::VoronoiSeamFinder>();
    } else if (seam_finder_type == 2) {
        std::cout << "Using dp color seam finder" << std::endl;
        seam_finder_ = makePtr<detail::DpSeamFinder>(detail::DpSeamFinder::COLOR);
    } else if (seam_finder_type == 3) {
        seam_finder_ = makePtr<detail::DpSeamFinder>(detail::DpSeamFinder::COLOR_GRAD);
    } else if (seam_finder_type == 4) {
        seam_finder_ = makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR);
    } else if (seam_finder_type == 5) {
        seam_finder_ = makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    } else {
        Exception e;
        e.msg = "Invalid seam finder type.";
        throw e;
    }

    try {
        clock_t start, end;
        // prepare the images
        start = clock();
        prepareImages();
        end = clock();
        std::cout << std::setw(28) << std::right << "Prepare images time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;
        // detect the features
        start = clock();
        detectFeatures();
        end = clock();
        std::cout << std::setw(28) << std::right << "Detect features time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;
    } catch (const Exception& e) {
        throw e;
    }
}

MyStitcher::~MyStitcher() {
}

void MyStitcher::stitch(Mat &dst) {
    clock_t start, end, total_start, total_end;
    std::cout << "Stitching based on " << filenames_[0] << std::endl;
    total_start = clock();
    for (int i = 1; i < filenames_.size(); i++) {
        std::cout << "Stitching " << filenames_[i] << " to the base image" << std::endl;
        start = clock();
        // Step 2: Feature Matching
        // **choose the best match image**
        // prepare the srcB from the dst
        Mat srcA, srcB;
        std::vector<KeyPoint> keypointsA, keypointsB;
        std::vector<DMatch> best_matches;
        featureMatching(srcA, srcB, keypointsA, keypointsB, best_matches);
        end = clock();
        std::cout << std::setw(28) << std::right << "Feature matching time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;

        // output the inner image
        if (output_inner_) {
            Mat keypointsA_img, keypointsB_img;
            drawKeypoints(srcA, keypointsA, keypointsA_img);
            drawKeypoints(srcB, keypointsB, keypointsB_img);
            imwrite("inner/" + std::to_string(i-1) + "_keypoints_A.png", keypointsA_img);
            imwrite("inner/" + std::to_string(i-1) + "_keypoints_B.png", keypointsB_img);
            Mat matches_img;
            drawMatches(srcA, keypointsA, srcB, keypointsB, best_matches, matches_img,
                        Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
                        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            imwrite("inner/" + std::to_string(i-1) + "_matches.png", matches_img);
        }

        // Step 3: Homography Estimation
        // use RANSAC to estimate the homography matrix
        start = clock();
        Mat H;
        try {
            homographyEstimation(srcA, srcB, keypointsA, keypointsB, best_matches, H);
        } catch (const Exception& e) {
            throw e;
        }
        end = clock();
        std::cout << std::setw(28) << std::right << "Homography estimation time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;

        // Step 4: Image Warping
        start = clock();
        Mat base, warpedA;
        imageWarping(srcA, srcB, H, base, warpedA);
        if (output_inner_) {
            imwrite("inner/" + std::to_string(i-1) + "_base.png", base);
            imwrite("inner/" + std::to_string(i-1) + "_warpedA.png", warpedA);
        }
        end = clock();
        std::cout << std::setw(28) << std::right << "Image warping time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;

        // Step 5: Image Blending
        start = clock();
        imageBlending(base, warpedA, i-1);
        end = clock();
        std::cout << std::setw(28) << std::right << "Image blending time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;

        // output the inner image
        if (output_inner_) {
            imwrite("inner/" + std::to_string(i-1) + "_stitiched.png", dst_);
        }
    }


    // Step Final: cut the black part
    start = clock();
    Mat final_mask;
    getMask(dst_, final_mask);
    Rect final_rect;
    getMaxInnerRect(final_mask, final_rect);
    dst_ = dst_(final_rect);
    end = clock();
    // align the cout of the time
    std::cout << std::setw(28) << std::right << "Cut the black part time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(end - start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;

    // output the final image
    dst = dst_;

    // output the total time
    total_end = clock();
    std::cout << std::setw(28) << std::right << "Total time: " << std::fixed << std::setprecision(6) << std::setw(10) << std::left << (double)(total_end - total_start) / CLOCKS_PER_SEC << std::right << "s" << std::endl;
}

void MyStitcher::prepareImages() {
    // Step 0: Prepare the images
    // convert the images to grayscale
    // read the all the images
    // check if the images are read successfully
    // and convert the images to grayscale
    images_.resize(filenames_.size()-1);
    images_gray_.resize(filenames_.size()-1);
    for (int i = 1; i < filenames_.size(); i++) {
        Mat img = imread(filenames_[i]);
        checkImg(img.data, filenames_[i]);
        images_[i-1] = img;
        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        images_gray_[i-1] = img_gray;
    }
    dst_ = imread(filenames_[0]);
    checkImg(dst_.data, filenames_[0]);
}

void MyStitcher::detectFeatures() {
    // Step 1: Feature Detection
    // use SIFT to detect keypoints and extract descriptors
    images_keypoints_.resize(filenames_.size()-1);
    images_descriptors_.resize(filenames_.size()-1);
    for (int i = 0; i < filenames_.size()-1; i++) {
        std::vector<KeyPoint> keypoints;
        Mat descriptors;
        sift_->detectAndCompute(images_gray_[i], Mat(), keypoints, descriptors);
        images_keypoints_[i] = keypoints;
        images_descriptors_[i] = descriptors;
    }
}

void MyStitcher::featureMatching(Mat &srcA, Mat &srcB, 
                                 std::vector<KeyPoint> &keypointsA,
                                 std::vector<KeyPoint> &keypointsB,
                                 std::vector<DMatch> &best_matches) {
    // Step 2: Feature Matching
    // **choose the best match image**
    // prepare the srcB from the dst
    srcB = dst_;
    Mat descriptorsB;
    sift_->detectAndCompute(srcB, Mat(), keypointsB, descriptorsB);

    // use BFMatcher to match the descriptors
    // the knnmatch is not the knn used to classify
    // it is the k nearest neighbors (in the trainImage)
    // of each descriptor in the queryImage
    // refer to https://docs.opencv.org/4.x/db/d39/classcv_1_1DescriptorMatcher.html#aa880f9353cdf185ccf3013e08210483a
    // BFMatcher matcher(NORM_L2, false);
    // use FlannBasedMatcher to match the descriptors to speed up
    int idxA = 0;
#pragma omp parallel for
    for (int j = 0; j < images_descriptors_.size(); j++) {
        std::vector<std::vector<DMatch>> matches;
        matcher_->knnMatch(images_descriptors_[j], descriptorsB, matches, 2);
        // add ratio test, refer to Figure 11 in paper
        // Distinctive Image Features from Scale-Invariant Keypoints
        // > In other documents (https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html),
        // > it is also mentioned that
        // > (BFMatcher) Second param is boolean variable,
        // > crossCheck which is false by default. If it is true
        // > ... It provides consistent result, and is a good alternative
        // > to ratio test proposed by D.Lowe in SIFT paper.
        std::vector<DMatch> good_matches;
        for (int k = 0; k < matches.size(); k++) {
            if (matches[k][0].distance < 0.75 * matches[k][1].distance) {
                good_matches.push_back(matches[k][0]);
            }
        }
#pragma omp critical
        if (good_matches.size() > best_matches.size()) {
            best_matches = good_matches;
            idxA = j;
        }
    }

    // prepare the srcA from the best match image
    srcA = images_[idxA];
    keypointsA = images_keypoints_[idxA];

    // pop the idxA image
    images_.erase(images_.begin() + idxA);
    images_gray_.erase(images_gray_.begin() + idxA);
    images_keypoints_.erase(images_keypoints_.begin() + idxA);
    images_descriptors_.erase(images_descriptors_.begin() + idxA);
}

void MyStitcher::homographyEstimation(Mat &srcA, Mat &srcB,
                                      std::vector<KeyPoint> &keypointsA,
                                      std::vector<KeyPoint> &keypointsB,
                                      std::vector<DMatch> &best_matches,
                                      Mat &H) {
    // Step 3: Homography Estimation
    // use RANSAC to estimate the homography matrix
    if (best_matches.size() < 4) {
        Exception e;
        e.msg = "Not enough matches.";
        throw e;
    }
    std::vector<Point2f> srcA_pts, srcB_pts;
    for (int j = 0; j < best_matches.size(); j++) {
        srcA_pts.push_back(keypointsA[best_matches[j].queryIdx].pt);
        srcB_pts.push_back(keypointsB[best_matches[j].trainIdx].pt);
    }
    H = findHomography(srcA_pts, srcB_pts, RANSAC, 3.0);
}

void MyStitcher::imageWarping(Mat &srcA, Mat &srcB, Mat &H,
                              Mat &base, Mat &warped) {
    // Step 4: Image Warping
    // Step 4.1: Calculate the size of the canvas
    // refer to https://stackoverflow.com/q/31978721
    std::vector<Point2f> srcA_corners(4);
    srcA_corners[0] = Point2f(0, 0);
    srcA_corners[1] = Point2f(srcA.cols, 0);
    srcA_corners[2] = Point2f(srcA.cols, srcA.rows);
    srcA_corners[3] = Point2f(0, srcA.rows);
    std::vector<Point2f> projected_corners(4);
    perspectiveTransform(srcA_corners, projected_corners, H);
    // get the final padding size
    std::vector<int> projected_bounds(4);
    projected_bounds[0] = floorMin(projected_corners[0].y, projected_corners[1].y,
                                   projected_corners[2].y, projected_corners[3].y); // top
    projected_bounds[1] = ceilMax(projected_corners[0].y, projected_corners[1].y,
                                  projected_corners[2].y, projected_corners[3].y);  // bottom
    projected_bounds[2] = floorMin(projected_corners[0].x, projected_corners[1].x,
                                   projected_corners[2].x, projected_corners[3].x); // left
    projected_bounds[3] = ceilMax(projected_corners[0].x, projected_corners[1].x,
                                  projected_corners[2].x, projected_corners[3].x);  // right
    std::vector<int> padding_size(4);
    padding_size[0] = std::max(0, -projected_bounds[0]);                            // top
    padding_size[1] = std::max(0, projected_bounds[1] - srcB.rows);                 // bottom
    padding_size[2] = std::max(0, -projected_bounds[2]);                            // left
    padding_size[3] = std::max(0, projected_bounds[3] - srcB.cols);                 // right
    // get the final canvas size
    int canvas_width = srcB.cols + padding_size[2] + padding_size[3];
    int canvas_height = srcB.rows + padding_size[0] + padding_size[1];

    // Step 4.2: Warp the images
    // apply Ht as the transformation for moving the image
    // refer to https://stackoverflow.com/q/13063201
    /*
     * [1 0 tx]
     * [0 1 ty]
     * [0 0 1 ]
     * where tx = padding_size[2], ty = padding_size[0]
     */
    Mat Ht = Mat::eye(3, 3, CV_64F);
    Ht.at<double>(0, 2) = padding_size[2];
    Ht.at<double>(1, 2) = padding_size[0];
    H = Ht * H;

    // use the homography matrix to warp the images
    warpPerspective(srcA, warped, H, Size(canvas_width, canvas_height), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    
    // Step 4.3: Copy the images to the canvas
    // create the canvas
    base = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    // copy the base image to the dst image with the mask
    srcB.copyTo(Mat(base, Rect(padding_size[2], padding_size[0], srcB.cols, srcB.rows)));
}

void MyStitcher::imageBlending(Mat &base, Mat &warped, int idx) {
    CV_Assert(base.size() == warped.size() && base.type() == warped.type());
    int canvas_width = base.cols;
    int canvas_height = base.rows;

    CV_Assert(blend_type_ >= 0 && blend_type_ <= 4);
    if (blend_type_ == NO_BLEND) {
        // 0. not use blending
        // copy the warped image to the dst image with the mask
        // create mask for the warped image
        Mat maskA;
        getMask(warped, maskA, 1, 2);
        dst_ = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
        base.copyTo(Mat(dst_, Rect(0, 0, base.cols, base.rows)));
        warped.copyTo(Mat(dst_, Rect(0, 0, warped.cols, warped.rows)), maskA);
    } else if (blend_type_ == LINEAR_BLEND) {
        // 1. use linear blend to blend the base and warped image
        linearBlend(base, warped, dst_);
    } else if (blend_type_ == ALPHA_BLEND) {
        // 4. use alpha blending to blend the base and warped image
        // refer to https://stackoverflow.com/a/22324790
        Mat maskA, maskB;
        getMask(warped, maskA, 1, 2);
        getMask(base, maskB);
        dst_ = computeAlphaBlending(warped, maskA, base, maskB);
    } else if (blend_type_ == MY_MULTIBAND_BLEND || blend_type_ == OPENCV_MULTIBAND_BLEND) {
        // get the masks and the roi
        Mat base_mask, warped_mask;
        getMask(base, base_mask);
        getMask(warped, warped_mask, 1, 2);
        Mat overlap_mask = base_mask & warped_mask;
        Rect roi = boundingRect(overlap_mask);

        // get the images, corners and masks
        std::vector<Mat> images;
        images.resize(2);
        images[0] = base;
        images[1] = warped;
        std::vector<Point> corners;
        corners.resize(2);
        corners[0] = roi.tl();
        corners[1] = roi.tl();
        std::vector<Mat> masks;
        masks.resize(2);
        masks[0] = base_mask;
        masks[1] = warped_mask;

        // find the seam
        seamFind(images, corners, masks);

        dilate(masks[0], masks[0], Mat());
        masks[0] = masks[0] & base_mask;
        dilate(masks[1], masks[1], Mat());
        masks[1] = masks[1] & warped_mask;


        // output the inner image
        if (output_inner_) {
            for (int i = 0; i < images.size(); i++) {
                imwrite("inner/" + std::to_string(idx) + "_mask_" + std::to_string(i) + ".png", masks[i]);
            }
        }

        if (blend_type_ == MY_MULTIBAND_BLEND) {
            // 2. use self implemented multi-band blending to blend the base and warped image
            MyMultiBandBlender blender(images, masks, output_inner_);
            blender.blend(dst_);
        } else  {   // OPENCV_MULTIBAND_BLEND
            // 3. use opencv multi-band blending to blend the base and warped image
            // create the blender
            detail::MultiBandBlender blender(false);
            blender.prepare(Rect(0, 0, canvas_width, canvas_height));
            blender.feed(images[0], masks[0], Point(0, 0));
            blender.feed(images[1], masks[1], Point(0, 0));
            blender.blend(dst_, Mat());
            dst_.convertTo(dst_, CV_8UC3);
        }
    } else {
        Exception e;
        e.msg = "Invalid blend type.";
        throw e;
    }
}


void MyStitcher::seamFind(std::vector<cv::Mat> &images,
                          std::vector<cv::Point> &corners,
                          std::vector<cv::Mat> &masks) {
    // use the graph cut to find the seam
    std::vector<UMat> seam_images;
    std::vector<Point> seam_corners;
    std::vector<UMat> seam_masks;
    seam_images.resize(images.size());
    seam_corners.resize(images.size());
    seam_masks.resize(images.size());
    for (int i = 0; i < images.size(); i++) {
        images[i].convertTo(seam_images[i], CV_32F);
        seam_masks[i] = masks[i].getUMat(ACCESS_RW);
        seam_corners[i] = corners[i];
    }

    // find the seam
    seam_finder_->find(seam_images, seam_corners, seam_masks);
}