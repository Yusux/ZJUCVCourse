#include "canny.hpp"

/*
 * Discrete angle
 * return the sector index of the angle
 * @param 
 * @return: the sector index of the angle
 *         0: -pi/8 ~ pi/8 or -pi ~ -7pi/8 or 7pi/8 ~ pi
 *         1: -7pi/8 ~ -5pi/8 or pi/8 ~ 3pi/8
 *         2: -5pi/8 ~ -3pi/8 or 3pi/8 ~ 5pi/8
 *         3: -3pi/8 ~ -pi/8 or 5pi/8 ~ 7pi/8
 */
uchar giveSector(float angle) {
    // if the angle is negative, add pi to make it positive
    if (angle < 0) {
        angle += CV_PI;
    }

    // get the sector index of the angle
    uchar sector = 0;
    if (angle >= 0 && angle < CV_PI / 8.0) {
        sector = 0;
    } else if (angle >= CV_PI / 8.0 && angle < 3.0 * CV_PI / 8.0) {
        sector = 1;
    } else if (angle >= 3.0 * CV_PI / 8.0 && angle < 5.0 * CV_PI / 8.0) {
        sector = 2;
    } else if (angle >= 5.0 * CV_PI / 8.0 && angle < 7.0 * CV_PI / 8.0) {
        sector = 3;
    } else if (angle >= 7.0 * CV_PI / 8.0 && angle < CV_PI) {
        sector = 0;
    }

    return sector;
}

/*
 * Calculate the gradient magnitude and direction angle
 * @param source: input image
 * @param grad_m: gradient magnitude
 * @param dict_angle: gradient direction angle
 */
void calGradandAngle(Mat &source, Mat &grad_m, Mat &dict_angle) {
    float max_grad_m = 0;
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            if (i == source.rows - 1 || j == source.cols - 1) {
                // if the pixel is at the border,
                // set the gradient magnitude to the left pixel's value
                // to avoid abnormal values
                grad_m.at<uchar>(i, j) = grad_m.at<uchar>(i, j-1);
                continue;
            }
            float Gx = (mat_at(source, i, j+1) - mat_at(source, i, j) + mat_at(source, i+1, j+1) - mat_at(source, i+1, j)) / 2.0;
            float Gy = (mat_at(source, i, j) - mat_at(source, i+1, j) + mat_at(source, i, j+1) - mat_at(source, i+1, j+1)) / 2.0;
            
            // calculate gradient magnitude
            grad_m.at<uchar>(i, j) = (uchar)sqrtf32(Gx*Gx + Gy*Gy);
            // update max_grad_m
            if (grad_m.at<uchar>(i, j) > max_grad_m) {
                max_grad_m = grad_m.at<uchar>(i, j);
            }
            // calculate gradient direction angle
            dict_angle.at<float>(i, j) = atan2f32(Gy, Gx);
        }
    }

    // normalize the gradient magnitude
    for (int i = 0; i < grad_m.rows; i++) {
        for (int j = 0; j < grad_m.cols; j++) {
            grad_m.at<uchar>(i, j) = grad_m.at<uchar>(i, j) / max_grad_m * 255;
        }
    }
}

/*
 * Non-maximum suppression's Implementation.
 * If the pixel is not the local maximum
 * in the direction of the sector, set it to 0.
 * @param grad_m: gradient magnitude
 * @param dict_angle: gradient direction angle
 * @param dst: output image
 */
void nonMaxSuppression(Mat &grad_m, Mat &dict_angle, Mat &dst) {
    for (int i = 0; i < grad_m.rows; i++) {
        for (int j = 0; j < grad_m.cols; j++) {
            // get the sector index of the angle
            uchar sector = giveSector(dict_angle.at<float>(i, j));
            // get the gradient magnitude of the pixel
            uchar grad_mag = grad_m.at<uchar>(i, j);
            // get the max gradient magnitude of the pixel
            // in the direction of the sector to compare
            uchar grad_mag_cmp = 0;
            switch (sector) {
                /* be careful of the direction of the gradient
                 * ----→ Gx
                 *     |
                 *     ↓ Gy
                 * can be viewed as -pi/4, while
                 *     ↑ Gy
                 *     |
                 * ----→ Gx
                 * can be viewed as pi/4
                 */
                case 0: // 0: viewed as 0
                    grad_mag_cmp = max(mat_at(grad_m, i, j-1), mat_at(grad_m, i, j+1));
                    break;
                case 1: // 1: viewed as pi/4
                    grad_mag_cmp = max(mat_at(grad_m, i-1, j-1), mat_at(grad_m, i+1, j+1));
                    break;
                case 2: // 2: viewed as pi/2
                    grad_mag_cmp = max(mat_at(grad_m, i-1, j), mat_at(grad_m, i+1, j));
                    break;
                case 3: // 3: viewed as -pi/4
                    grad_mag_cmp = max(mat_at(grad_m, i-1, j+1), mat_at(grad_m, i+1, j-1));
                    break;
            }

            // if the pixel is not the local maximum, set it to 0
            if (grad_mag < grad_mag_cmp) {
                dst.at<uchar>(i, j) = 0;
            } else {
                dst.at<uchar>(i, j) = grad_mag;
            }
        }
    }
}

/*
 * Hysteresis thresholding's Step 1
 * Get the strong and weak edges
 * @param src: input image
 * @param strong_edges: output image (containing strong edges)
 * @param weak_edges: output image (containing weak edges)
 * @param lowThreshold: first threshold for the hysteresis procedure
 * @param highThreshold: second threshold for the hysteresis procedure
 */
void hysteresisThresholdingStep1(Mat &src, Mat &strong_edges, Mat &weak_edges, int lowThreshold, int highThreshold) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // if the pixel is larger than highThreshold, it is a strong edge
            if (mat_at(src, i, j) >= highThreshold) {
                strong_edges.at<uchar>(i, j) = 255;
                continue;
            } else {
                strong_edges.at<uchar>(i, j) = 0;
            }
            // if the pixel is larger than lowThreshold,
            // but smaller than highThreshold, it is a weak edge
            if (mat_at(src, i, j) >= lowThreshold) {
                weak_edges.at<uchar>(i, j) = 255;
            } else {
                weak_edges.at<uchar>(i, j) = 0;
            }
        }
    }
}

/*
 * Hysteresis thresholding's Step 2
 * Connect the weak edges to the strong edges to get the final edges
 * @param src: input image (containing weak edges)
 * @param dst: output image (containing strong edges)
 */
#ifdef USE_WEAK_EDGE
void hysteresisThresholdingStep2(Mat &src, Mat &dst) {
    // Step 2: connect the weak edges to the strong edges to get the final edges
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            // if the pixel is a weak edge
            if (mat_at(src, i, j) == 255) {
                // check if it has a strong edge in its 8-neighborhood
                bool has_strong_edge = false;
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        // if the pixel is a strong edge
                        if (mat_at(dst, i+k, j+l) == 255) {
                            has_strong_edge = true;
                            break;
                        }
                    }
                }
                // if it has a strong edge in its 8-neighborhood, set it to 255
                if (has_strong_edge) {
                    dst.at<uchar>(i, j) = 255;
                } else {
                    dst.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}
#else
void hysteresisThresholdingStep2(Mat &src, Mat &dst) {
    // Step 2: connect the weak edges to the strong edges to get the final edges
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            // if the pixel is a strong edge
            if (mat_at(dst, i, j) == 255) {
                // check if it has a strong edge in its 8-neighborhood
                bool has_strong_edge = false;
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        if (k == 0 && l == 0) {
                            continue;
                        }
                        // if the pixel is a strong edge
                        if (mat_at(dst, i+k, j+l) == 255) {
                            has_strong_edge = true;
                            break;
                        }
                    }
                }
                // if it has no strong edge in its 8-neighborhood,
                // copy the weak edge in its 8-neighborhood to the strong edge
                if (!has_strong_edge) {
                    for (int k = -1; k <= 1; k++) {
                        for (int l = -1; l <= 1; l++) {
                            // if the pixel is a weak edge
                            if (mat_at(src, i+k, j+l) == 255) {
                                if (i+k < 0 || i+k >= src.rows || j+l < 0 || j+l >= src.cols) {
                                    continue;
                                }
                                dst.at<uchar>(i+k, j+l) = 255;
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

void hysteresisThresholdingStep2_1(Mat &src, Mat &dst) {
    // Step 2: connect the weak edges to the strong edges to get the final edges
    for (int i = dst.rows - 1; i >= 0; i--) {
        for (int j = dst.cols - 1; j >= 0; j--) {
            // if the pixel is a weak edge
            if (mat_at(src, i, j) == 255) {
                // check if it has a strong edge in its 8-neighborhood
                bool has_strong_edge = false;
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        // if the pixel is a strong edge
                        if (mat_at(dst, i+k, j+l) == 255) {
                            has_strong_edge = true;
                            break;
                        }
                    }
                }
                // if it has a strong edge in its 8-neighborhood, set it to 255
                if (has_strong_edge) {
                    dst.at<uchar>(i, j) = 255;
                } else {
                    dst.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}

/*
 * Hysteresis thresholding's Implementation.
 * Divide into 2 parts:
 * 1. get the strong and weak edges
 * 2. connect the weak edges to the strong edges to get the final edges
 * @param src: input image
 * @param dst: output image
 * @param lowThreshold: first threshold for the hysteresis procedure
 * @param highThreshold: second threshold for the hysteresis procedure
 */
void hysteresisThresholding(Mat &src, OutputArray dst, int lowThreshold, int highThreshold) {
    // Step 1: get the strong and weak edges
    Mat strong_edges, weak_edges;
    strong_edges.create(src.size(), CV_8U);
    weak_edges.create(src.size(), CV_8U);
    hysteresisThresholdingStep1(src, strong_edges, weak_edges, lowThreshold, highThreshold);

    // Step 2: connect the weak edges to the strong edges to get the final edges
    hysteresisThresholdingStep2(weak_edges, strong_edges);
    hysteresisThresholdingStep2_1(weak_edges, strong_edges);

    // output the final edges
    dst.create(src.size(), CV_8U);
    dst.assign(strong_edges);
}

/*
 * Implemented myCanny function
 * (args are similar to Canny function)
 * @param image: input image (gray image)
 * @param edges: output image (containing edges)
 * @param highThreshold: low threshold for the hysteresis procedure
 * @param highThreshold: high threshold for the hysteresis procedure
 * @param apertureSize: aperture size for the Sobel operator, NOT USED
 * @param outputInner: a flag, indicating whether output the inner image
 * @param L2gradient: a flag, indicating whether a more accurate L2 norm
 */
void myCanny(InputArray image, OutputArray edges,
             double lowThreshold, double highThreshold,
             int apertureSize, bool outputInner,
             bool L2gradient) {
    // Step 1: blur the gray image
    Mat blurred;
    GaussianBlur(image, blurred, Size(3,3), 0.0);

    // Step 2: calculate gradient magnitude and direction angle
    Mat grad_m, dict_angle;
    // use uchar to store gradient magnitude
    // use float to store direction angle
    grad_m.create(blurred.size(), CV_8U);
    dict_angle.create(blurred.size(), CV_32F);
    calGradandAngle(blurred, grad_m, dict_angle);
    // if outputInner is true, output the inner image
    if (outputInner) {
        // output the inner image of grad magtinude
        imwrite("inner/grad.png", grad_m);
    }

    // Step 3: non-maximum suppression
    Mat non_max_sup;
    non_max_sup.create(grad_m.size(), CV_8U);
    nonMaxSuppression(grad_m, dict_angle, non_max_sup);
    // if outputInner is true, output the inner image
    if (outputInner) {
        // output the inner image of non-maximum suppression
        imwrite("inner/nms.png", non_max_sup);
    }

    // Step 4: hysteresis thresholding
    hysteresisThresholding(non_max_sup, edges, lowThreshold, highThreshold);
    // if outputInner is true, output the inner image
    if (outputInner) {
        // output the inner image of hysteresis thresholding
        imwrite("inner/hysteresis.png", edges);
    }
}