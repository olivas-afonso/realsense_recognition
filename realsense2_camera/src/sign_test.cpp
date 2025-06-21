#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

class SignDetector : public rclcpp::Node {
public:
    SignDetector() : Node("sign_detector") {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            std::bind(&SignDetector::imageCallback, this, std::placeholders::_1));
        
        publisher_ = this->create_publisher<std_msgs::msg::String>("/detected_sign", 10);
        RCLCPP_INFO(this->get_logger(), "Sign Detector Node Started");
    }

private:
    struct SignDetection {
        std::vector<cv::Point> quad;
        double area;
        std::string type;
    };

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto sign_msg = std_msgs::msg::String();
        sign_msg.data = "none";

        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

            // Step 1: Create both black masks
            cv::Mat hsv, black_mask_dark, black_mask_light;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            
            // Dark condition mask (works in normal lighting)
            cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 144, 150), black_mask_dark);
            
            // Light condition mask (works when closer to signs)
            cv::inRange(hsv, cv::Scalar(47, 9, 0), cv::Scalar(149, 166, 204), black_mask_light);

            // Process each mask separately
            std::vector<SignDetection> dark_detections = processMask(frame, black_mask_dark);
            std::vector<SignDetection> light_detections = processMask(frame, black_mask_light);

            // Combine results from both masks
            std::vector<SignDetection> all_detections;
            all_detections.insert(all_detections.end(), dark_detections.begin(), dark_detections.end());
            all_detections.insert(all_detections.end(), light_detections.begin(), light_detections.end());

            // Find the best detection (largest area)
            if (!all_detections.empty()) {
                // Sort detections by area (descending)
                std::sort(all_detections.begin(), all_detections.end(),
                    [](const SignDetection& a, const SignDetection& b) {
                        return a.area > b.area;
                    });
                
                // Use the detection with largest area
                SignDetection best_detection = all_detections[0];
                sign_msg.data = best_detection.type;
                
                // Print the detected sign on every frame
                RCLCPP_INFO(this->get_logger(), "Detected sign: %s", best_detection.type.c_str());
            }

            publisher_->publish(sign_msg);

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
        }
    }

    std::vector<SignDetection> processMask(const cv::Mat& frame, const cv::Mat& mask) {
        std::vector<SignDetection> detections;
        
        // Process the mask
        cv::Mat processed_mask;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
        cv::morphologyEx(mask, processed_mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(processed_mask, processed_mask, cv::MORPH_OPEN, kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(processed_mask.clone(), contours, hierarchy, 
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Parameters for contour filtering
        const double MIN_CONTOUR_AREA = 1000;
        const double MAX_ASPECT_RATIO = 1.5;
        const double EXPECTED_ASPECT = 1.25;

        // Process each contour
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < MIN_CONTOUR_AREA) continue;

            cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
            
            // Check aspect ratio
            float width = std::max(rotated_rect.size.width, rotated_rect.size.height);
            float height = std::min(rotated_rect.size.width, rotated_rect.size.height);
            float aspect_ratio = width / height;
            if (aspect_ratio > MAX_ASPECT_RATIO || aspect_ratio < 0.8) continue;

            // Check size
            if (width < 100 || height < 80) continue;
            
            // Validate sign panel
            if (!isPotentialSignPanel(frame, processed_mask, contour)) continue;
            
            // Get perfect rectangle
            std::vector<cv::Point> perfect_rect = enforcePerfectRectangle(contour, frame.size());
            
            // Get ROI and analyze content
            cv::Mat sign_roi = getQuadrilateralROI(frame, perfect_rect);
            cv::Mat sign_mask_roi = getQuadrilateralROI(processed_mask, perfect_rect);
            
            // Invert mask to get inner symbol
            cv::Mat inner_symbol_mask;
            cv::bitwise_not(sign_mask_roi, inner_symbol_mask);
            
            // Clean up the mask
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(inner_symbol_mask, inner_symbol_mask, cv::MORPH_OPEN, kernel);
            
            // Analyze sign content
            std::string sign_type = analyzeSignContent(sign_roi, inner_symbol_mask);
            
            if (sign_type != "none") {
                detections.push_back({perfect_rect, area, sign_type});
            }
        }

        return detections;
    }

    std::vector<cv::Point> enforcePerfectRectangle(const std::vector<cv::Point>& contour, cv::Size frameSize) {
        // Get the minimal area rectangle
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        
        // Convert to 4 perfect corners
        cv::Point2f rect_points[4];
        rotated_rect.points(rect_points);
        
        // Convert to integer points
        std::vector<cv::Point> perfect_rect;
        for (int i = 0; i < 4; i++) {
            perfect_rect.push_back(rect_points[i]);
        }
        
        // Get the rotation angle
        float angle = rotated_rect.angle;
        if (rotated_rect.size.width < rotated_rect.size.height) {
            angle += 90.f;
        }
        
        // Order points with angle information
        return orderQuadrilateralPoints(perfect_rect);
    }

    std::vector<cv::Point> refineQuadrilateralCorners(const std::vector<cv::Point>& quad, const cv::Size& frameSize) {
        const float EXPECTED_ASPECT = 34.0f / 27.0f; // Known physical aspect ratio
        const float ASPECT_TOLERANCE = 0.2f; // Allow 20% aspect ratio variation
        
        // 1. Get initial rotated rectangle
        cv::RotatedRect rotatedRect = cv::minAreaRect(quad);
        
        // 2. Adjust rectangle to match known aspect ratio
        float currentAspect = rotatedRect.size.width / rotatedRect.size.height;
        if (currentAspect > EXPECTED_ASPECT * (1 + ASPECT_TOLERANCE)) {
            // Too wide - adjust height
            rotatedRect.size.height = rotatedRect.size.width / EXPECTED_ASPECT;
        } else if (currentAspect < EXPECTED_ASPECT * (1 - ASPECT_TOLERANCE)) {
            // Too tall - adjust width
            rotatedRect.size.width = rotatedRect.size.height * EXPECTED_ASPECT;
        }
        
        // 3. Get the four corners of the adjusted rectangle
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);
        
        // 4. Find closest points in original contour to these ideal corners
        std::vector<cv::Point> refinedCorners(4);
        for (int i = 0; i < 4; i++) {
            double minDist = std::numeric_limits<double>::max();
            for (const auto& pt : quad) {
                double dist = cv::norm(cv::Point2f(pt) - rectPoints[i]);
                if (dist < minDist) {
                    minDist = dist;
                    refinedCorners[i] = pt;
                }
            }
        }
        
        // 5. Special handling for bottom points (known to be more reliable)
        // Calculate expected bottom point positions based on known aspect ratio
        float width = cv::norm(refinedCorners[1] - refinedCorners[0]);
        float expectedHeight = width / EXPECTED_ASPECT;
        
        // Adjust bottom points vertically to match expected height
        cv::Point2f topCenter = (cv::Point2f(refinedCorners[0]) + cv::Point2f(refinedCorners[1])) * 0.5f;
        cv::Point2f bottomCenter = (cv::Point2f(refinedCorners[2]) + cv::Point2f(refinedCorners[3])) * 0.5f;
        
        float currentHeight = cv::norm(bottomCenter - topCenter);
        if (abs(currentHeight - expectedHeight) > 5) { // 5 pixel tolerance
            float scale = expectedHeight / currentHeight;
            cv::Point2f vec = bottomCenter - topCenter;
            bottomCenter = topCenter + vec * scale;
            
            // Adjust bottom points accordingly
            cv::Point2f bottomVec = (cv::Point2f(refinedCorners[2]) - cv::Point2f(refinedCorners[3])) * 0.5f;
            refinedCorners[2] = bottomCenter + bottomVec;
            refinedCorners[3] = bottomCenter - bottomVec;
        }
        
        return refinedCorners;
    }

    // Add this helper function to your class:
    cv::Mat getQuadrilateralROI(const cv::Mat& frame, const std::vector<cv::Point>& quad) {
        // First, ensure the points are ordered consistently
        std::vector<cv::Point> ordered_quad = orderQuadrilateralPoints(quad);
        
        // Calculate width as max of top and bottom edge lengths
        double width1 = cv::norm(ordered_quad[1] - ordered_quad[0]);
        double width2 = cv::norm(ordered_quad[2] - ordered_quad[3]);
        double width = std::max(width1, width2);
        
        // Calculate height as max of left and right edge lengths
        double height1 = cv::norm(ordered_quad[3] - ordered_quad[0]);
        double height2 = cv::norm(ordered_quad[2] - ordered_quad[1]);
        double height = std::max(height1, height2);
        
        // Destination points for perspective transform
        std::vector<cv::Point2f> dst_pts = {
            {0, 0},
            {static_cast<float>(width), 0},
            {static_cast<float>(width), static_cast<float>(height)},
            {0, static_cast<float>(height)}
        };
        
        // Convert source points to Point2f
        std::vector<cv::Point2f> src_pts;
        for (const auto& p : ordered_quad) src_pts.emplace_back(p);
        
        // Get perspective transform and apply it
        cv::Mat transform = cv::getPerspectiveTransform(src_pts, dst_pts);
        cv::Mat roi;
        cv::warpPerspective(frame, roi, transform, cv::Size(width, height));
        
        // Additional check for frontal view (when camera has no yaw)
        if (width1 > height1 * 1.5 || width2 > height2 * 1.5) {
            // If the sign appears much wider than tall, it's likely a frontal view
            // Ensure the ROI is in landscape orientation
            if (roi.rows > roi.cols) {
                cv::rotate(roi, roi, cv::ROTATE_90_CLOCKWISE);
            }
        } else {
            // For angled views, maintain the detected orientation
            if (roi.cols < roi.rows) {
                cv::rotate(roi, roi, cv::ROTATE_90_CLOCKWISE);
            }
        }
        
        return roi;
    }
    // Helper function to order quadrilateral points consistently
    std::vector<cv::Point> orderQuadrilateralPoints(const std::vector<cv::Point>& quad) {
        if (quad.size() != 4) return quad;

        // Find the center point
        cv::Point2f center(0, 0);
        for (const auto& p : quad) center += cv::Point2f(p);
        center *= 0.25f;

        // Sort points based on their angle from center (clockwise)
        std::vector<std::pair<float, cv::Point>> angle_points;
        for (const auto& p : quad) {
            cv::Point2f vec = cv::Point2f(p) - center;
            float angle = std::atan2(vec.y, vec.x);
            angle_points.emplace_back(angle, p);
        }

        // Sort points by angle (clockwise order)
        std::sort(angle_points.begin(), angle_points.end(),
            [](const std::pair<float, cv::Point>& a, const std::pair<float, cv::Point>& b) {
                return a.first > b.first; // Sort angles in descending order
            });

        // Extract just the points in order
        std::vector<cv::Point> ordered;
        for (const auto& ap : angle_points) {
            ordered.push_back(ap.second);
        }

        // Now ensure the first point is top-left (minimum x + y)
        int min_sum = ordered[0].x + ordered[0].y;
        int min_idx = 0;
        for (int i = 1; i < 4; i++) {
            int current_sum = ordered[i].x + ordered[i].y;
            if (current_sum < min_sum) {
                min_sum = current_sum;
                min_idx = i;
            }
        }

        // Rotate the vector so the top-left point comes first
        std::rotate(ordered.begin(), ordered.begin() + min_idx, ordered.end());

        // Verify the order is top-left, top-right, bottom-right, bottom-left
        // If not, adjust the ordering
        cv::Point vec1 = ordered[1] - ordered[0]; // Top edge
        cv::Point vec2 = ordered[2] - ordered[1]; // Right edge
        if (vec1.x * vec2.y - vec1.y * vec2.x < 0) {
            // If the cross product is negative, points are in CCW order
            // Swap points 3 and 1 to make it CW
            std::swap(ordered[1], ordered[3]);
        }

        return ordered;
    }

    /*
    cv::Mat extractInnerSymbol(const cv::Mat& sign_roi) {
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(sign_roi, gray, cv::COLOR_BGR2GRAY);
        cv::imshow("5a. Grayscale", gray);

        cv::Mat gray_eq;
        cv::equalizeHist(gray, gray_eq);
        cv::imshow("5a. Grayscale - Equalized", gray_eq);

        
        // Apply adaptive thresholding - often works better than global threshold
        cv::Mat thresh;
        cv::adaptiveThreshold(gray_eq, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                            cv::THRESH_BINARY_INV, 11, 5);
        
        // Clean up the mask - use closing first to fill gaps, then opening to remove noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);
        cv::imshow("5c. After Morphology", thresh);
        
        // Find contours of the inner symbol
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Create visualization images
        cv::Mat contour_visualization = cv::Mat::zeros(sign_roi.size(), CV_8UC3);
        cv::Mat inner_mask = cv::Mat::zeros(sign_roi.size(), CV_8UC1);
        
        if (!contours.empty()) {
            // Find largest contour (the actual symbol)
            auto largest_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });
            
            // Draw the largest contour on the visualization
            cv::drawContours(contour_visualization, std::vector<std::vector<cv::Point>>{largest_contour}, 
                            -1, cv::Scalar(0, 255, 0), 2);
            
            // Create the mask from the largest contour
            cv::drawContours(inner_mask, std::vector<std::vector<cv::Point>>{largest_contour}, 
                            -1, cv::Scalar(255), cv::FILLED);
            
            // Also show the original ROI with the contour overlay
            cv::Mat roi_with_contour = sign_roi.clone();
            cv::drawContours(roi_with_contour, std::vector<std::vector<cv::Point>>{largest_contour}, 
                            -1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("5d. ROI with Contour", roi_with_contour);
        }
        
        // Show contour visualization on black background
        cv::imshow("5e. Contour Visualization", contour_visualization);
        
        // Show final inner mask
        cv::imshow("5f. Final Inner Mask", inner_mask);
        
        return inner_mask;
    }
    */

    cv::Mat extractInnerSymbol(const cv::Mat& sign_roi, const cv::Mat& full_black_mask, const cv::Rect& roi_rect) {
        // Extract the corresponding region from the processed black mask
        cv::Mat roi_black_mask = full_black_mask(roi_rect).clone();
        
        // Invert the mask (black border becomes 0, inner symbol becomes 255)
        cv::Mat inner_symbol_mask;
        cv::bitwise_not(roi_black_mask, inner_symbol_mask);
        
        // Clean up the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(inner_symbol_mask, inner_symbol_mask, cv::MORPH_OPEN, kernel);
        
        // Visualizations
        cv::imshow("5a. ROI Black Mask", roi_black_mask);
        cv::imshow("5b. Inverted Mask", inner_symbol_mask);
        
        return inner_symbol_mask;
    }

    cv::Rect getConservativeROI(const cv::Rect& original_rect, float shrink_factor = 0.8) {
        // Calculate new dimensions
        int new_width = original_rect.width * shrink_factor;
        int new_height = original_rect.height * shrink_factor;
        
        // Calculate new top-left corner to keep it centered
        int new_x = original_rect.x + (original_rect.width - new_width) / 2;
        int new_y = original_rect.y + (original_rect.height - new_height) / 2;
        
        return cv::Rect(new_x, new_y, new_width, new_height);
    }

    // Add this helper function to your class:
    bool isPotentialSignPanel(const cv::Mat& frame, const cv::Mat& black_mask, const std::vector<cv::Point>& contour) {
        // 1. Get the bounding rectangle of the contour
        cv::Rect boundRect = cv::boundingRect(contour);
        
        // 2. Get a conservative center region (50% of width/height)
        int centerWidth = boundRect.width * 0.5;
        int centerHeight = boundRect.height * 0.5;
        cv::Rect centerRect(
            boundRect.x + boundRect.width/2 - centerWidth/2,
            boundRect.y + boundRect.height/2 - centerHeight/2,
            centerWidth,
            centerHeight
        );
        
        // Ensure the center rect is within image bounds
        centerRect &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (centerRect.area() <= 0) return false;
        
        // 3. Check for specific sign colors in the center region
        cv::Mat centerRegion = frame(centerRect);
        cv::Mat hsvCenter;
        cv::cvtColor(centerRegion, hsvCenter, cv::COLOR_BGR2HSV);
        
        // Create masks for each sign color (using same thresholds as analyzeSignContent)
        cv::Mat yellow_mask, green_mask, red_mask;
        
        // Yellow (parking signs)
        cv::inRange(hsvCenter, cv::Scalar(21, 40, 164), cv::Scalar(58, 255, 255), yellow_mask);
        // Green (arrow signs)
        cv::inRange(hsvCenter, cv::Scalar(40, 40, 40), cv::Scalar(85, 255, 255), green_mask);
        // Red (stop signs and checker patterns)
        cv::inRange(hsvCenter, cv::Scalar(0, 70, 95), cv::Scalar(16, 255, 255), red_mask);
        
        // Combine all color masks
        cv::Mat combined_mask;
        cv::bitwise_or(yellow_mask, green_mask, combined_mask);
        cv::bitwise_or(combined_mask, red_mask, combined_mask);
        
        // Count colored pixels in center
        int coloredPixelCount = cv::countNonZero(combined_mask);
        int totalPixels = centerRect.width * centerRect.height;
        
        // The center should have significant sign color content (at least 20%)
        float coloredPercentage = static_cast<float>(coloredPixelCount) / totalPixels;
        return coloredPercentage > 0.1;
    }


    std::string analyzeSignContent(const cv::Mat& sign_roi, const cv::Mat& inner_mask) {
        // Convert to HSV for color analysis
        cv::Mat hsv;
        cv::cvtColor(sign_roi, hsv, cv::COLOR_BGR2HSV);

        // Define color ranges (adjust as needed)
        cv::Mat yellow_mask, green_mask, red_mask;

        // Yellow (parking signs)
        cv::inRange(hsv, cv::Scalar(21, 40, 164), cv::Scalar(58, 255, 255), yellow_mask);
        cv::bitwise_and(yellow_mask, inner_mask, yellow_mask);
        
        // Green (arrow signs)
        cv::inRange(hsv, cv::Scalar(40, 40, 40), cv::Scalar(85, 255, 255), green_mask);
        cv::bitwise_and(green_mask, inner_mask, green_mask);
        
        // Red (stop signs and checker patterns)
        cv::inRange(hsv, cv::Scalar(0, 70, 95), cv::Scalar(16, 255, 255), red_mask);
        
        cv::bitwise_and(red_mask, inner_mask, red_mask);

        // Calculate the percentage of each color in the ROI
        double total_pixels = cv::countNonZero(inner_mask);
        if (total_pixels == 0) return "none";
        
        double yellow_percent = cv::countNonZero(yellow_mask) / total_pixels;
        double green_percent = cv::countNonZero(green_mask) / total_pixels;
        double red_percent = cv::countNonZero(red_mask) / total_pixels;

        // Check for different sign types based on color content
        if (yellow_percent > 0.2) {
            return "parking";
        } else if (red_percent > 0.1 && green_percent > 0.1) { 
            return "checker";
        } else if (green_percent > 0.2) {
            // Use the new triangle detection for arrows
            cv::Rect arrow_rect;
            return detectTriangles(green_mask, arrow_rect);
        } else if (red_percent > 0.2) {
            return "stop";
        }

        return "none";
    }

    
    std::string detectTriangles(const cv::Mat& mask, cv::Rect& out_rect) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 500) continue;

            // Get bounding rect
            out_rect = cv::boundingRect(contour);

            // Calculate centroid
            cv::Moments m = cv::moments(contour);
            cv::Point center(m.m10/m.m00, m.m01/m.m00);

            // METHOD 1: Tip detection to determine orientation
            bool is_horizontal = false;
            cv::Point tip;
            double max_dist = 0;
            
            // Find furthest point from center
            for (const auto& p : contour) {
                double dist = cv::norm(p - center);
                if (dist > max_dist) {
                    max_dist = dist;
                    tip = p;
                }
            }
            
            // Determine orientation based on tip position
            cv::Point diff = tip - center;
            is_horizontal = (abs(diff.x) > abs(diff.y));

            // METHOD 2: Pixel distribution to verify direction
            if (is_horizontal) {
                // Horizontal arrow - compare left/right halves
                cv::Mat left_half = mask(cv::Rect(out_rect.x, out_rect.y, 
                                        out_rect.width/2, out_rect.height));
                cv::Mat right_half = mask(cv::Rect(out_rect.x + out_rect.width/2, 
                                        out_rect.y, out_rect.width/2, out_rect.height));

                int left_pixels = cv::countNonZero(left_half);
                int right_pixels = cv::countNonZero(right_half);

                // Cross-validate with tip position
                bool tip_is_left = (tip.x < center.x);
                bool pixels_suggest_left = (left_pixels < right_pixels);

                // Final decision (give more weight to pixel distribution)
                if (pixels_suggest_left) {
                    return "arrow_right";
                } else {
                    return "arrow_left";
                }
            } else {
                return "arrow_up";
            }
        }
        return "none";
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SignDetector>());
    rclcpp::shutdown();
    return 0;
}