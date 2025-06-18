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
        
        // Create debug windows
        cv::namedWindow("1. Original Image", cv::WINDOW_NORMAL);
        cv::namedWindow("2. Black Mask (Raw)", cv::WINDOW_NORMAL);
        cv::namedWindow("3. Black Mask (Processed)", cv::WINDOW_NORMAL);
        cv::namedWindow("4. Contours Detection", cv::WINDOW_NORMAL);
        cv::namedWindow("5. Inner Symbol Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("6. Final Detection", cv::WINDOW_NORMAL);
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto sign_msg = std_msgs::msg::String();
        sign_msg.data = "none";

        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            
            // Show original image
            cv::imshow("1. Original Image", frame);

            // Step 1: Convert to HSV and create black mask
            cv::Mat hsv, black_mask;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 80, 100), black_mask);
            cv::imshow("2. Black Mask (Raw)", black_mask);

            // Step 2: Clean up the mask
            cv::Mat processed_mask;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(black_mask, processed_mask, cv::MORPH_CLOSE, kernel);
            cv::morphologyEx(processed_mask, processed_mask, cv::MORPH_OPEN, kernel);
            cv::imshow("3. Black Mask (Processed)", processed_mask);

            // Step 3: Find contours
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(processed_mask.clone(), contours, hierarchy, 
                            cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Draw all contours for debugging
            cv::Mat contours_img = cv::Mat::zeros(frame.size(), CV_8UC3);
            cv::drawContours(contours_img, contours, -1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("4. Contours Detection", contours_img);

            // Variables to track the largest valid rectangle
            double max_area = 0;
            cv::Rect largest_rect;
            bool found_sign = false;
            cv::Mat final_detection = frame.clone();

            // Process each potential sign to find the largest valid rectangle
            for (const auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area < 500) continue; // Skip small contours

                // Approximate the contour to a polygon
                std::vector<cv::Point> approx;
                double peri = cv::arcLength(contour, true);
                cv::approxPolyDP(contour, approx, 0.02 * peri, true);

                // Check if it's a quadrilateral (rectangle)
                if (approx.size() == 4) {
                    // Get the bounding rectangle
                    cv::Rect boundRect = cv::boundingRect(approx);

                    // Draw all potential rectangles in blue
                    cv::rectangle(final_detection, boundRect, cv::Scalar(255, 0, 0), 2);

                    // Check if this is the largest rectangle we've found so far
                    if (area > max_area) {
                        max_area = area;
                        largest_rect = boundRect;
                        found_sign = true;
                    }
                }
            }

            // If we found a valid sign rectangle, analyze its content
            if (found_sign) {
                // Get a conservative ROI that's 80% of original size
                cv::Rect conservative_rect = getConservativeROI(largest_rect, 0.88);
                cv::Mat sign_roi = frame(conservative_rect);

                // Pass the processed_mask and conservative_rect to extractInnerSymbol
                cv::Mat inner_symbol_mask = extractInnerSymbol(sign_roi, processed_mask, conservative_rect);
                cv::imshow("5. Inner Symbol Mask", inner_symbol_mask);

                // Now analyze the sign content using the inner symbol mask
                std::string sign_type = analyzeSignContent(sign_roi, inner_symbol_mask);
                if (sign_type != "none") {
                    sign_msg.data = sign_type;
                    cv::putText(final_detection, sign_type, 
                               cv::Point(largest_rect.x, largest_rect.y - 10), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                }
            }

            // Show final detection
            cv::imshow("6. Final Detection", final_detection);

            // Publish the detected sign
            publisher_->publish(sign_msg);

            cv::waitKey(1);

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
        }
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


    std::string analyzeSignContent(const cv::Mat& sign_roi, const cv::Mat& inner_mask) {
        // Convert to HSV for color analysis
        cv::Mat hsv;
        cv::cvtColor(sign_roi, hsv, cv::COLOR_BGR2HSV);

        // Define color ranges (adjust as needed)
        cv::Mat yellow_mask, green_mask, red_mask;

        // Yellow (parking signs)
        cv::inRange(hsv, cv::Scalar(33, 59, 0), cv::Scalar(58, 255, 255), yellow_mask);
        cv::bitwise_and(yellow_mask, inner_mask, yellow_mask);
        
        // Green (arrow signs)
        cv::inRange(hsv, cv::Scalar(40, 40, 40), cv::Scalar(80, 255, 255), green_mask);
        cv::bitwise_and(green_mask, inner_mask, green_mask);
        
        // Red (stop signs and checker patterns)
        cv::inRange(hsv, cv::Scalar(0, 70, 93), cv::Scalar(60, 255, 255), red_mask);
        cv::bitwise_and(red_mask, inner_mask, red_mask);

        // Calculate the percentage of each color in the ROI
        double total_pixels = cv::countNonZero(inner_mask);
        if (total_pixels == 0) return "none";
        
        double yellow_percent = cv::countNonZero(yellow_mask) / total_pixels;
        double green_percent = cv::countNonZero(green_mask) / total_pixels;
        double red_percent = cv::countNonZero(red_mask) / total_pixels;

        // Check for different sign types based on color content
        if (yellow_percent > 0.3) {
            return "parking";
        } else if (red_percent > 0.2 && green_percent > 0.2) { 
            return "checker";
        } else if (green_percent > 0.3) {
            // Use the new triangle detection for arrows
            cv::Rect arrow_rect;
            return detectTriangles(green_mask, arrow_rect);
        } else if (red_percent > 0.3) {
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