#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

class SignDetector : public rclcpp::Node {
public:
    SignDetector() : Node("sign_detector") {
        // Subscribe to camera feed
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            std::bind(&SignDetector::imageCallback, this, std::placeholders::_1));
        
        // Publish detected signs (e.g., "parking", "stop", "arrow", "checker")
        publisher_ = this->create_publisher<std_msgs::msg::String>("/detected_sign", 10);
        
        RCLCPP_INFO(this->get_logger(), "Sign Detector Node Started");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto sign_msg = std_msgs::msg::String();
        sign_msg.data = "none";  // Default: no sign detected

        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            cv::Mat hsv, hsv_channels[3];

            // Now convert to HSV
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            //debugMaskPixels(hsv, frame);


            // Replace (x, y) with a pixel coordinate from your yellow sign
            /*
            int x = 320;  // Example X-coordinate (adjust)
            int y = 240;  // Example Y-coordinate (adjust)
            if (x < frame.cols && y < frame.rows) {
                cv::Vec3b hsv_pixel = hsv.at<cv::Vec3b>(y, x);
                RCLCPP_INFO(this->get_logger(), "HSV at (%d,%d): H=%d, S=%d, V=%d", 
                            x, y, hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]);
            }
            */

            // Color masks (adjust HSV ranges as needed)
            // Updated HSV ranges (looser constraints for S/V)

            cv::Mat blurred;
            cv::GaussianBlur(hsv, blurred, cv::Size(5, 5), 0);

            cv::Mat yellow_mask, green_mask, red_mask, noise_mask_green, noise_mask_red;

            //cv::inRange(hsv, cv::Scalar(145, 128, 109), cv::Scalar(180, 255, 255), red_mask);  // Main red range       // Red (low)
            cv::inRange(hsv, cv::Scalar(101, 120, 255), cv::Scalar(180, 230, 255), red_mask);  // Main red range       // Red (low)

            // Noise mask for red (H:7-13)
            //cv::inRange(hsv, cv::Scalar(7, 154, 46), cv::Scalar(13, 255, 255), noise_mask_red);

            // Remove noise from low red range using XOR
            //cv::bitwise_xor(red_mask, noise_mask_red, red_mask);

            cv::Mat kernel_red = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));  // Bigger than green's (5x5)
            //cv::Mat kernel_red2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));  // Bigger than green's (5x5)
            cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel_red);   // Remove small noise first
            cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel_red);  // Fill larger holes
            
           
            


            
            //cv::inRange(hsv, cv::Scalar(160, 70, 50), cv::Scalar(180, 255, 255), red_mask_high); // Red (high)
            //red_mask |= red_mask_high;


   
            cv::inRange(hsv, cv::Scalar(0, 28, 94), cv::Scalar(104, 255, 255), yellow_mask);


            // Original green mask (your desired range)
            //cv::inRange(hsv, cv::Scalar(53, 141, 43), cv::Scalar(95, 255, 255), green_mask);
            cv::inRange(hsv, cv::Scalar(55, 11, 106), cv::Scalar(128, 255, 255), green_mask);
            // Noise mask (range you want to exclude)
            //cv::inRange(hsv, cv::Scalar(86, 141, 43), cv::Scalar(95, 255, 255), noise_mask_green);
            //cv::inRange(hsv, cv::Scalar(43, 212, 119), cv::Scalar(85, 250, 255), noise_mask_green);

            // Remove overlapping noise using XOR
            //cv::bitwise_xor(green_mask, noise_mask_green, green_mask);

            cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel1);  // Fill holes
            cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel1);   // Remove small noise


            // Additional cleanup for yellow (optional but recommended)
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(yellow_mask, yellow_mask, cv::MORPH_CLOSE, kernel);  // Fill small holes
            cv::morphologyEx(yellow_mask, yellow_mask, cv::MORPH_OPEN, kernel);   // Remove small noise


            cv::Mat yellow_debug = cv::Mat::zeros(frame.size(), CV_8UC3);
            cv::Mat green_debug = cv::Mat::zeros(frame.size(), CV_8UC3);

            // Apply color to the masks for better visualization
            yellow_debug.setTo(cv::Scalar(0, 255, 255), yellow_mask); // Yellow in BGR
            green_debug.setTo(cv::Scalar(0, 255, 0), green_mask);     // Green in BGR

            // Add text labels
            cv::putText(yellow_debug, "Yellow Mask", cv::Point(20, 50), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            cv::putText(green_debug, "Green Mask", cv::Point(20, 50), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            // Show the individual masks
            cv::imshow("Yellow Mask Debug", yellow_debug);
            cv::imshow("Green Mask Debug", green_debug);

            // --- Add Red Mask Debug (new) ---
            cv::Mat red_debug = cv::Mat::zeros(frame.size(), CV_8UC3);
            red_debug.setTo(cv::Scalar(0, 0, 255), red_mask);  // Red in BGR
            cv::putText(red_debug, "Red Mask", cv::Point(20, 50), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Red Mask Debug", red_debug);

            // --- Update Combined Mask to Include Red ---
            cv::Mat combined_mask = yellow_mask | green_mask | red_mask;  // Now includes red
            cv::Mat combined_debug = cv::Mat::zeros(frame.size(), CV_8UC3);
            combined_debug.setTo(cv::Scalar(0, 255, 255), yellow_mask);  // Yellow
            combined_debug.setTo(cv::Scalar(0, 255, 0), green_mask);     // Green
            combined_debug.setTo(cv::Scalar(0, 0, 255), red_mask);       // Red (new)
            cv::putText(combined_debug, "Combined Mask (Y+G+R)", cv::Point(20, 50), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Combined Mask Debug", combined_debug);

            // Reordered logic (red/green first)
            // Reordered logic (red/green first) - FILLED IN
            if (cv::countNonZero(red_mask) > 100) {
                if (cv::countNonZero(green_mask) > 200) {  // Checker has red + green
                    sign_msg.data = "checker";
                    RCLCPP_INFO(this->get_logger(), "Detected: Checker Sign");
                } else {  // Only red → Stop sign
                    sign_msg.data = "stop";
                    RCLCPP_INFO(this->get_logger(), "Detected: Stop Sign");
                }
            } 

            else if (cv::countNonZero(green_mask) > 200) {
                cv::Rect arrow_rect;  // Declare rectangle to store arrow bounding box
                std::string arrow_dir = detectTriangles(green_mask, arrow_rect);  // Pass both parameters
                if (arrow_dir != "none") {
                    sign_msg.data = "arrow_" + arrow_dir;
                    RCLCPP_INFO(this->get_logger(), "Detected: Arrow (%s)", arrow_dir.c_str());
                    
                    // Optional: Draw arrow bounding box for debugging
                    /**/
                    
                }
            }
            else if (cv::countNonZero(yellow_mask) > 400) {
                sign_msg.data = "parking";
                RCLCPP_INFO(this->get_logger(), "Detected: Parking Sign");
            }
            
            

            // Publish the detected sign
            publisher_->publish(sign_msg);

            // --- Debugging Visualizations ---
            cv::Mat debug_frame;
            cv::bitwise_and(frame, frame, debug_frame, yellow_mask | green_mask | red_mask);  // Overlay masks
            cv::putText(debug_frame, sign_msg.data, cv::Point(20, 50), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("Debug View", debug_frame);
            cv::waitKey(1);

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
        }
    }

    // Detect triangle orientation (simplified example)
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
                double dist = norm(p - center);
                if (dist > max_dist) {
                    max_dist = dist;
                    tip = p;
                }
            }
            
            // Determine orientation based on tip position
            cv::Point diff = tip - center;
            is_horizontal = (abs(diff.x) > abs(diff.y));

             // Debug visualization
             /*
            cv::Mat debug = cv::Mat::zeros(mask.size(), CV_8UC3);
            cv::drawContours(debug, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0,255,0), 2);
            cv::circle(debug, center, 5, cv::Scalar(255,0,0), -1);
            cv::circle(debug, tip, 5, cv::Scalar(0,0,255), -1);
            
            std::string orientation = is_horizontal ? "Horizontal" : "Vertical";
            cv::putText(debug, orientation, cv::Point(10,30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 1);
            cv::imshow("Arrow Analysis", debug);
            */

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
                    return "right";
                } else {
                    return "left";
                }
            } else {
                /*
                // Vertical arrow - compare top/bottom halves
                cv::Mat top_half = mask(cv::Rect(out_rect.x, out_rect.y, 
                                            out_rect.width, out_rect.height/2));
                cv::Mat bottom_half = mask(cv::Rect(out_rect.x, 
                                                out_rect.y + out_rect.height/2, 
                                                out_rect.width, out_rect.height/2));

                int top_pixels = cv::countNonZero(top_half);
                int bottom_pixels = cv::countNonZero(bottom_half);

                // Cross-validate with tip position
                bool tip_is_top = (tip.y < center.y);
                bool pixels_suggest_top = (top_pixels < bottom_pixels);
                */
                return "up";
                // Final decision
                /*
                if (pixels_suggest_top) {
                    return "down";
                } else {
                    return "up";
                }
                    */
            }

           
        }
        return "none";
    }


        // New debug function to visualize mask pixels
    void debugMaskPixels(const cv::Mat& hsv, const cv::Mat& frame) {
        // Create trackbars for HSV range adjustment
        cv::namedWindow("HSV Adjuster", cv::WINDOW_NORMAL);
        
        // Initial values for yellow (adjust as needed)
        int h_low = 20, h_high = 30;
        int s_low = 100, s_high = 255;
        int v_low = 100, v_high = 255;
        
        cv::createTrackbar("H Low", "HSV Adjuster", &h_low, 180);
        cv::createTrackbar("H High", "HSV Adjuster", &h_high, 180);
        cv::createTrackbar("S Low", "HSV Adjuster", &s_low, 255);
        cv::createTrackbar("S High", "HSV Adjuster", &s_high, 255);
        cv::createTrackbar("V Low", "HSV Adjuster", &v_low, 255);
        cv::createTrackbar("V High", "HSV Adjuster", &v_high, 255);

        while (true) {
            // Create mask with current trackbar values
            cv::Mat mask;
            cv::inRange(hsv, 
                       cv::Scalar(h_low, s_low, v_low), 
                       cv::Scalar(h_high, s_high, v_high), 
                       mask);

            // Show the mask
            cv::imshow("Current Mask", mask);

            // Create a visualization showing which pixels are being captured
            cv::Mat debug_img = frame.clone();
            debug_img.setTo(cv::Scalar(0, 255, 255), mask); // Highlight masked pixels in yellow
            
            // Add text showing current HSV range
            std::string range_text = cv::format("H: %d-%d, S: %d-%d, V: %d-%d", 
                                               h_low, h_high, s_low, s_high, v_low, v_high);
            cv::putText(debug_img, range_text, cv::Point(20, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            
            // Show count of masked pixels
            int pixel_count = cv::countNonZero(mask);
            std::string count_text = cv::format("Pixels: %d", pixel_count);
            cv::putText(debug_img, count_text, cv::Point(20, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            
            cv::imshow("Masked Pixels", debug_img);

            // Check for key press
            int key = cv::waitKey(30);
            if (key == 27) { // ESC to exit
                break;
            }
            else if (key == 's') { // 's' to sample a pixel
                samplePixelHSV(hsv);
            }
        }
        cv::destroyWindow("HSV Adjuster");
        cv::destroyWindow("Current Mask");
        cv::destroyWindow("Masked Pixels");
    }

    // Helper function to sample a pixel's HSV values
    void samplePixelHSV(const cv::Mat& hsv) {
        cv::namedWindow("Sample Pixel", cv::WINDOW_NORMAL);
        cv::setMouseCallback("Sample Pixel", [](int event, int x, int y, int flags, void* userdata) {
            if (event == cv::EVENT_LBUTTONDOWN) {
                cv::Mat* hsv_ptr = (cv::Mat*)userdata;
                if (x < hsv_ptr->cols && y < hsv_ptr->rows) {
                    cv::Vec3b hsv_pixel = hsv_ptr->at<cv::Vec3b>(y, x);
                    std::cout << "Sampled HSV at (" << x << "," << y << "): "
                              << "H=" << (int)hsv_pixel[0] << ", "
                              << "S=" << (int)hsv_pixel[1] << ", "
                              << "V=" << (int)hsv_pixel[2] << std::endl;
                }
            }
        }, (void*)&hsv);

        // Create a temporary window for sampling
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::imshow("Sample Pixel", bgr);
        cv::waitKey(0);
        cv::destroyWindow("Sample Pixel");
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