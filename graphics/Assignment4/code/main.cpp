#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN)// && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;
     window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
   
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    std::vector<cv::Point2f> vec;
    for (size_t i = 0; i < control_points.size(); i++)
    {
        if(i+1 >= control_points.size()){
            break;
        }
        cv::Point2f new_point = (1-t)* control_points[i]+t*control_points[i+1];
        vec.push_back(new_point);
    }
    cv::Point2f final_point;
    if(vec.size() == 1){
        final_point = vec[0];
    }else{
        final_point = recursive_bezier(vec,t);
    }

    
    return final_point;

}

void setColor(cv::Point2f &point, cv::Mat &window){
    auto x = point.x ;
    auto y = point.y ;
    cv::Point2f left_top = {floor(x)*1.0f,ceil(y)*1.0f};
    cv::Point2f right_top = {ceil(x)*1.0f,ceil(y)*1.0f};
    cv::Point2f right_bottom = {ceil(x)*1.0f,floor(y)*1.0f};
    cv::Point2f left_bottom = {floor(x)*1.0f,floor(y)*1.0f};

    float d1 = norm(point - right_top);
    float d2 = norm(point - left_top);
    float d3 = norm(point - left_bottom);
    float d4 = norm(point - right_bottom);
    window.at<cv::Vec3b>(right_top.y, right_top.x)[1] = MAX(window.at<cv::Vec3b>(right_top.y, right_top.x)[1], 255);
    window.at<cv::Vec3b>(left_top.y, left_top.x)[1] = MAX(window.at<cv::Vec3b>(left_top.y, left_top.x)[1], 255*(d1/d2));
    window.at<cv::Vec3b>(left_bottom.y, left_bottom.x)[1] = MAX(window.at<cv::Vec3b>(left_bottom.y, left_bottom.x)[1], 255*(d1/d3));
    window.at<cv::Vec3b>(right_bottom.y, right_bottom.x)[1] = MAX(window.at<cv::Vec3b>(right_bottom.y, right_bottom.x)[1], 255*(d1/d4));


}


void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    float t = 0.0f;
    std::vector<cv::Point2f>new_vec;
    while(t <= 1.0f){
        cv::Point2f point = recursive_bezier(control_points,t);
        new_vec.push_back(point);
        t += 0.0001;
    }
    
    for (size_t i = 0; i < new_vec.size(); i++)
    {
        cv::Point2f point = new_vec[i];
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
        setColor(point,window);
    }

}



int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (key == 'a')//control_points.size() == 4) 
        {
            //naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
