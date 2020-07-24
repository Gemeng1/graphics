//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

     Eigen::Vector3f getColor(Eigen::Vector2f uv)
    {
        auto u_img = (uv.x()/width) * width;
        auto v_img = (1 - (uv.y()/height)) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u,float v){
        auto u_img = u * width;
        auto v_img = v * height;
        Eigen::Vector2f left_top = {floor(u_img)*1.f,ceil(v_img)*1.f};
        Eigen::Vector2f right_top = {ceil(u_img)*1.f,ceil(v_img)*1.f};
        Eigen::Vector2f right_bottom = {ceil(u_img)*1.f,floor(v_img)*1.f};
        Eigen::Vector2f left_bottom = {floor(u_img)*1.f,floor(v_img)*1.f};
        float r1 = u_img-left_bottom.x();
        float r2 = v_img-left_bottom.y();
        Eigen::Vector3f c1 = lerp(getColor(left_top),getColor(right_top),r1);
        Eigen::Vector3f c2 = lerp(getColor(left_bottom),getColor(right_bottom),r1);
        return lerp(c2,c1,r2);
    }

    Eigen::Vector3f lerp(Eigen::Vector3f color1,Eigen::Vector3f color2,float radio){
        return color1+(color2-color1)*radio;
    }


};
#endif //RASTERIZER_TEXTURE_H
