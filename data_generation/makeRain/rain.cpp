#include "rain.h"
#include <cmath>
#include <regex>
#include <jsoncpp/json/json.h>
#include <random>

//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wunused-variable"
using namespace std;
using namespace arma;

Rain::Rain(map<string, double> params, string image_path) {
    M = params["M"];
    B = params["B"];
    psi = params["psi"] / 180.0 * M_PI;
    gamma = asin(n_air / n_water);

    image = cv::imread(image_path);
    intrinsic = get_intrinsic(image_path);
    normal = Row<double> {0.0, -1.0 * cos(psi), sin(psi)};
    o_g = (normal(2) * M / dot(normal, normal)) * normal;
}

Mat<double> Rain::get_intrinsic(const string &image_path) {
    string json_path;
    json_path = regex_replace(image_path, regex(R"(leftImage)"), "camera");
    json_path = regex_replace(json_path, regex(R"(leftImg8bit.png$)"), "camera.json");
    //json_path = regex_replace(image_path, regex(R"(.png$)"), ".json");

    ifstream stream(json_path, ifstream::binary);
    Json::Value root;
    stream >> root;
    root = root.get("intrinsic", 0);

    intrinsic = zeros<mat>(3, 3);
    intrinsic(0, 0) = root.get("fx", 0).asDouble();
    intrinsic(1, 1) = root.get("fy", 0).asDouble();
    intrinsic(2, 2) = 1.0;
    intrinsic(0, 2) = root.get("u0", 0).asDouble();
    intrinsic(1, 2) = root.get("v0", 0).asDouble();

    return intrinsic;
}

Row<double> Rain::to_glass(const double &x, const double &y) {
    double w = M * tan(psi) / (tan(psi) - (y - intrinsic(1, 2)) / intrinsic(1, 1));
    double u = w * (x - intrinsic(0, 2)) / intrinsic(0, 0);
    double v = w * (y - intrinsic(1, 2)) / intrinsic(1, 1);

    return Row<double> {u, v, w};
}

double Rain::w_in_plane(const double &u, const double &v) {
    return (normal(2)*M - normal(0)*u - normal(1)*v) / normal(2);
}

void Rain::get_sphere_raindrop(const int &W, const int &H) {
    g_centers.clear();
    g_radius.clear();
    centers.clear();
    radius.clear();


    auto left_upper = to_glass(0, 0);
    auto left_bottom = to_glass(0, H);
    auto right_upper = to_glass(W, 0);
    auto right_bottom = to_glass(W, H);

    mt19937 rng;
    rng.seed(random_device()());
    uniform_int_distribution<mt19937::result_type> random_rain(100, 200);
    uniform_int_distribution<mt19937::result_type> random_tau(30, 45);
    uniform_real_distribution<double> random_loc(0.0, 1.0);

    int n = random_rain(rng);
    for(int i = 0; i < n; i++) {
        double u = left_bottom(0) + (right_bottom(0) - left_bottom(0)) * random_loc(rng);
        double v = left_upper(1) + (right_bottom(1) - left_upper(1)) * random_loc(rng);
//        double u = left_bottom(0) + (right_bottom(0) - left_bottom(0)) * 0.5;
//        double v = left_upper(1) + (right_bottom(1) - left_upper(1)) * 0.1;
        double w = w_in_plane(u, v);

        double tau = random_tau(rng);
        tau  = tau / 180 * M_PI;

        double glass_r = 0.8 + 0.6 * random_loc(rng);

        // raindrop radius in sky dataset
        double r_sphere = glass_r / sin(tau);

        Row<double> g_c  {u, v, w};
        Row<double> c = g_c - normal * r_sphere * cos(tau);

        g_centers.push_back(move(g_c));
        g_radius.push_back(glass_r);
        centers.push_back(move(c));
        radius.push_back(r_sphere);
    }
}


int Rain::in_sphere_raindrop(const int &x, const int &y) {
    auto p = to_glass(x, y);

    for(uint i = 0; i < g_centers.size(); i++) {
        if(norm(p - g_centers[i]) <= g_radius[i])
            return i;
    }
    return -1;
}

/**
 * Using the sphere section model
 * @param x
 * @param y
 * @param id    The id of rain drops
 * @return
 */
Row<double> Rain::to_sphere_section_env(const int &x, const int &y, const int &id) {
    Row<double> center = centers[id];
    double r_sphere = radius[id];

    Row<double> p_g = to_glass(x, y);

    double alpha = acos(dot(p_g, normal) / norm(p_g));
    double beta = asin(n_air * sin(alpha) / n_water);

    Row<double> po = p_g - o_g;
    po = po / norm(po);
    Row<double> i_1 = normal + tan(beta) * po;
    i_1 = i_1 / norm(i_1);

    Row<double> oc = p_g - center;
    double tmp = dot(i_1, oc);
    double d = -(tmp) + sqrt(pow(tmp, 2.0) - dot(oc, oc) + pow(r_sphere, 2.0));
    Row<double> p_w = p_g + d * i_1;

    Row<double> normal_w = p_w - center;
    normal_w = normal_w / norm(normal_w);

    d = (dot(p_w, normal_w) - dot(normal_w, p_g)) / dot(normal_w, normal_w);
    Row<double> p_a = p_w - (d * normal_w + p_g);
    p_a = p_a / norm(p_a);

    double eta = acos(dot(normal_w, p_w - p_g) / norm(p_w - p_g));
    if(eta >= gamma)
        throw "total refrection";

//    std::cout << "eta: " << eta << ", gamma: " << gamma << " (" << x << ", " << y << "), id: " << id << std::endl;

    double theta = asin(n_water * sin(eta) / n_air);
    Row<double> i_2 = normal_w + tan(theta) * p_a;

    Row<double> p_e = p_w + ((B - p_w[2]) / i_2[2]) * i_2;
    Row<double> p_i = trans(intrinsic * trans(p_e) / B);
    p_i = round(p_i);
    return p_i;
}

/**
 * Render the rain-drop image
 * @param mode in [sphere, bezier, random]
 */
void Rain::render(const std::string mode) {
    int h = image.rows;
    int w = image.cols;
    rain_image = image.clone();
    mask = cv::Mat(h, w, CV_8UC1, cv::Scalar(0));

    get_sphere_raindrop(w, h);
    Row<double> p;
    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            int i;
            i = in_sphere_raindrop(x, y);
            if(i != -1) {
                try {
                    p = to_sphere_section_env(x, y, i);
                } catch (const char* msg) {
                    rain_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    mask.at<char>(y, x) = (char)255;
                    continue;
                }
                int u = p(0), v = p(1);
                if(u >= w)
                    u = w - 1;
                else if(u < 0)
                    u = 0;

                if(v >= h)
                    v = h - 1;
                else if(v < 0)
                    v = 0;

                rain_image.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(v, u);
                mask.at<char>(y, x) = char(255);
            }
        }
    }
}

cv::Mat Rain::get_kernel(int diameter) {
    cv::Mat kernel(diameter, diameter, CV_32FC1, cv::Scalar(0.0));
    double radius = diameter / 2.0;
    int count = 0;
    for(int i = 0;  i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            if(pow(i - radius, 2.0) + pow(j - radius, 2.0) < pow(radius, 2.0)) {
                kernel.at<float>(i, j) = 1.0;
                ++count;
            }
        }
    }
    kernel /=  count;
    return kernel;
}

void Rain::blur(const cv::Mat &kernel) {
    blur_image = image.clone();
    cv::Mat blured;
    cv::filter2D(rain_image, blured, -1, kernel);

    blured = blured * 1.2;
    blured.copyTo(blur_image, mask);
}

#pragma clang diagnostic pop