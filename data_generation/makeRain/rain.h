#ifndef RAIN_H
#define RAIN_H

#include <map>
#include <string>
#include <vector>

#include <armadillo>
#include <opencv2/opencv.hpp>


class Rain {
    public:
        Rain() = delete;
        Rain(std::map<std::string, double> params, std::string image_path);
        void render(const std::string mode="sphere");
        void blur(const cv::Mat &kernel);
        cv::Mat get_kernel(int diameter=7);
        

    private:
        arma::Mat<double> get_intrinsic(const std::string &image_path);
        arma::Row<double> to_glass(const double &x, const double &y);
        double w_in_plane(const double &u, const double &v);
        void get_sphere_raindrop(const int &W, const int &H);
        void get_bezier_raindrop(const int &W, const int &H);
        int in_sphere_raindrop(const int &x, const int &y);
        int in_bezier_raindrop(const int &x, const int &y);
        void get_orth_bezier_normal(const arma::Row<double> &p_glass,
                const arma::Row<double> &i_pg, const int & id, arma::Row<double> &p_w, arma::Row<double> &normal_w);
        arma::Row<double> to_sphere_section_env(const int &x, const int &y, const int &id);
        arma::Row<double> to_orth_bezier_env(const int &x, const int &y, const int &id);
        auto cubic_bezier_curve(arma::vec pt1, arma::vec pt2, arma::vec pt3, arma::vec pt4);
        
    private:
        double M;
        double B;
        double psi;
        double n_water = 1.33;
        double n_air = 1.0;
        double gamma;

        arma::Mat<double> intrinsic;
        arma::Row<double> normal;
        arma::Row<double> o_g;

        std::vector<arma::Row<double>> centers;
        std::vector<double> radius;
        // positions on the glass
        std::vector<double> g_radius;
        // [g_center, g_size, g_ratio] define a oval on the glass
        std::vector<arma::Row<double>> g_centers;
        std::vector<arma::Row<double>> g_sizes;
        std::vector<double> g_angles;
        std::vector<arma::Mat<double>> g_h_curves;
        std::vector<arma::Mat<double>> g_w_curves;

    public:
        cv::Mat image;
        cv::Mat mask;
        cv::Mat rain_image;
        cv::Mat blur_image;
};

#endif
