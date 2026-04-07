#pragma once
#include <Eigen/Dense>
#include <string>
#include <random>
#include <tuple>

Eigen::MatrixXd load_csv(const std::string& path, bool has_header);


std::tuple<Eigen::MatrixXd, Eigen::VectorXd,Eigen::MatrixXd, Eigen::VectorXd>
train_test_split(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd labels,
    double test_ratio = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
);