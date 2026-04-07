#pragma once
#include <Eigen/Dense>
#include <string>

Eigen::MatrixXd load_csv(const std::string& path, bool has_header);