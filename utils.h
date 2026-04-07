#pragma once
#include <Eigen/Dense>
#include <string>

Eigen::MatrixXd load_csv(const std::string& path, bool has_header);

std::unordered_map<std::string, std::string> load_config(const std::string& path);