#include <iostream>
#include "utils.h"

int main() {
    Eigen::MatrixXd data = load_csv("../diabetes.csv", true);

    std::cout << data << std::endl;
}