#include <iostream>
#include "utils.h"
#include "preprocess.h"


int main() {
    Eigen::MatrixXd data = load_csv("../diabetes.csv", true);

    int rows = data.rows();
    int cols = data.cols();

    Eigen::MatrixXd features = data.leftCols(cols - 1);
    Eigen::VectorXd labels = data.rightCols(1);


    auto [X_train, y_train, X_test, y_test] = train_test_split(
        features,
        labels,
        0.2,
        true,
        42
    );
    
    std::cout << "Training set: " << X_train.rows() << " samples" << std::endl;
    std::cout << "Test set: " << X_test.rows() << " samples" << std::endl;

    Scaler scaler;                // stack allocation
    Eigen::MatrixXd X_train_scaled = scaler.fit_transform(X_train);
    Eigen::MatrixXd X_test_scaled = scaler.transform(X_test);

    std::cout << X_train_scaled << std::endl;
    std::cout << X_test_scaled << std::endl;
}