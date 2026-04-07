#include <iostream>
#include <torch/torch.h>
#include "utils.h"
#include "preprocess.h"

using namespace std;

int main() {

    // Load Config file
    auto config = load_config("../config.txt");
    string data_path = config["data_path"];
    
    // Load dataset
    Eigen::MatrixXd data = load_csv(data_path, true);
    cout << data;

    int rows = data.rows();
    int cols = data.cols();

    Eigen::MatrixXd features = data.leftCols(cols - 1);
    Eigen::VectorXd labels = data.rightCols(1);

    // SPlit into train and test
    auto [X_train, y_train, X_test, y_test] = train_test_split(
        features,
        labels,
        0.2,
        true,
        42
    );
    
    std::cout << "Training set: " << X_train.rows() << " samples" << std::endl;
    std::cout << "Test set: " << X_test.rows() << " samples" << std::endl;

    // Z-Score Normalize the data
    Scaler scaler;                // stack allocation
    Eigen::MatrixXd X_train_scaled = scaler.fit_transform(X_train);
    Eigen::MatrixXd X_test_scaled = scaler.transform(X_test);

    std::cout << X_train_scaled << std::endl;
    std::cout << X_test_scaled << std::endl;

    // Convert to torch::Tensor (float32)
    torch::Tensor X_train_tensor = torch::from_blob(
        X_train_scaled.data(),
        {X_train_scaled.rows(), X_train_scaled.cols()},
        torch::kFloat64
    ).clone().to(torch::kFloat32);

    torch::Tensor y_train_tensor = torch::from_blob(
        y_train.data(),
        {y_train.size()},
        torch::kFloat64
    ).clone().to(torch::kFloat32).reshape({-1, 1});

    torch::Tensor X_test_tensor = torch::from_blob(
        X_test_scaled.data(),
        {X_test_scaled.rows(), X_test_scaled.cols()},
        torch::kFloat64
    ).clone().to(torch::kFloat32);

    torch::Tensor y_test_tensor = torch::from_blob(
        y_test.data(),
        {y_test.size()},
        torch::kFloat64
    ).clone().to(torch::kFloat32).reshape({-1, 1});

}