#include <iostream>
#include <torch/torch.h>
#include "utils.h"
#include "preprocess.h"
#include "model.h"
#include "dataset.h"

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

    // Split into train and test
    auto [X_train, y_train, X_test, y_test] = train_test_split(
        features,
        labels,
        0.2,
        true,
        42
    );
    
    // std::cout << "Training set: " << X_train.rows() << " samples" << std::endl;
    // std::cout << "Test set: " << X_test.rows() << " samples" << std::endl;

    // Z-Score Normalize the data
    Scaler scaler;                // stack allocation
    Eigen::MatrixXd X_train_scaled = scaler.fit_transform(X_train);
    Eigen::MatrixXd X_test_scaled = scaler.transform(X_test);

    // std::cout << X_train_scaled << std::endl;
    // std::cout << X_test_scaled << std::endl;

    // Convert to tensor (float32)
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

    // Create custom datasets
    auto train_dataset = DiabetesDataset(X_train_tensor, y_train_tensor);
    auto test_dataset = DiabetesDataset(X_test_tensor, y_test_tensor);

    // Create data loaders directly
    int batch_size = 32;

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::samplers::RandomSampler(train_dataset.size().value()),
        batch_size
    );

    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::samplers::SequentialSampler(test_dataset.size().value()),
        batch_size
    );

    // Create model instance
    auto model = std::make_shared<SimpleMLP>(
        X_train_scaled.cols(),
        std::stoi(config["hidden_size"]),
        1,
        std::stof(config["dropout"])
    );



}