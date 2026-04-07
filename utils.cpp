#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

/* -------------------- Load CSV -------------------- */
Eigen::MatrixXd load_csv(const string& path, bool has_header = true) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + path);
    }

    vector<double> matrixEntries;
    string line;
    int rowCount = 0;

    // Skip header if present
    if (has_header) {
        getline(file, line);  // read and discard
    }

    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            // Trim whitespace if any
            size_t start = cell.find_first_not_of(" \t\r\n");
            size_t end = cell.find_last_not_of(" \t\r\n");
            if (start != string::npos && end != string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            matrixEntries.push_back(stod(cell));
        }
        rowCount++;
    }

    int colCount = matrixEntries.size() / rowCount;
    Eigen::MatrixXd result(rowCount, colCount);
    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < colCount; ++j) {
            result(i, j) = matrixEntries[i * colCount + j];
        }
    }
    return result;
}

/* -------------------- Train Test Split -------------------- */
std::tuple<Eigen::MatrixXd, Eigen::VectorXd,Eigen::MatrixXd, Eigen::VectorXd>
train_test_split(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd labels,
    double test_ratio,
    bool shuffle,
    unsigned int seed
) {
    int n_samples = features.rows();
    int n_test = static_cast<int>(n_samples * test_ratio);
    int n_train = n_samples - n_test;

    // Create indices
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle if needed
    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // Split indices
    vector<int> train_indices(indices.begin(), indices.begin() + n_train);
    vector<int> test_indices(indices.begin() + n_train, indices.end());

    // Initialize X_train and y_train
    Eigen::MatrixXd X_train(n_train, features.cols());
    Eigen::VectorXd y_train(n_train);

    // Build train data
    for (int i = 0; i < n_train; i++) {
        X_train.row(i) = features.row(train_indices[i]);
        y_train.row(i) = labels.row(train_indices[i]);
    }   

    // Initialize X_train and y_train
    Eigen::MatrixXd X_test(n_test, features.cols());
    Eigen::VectorXd y_test(n_test);
    // Build test data
    for (int i = 0; i < n_test; i++) {
        X_test.row(i) = features.row(test_indices[i]);
        y_test.row(i) = labels.row(test_indices[i]);
    }

    return {X_train, y_train, X_test, y_test};
}