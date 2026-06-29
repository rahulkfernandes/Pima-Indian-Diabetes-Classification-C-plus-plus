#include "preprocess.h"
#include <vector>

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
    std::vector<int> train_indices(indices.begin(), indices.begin() + n_train);
    std::vector<int> test_indices(indices.begin() + n_train, indices.end());

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

/* -------------------- Normalization -------------------- */
void Scaler::fit(const Eigen::MatrixXd& X) {
    int n = X.rows();
    int m = X.cols();

    // Column means as row vector
    mean_ = X.colwise().mean();

    // Compute column standard deviations (sample std)
    Eigen::RowVectorXd variance(m);
    for (int j = 0; j < m; ++j) {
        double sum_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = X(i, j) - mean_(j);
            sum_sq += diff * diff;
        }
        variance(j) = sum_sq / (n - 1);  // sample variance
    }
    scale_ = variance.array().sqrt();

    // Avoid division by zero
    for (int j = 0; j < m; ++j) {
        if (scale_(j) < eps) scale_(j) = 1.0;
    }

    fitted_ = true;
}

Eigen::MatrixXd Scaler::transform(const Eigen::MatrixXd& X) const {
    assert(fitted_ && "Scaler must be fitted before transform.");
    // (X - mean_) / scale_   (row-wise broadcasting)
    return (X.rowwise() - mean_).array().rowwise() / scale_.array();
}

Eigen::MatrixXd Scaler::fit_transform(const Eigen::MatrixXd& X) {
    fit(X);
    return transform(X);
}

/* -------------------- Matrix to Tensor -------------------- */
std::tuple<torch::Tensor, torch::Tensor> convert_to_tensor(Eigen::MatrixXd X, Eigen::VectorXd y) {
    torch::Tensor X_tensor = torch::from_blob(
        X.data(),
        {X.rows(), X.cols()},
        torch::kFloat64
    ).clone().to(torch::kFloat32);

    torch::Tensor y_tensor = torch::from_blob(
        y.data(),
        {y.size()},
        torch::kFloat64
    ).clone().to(torch::kFloat32).reshape({-1, 1});

    return {X_tensor, y_tensor};
}