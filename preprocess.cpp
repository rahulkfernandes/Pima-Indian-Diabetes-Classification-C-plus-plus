#include "preprocess.h"

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