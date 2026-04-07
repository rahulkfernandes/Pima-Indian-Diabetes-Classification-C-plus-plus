#include <Eigen/Dense>

class Scaler {
public:
    Scaler() = default;

    void fit(const Eigen::MatrixXd& X);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& X);

    const Eigen::RowVectorXd& get_mean() const { return mean_; }
    const Eigen::RowVectorXd& get_scale() const { return scale_; }
    bool is_fitted() const { return fitted_; }

private:
    Eigen::RowVectorXd mean_;   // 1 x n_features
    Eigen::RowVectorXd scale_;  // 1 x n_features
    bool fitted_ = false;
    static constexpr double eps = 1e-12;
};