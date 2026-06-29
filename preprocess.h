#include <Eigen/Dense>
#include <random>
#include <tuple>
#include <torch/torch.h>

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>
train_test_split(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd labels,
    double test_ratio = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
);

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

std::tuple<torch::Tensor, torch::Tensor> convert_to_tensor(Eigen::MatrixXd X, Eigen::VectorXd y);