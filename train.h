#pragma once
#include <torch/torch.h>
#include <torch/data.h>
#include <memory>
#include <string>

template<typename ModuleType>
class Trainer {
public:
    Trainer(std::shared_ptr<ModuleType> model, double lr, int epochs, int batch_size = 32);

    template<typename DataLoaderType>
    void train(DataLoaderType& data_loader);
    template<typename DataLoaderType>
    torch::Tensor predict(DataLoaderType& data_loader);

    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    std::shared_ptr<ModuleType> model_;
    torch::optim::Adam optimizer_;
    torch::nn::BCEWithLogitsLoss loss_func_;
    int epochs_;
    int batch_size_;
};
