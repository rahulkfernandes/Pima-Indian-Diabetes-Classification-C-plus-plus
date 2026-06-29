#include "train.h"
#include <iostream>

// Constructor
template<typename ModuleType>
Trainer<ModuleType>::Trainer(std::shared_ptr<ModuleType> model, double lr, int epochs, int batch_size)
    : model_(model),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(lr)),
      epochs_(epochs),
      batch_size_(batch_size) {}


// Train method
template<typename ModuleType>
template<typename DataLoaderType>
void Trainer<ModuleType>::train(DataLoaderType& data_loader) {
    model_->train();
    for (int epoch = 1; epoch <= epochs_; ++epoch) {   // 1-indexed for clarity
        double epoch_loss = 0.0;
        int batch_count = 0;
        for (auto& batch : data_loader) {
            auto data = batch.data;
            auto targets = batch.target;
            optimizer_.zero_grad();
            auto output = model_->forward(data);
            auto loss = loss_func_(output, targets);
            loss.backward();
            optimizer_.step();
            epoch_loss += loss.template item<double>();
            ++batch_count;
        }
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / batch_count << std::endl;
        }
    }
}

// Evaluate method
template<typename ModuleType>
template<typename DataLoaderType>
torch::Tensor Trainer<ModuleType>::predict(DataLoaderType& data_loader) {
    model_->eval();
    std::vector<torch::Tensor> batch_predictions;
    
    for (auto& batch : data_loader) {
        auto data = batch.data;
        auto output = model_->forward(data);
        auto preds = torch::sigmoid(output) > 0.5;
        batch_predictions.push_back(preds);
    }
    
    // Concatenate all batch predictions into a single tensor
    return torch::cat(batch_predictions, /*dim=*/0);
}

// Save model
template<typename ModuleType>
void Trainer<ModuleType>::save_model(const std::string& path) {
    torch::save(model_, path);
    std::cout << "Model saved to " << path << std::endl;
}

// Load model
template<typename ModuleType>
void Trainer<ModuleType>::load_model(const std::string& path) {
    torch::load(model_, path);
    std::cout << "Model loaded from " << path << std::endl;
}

#include "model.h"
template class Trainer<SimpleMLP>;