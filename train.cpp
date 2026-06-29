#include "train.h"
#include <iostream>

// Constructor
template<typename ModuleType>
Trainer<ModuleType>::Trainer(std::shared_ptr<ModuleType> model, double lr, int epochs, float wt_decay, int batch_size)
    : model_(model),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(lr).weight_decay(wt_decay)),
      epochs_(epochs),
      batch_size_(batch_size) {}


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