#pragma once
#include <torch/torch.h>
#include <torch/data.h>
#include <memory>
#include <string>

template<typename ModuleType>
class Trainer {
public:
    Trainer(std::shared_ptr<ModuleType> model, double lr, int epochs, float wt_decay, int batch_size = 32);

    // Train method - moved to header
    template<typename DataLoaderType>
    void train(DataLoaderType& data_loader) {
        model_->train();
        for (int epoch = 1; epoch <= epochs_; ++epoch) {  // 1-indexed for clarity
            double epoch_loss = 0.0;
            int batch_count = 0;
            double correct = 0.0;
            int total = 0;
            
            
            for (auto& batch : data_loader) {
                auto data = batch.data;
                auto targets = batch.target;
                optimizer_.zero_grad();
                auto output = model_->forward(data);
                auto loss = loss_func_(output, targets);
                loss.backward();
                optimizer_.step();
                epoch_loss += loss.template item<double>();
                
                // Compute training accuracy
                auto preds = torch::sigmoid(output) > 0.5;
                correct += (preds == targets).sum().template item<double>();
                total += targets.size(0);
            
                ++batch_count;
            }
            if (epoch % 10 == 0) {
                double train_acc = correct / total;
                std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / batch_count 
                        << " | Train Acc: " << train_acc * 100 << "%" << std::endl;
                }
        }
    }

    // Evaluate method - moved to header
    template<typename DataLoaderType>
    torch::Tensor predict(DataLoaderType& data_loader) {
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
    

    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    std::shared_ptr<ModuleType> model_;
    torch::optim::Adam optimizer_;
    torch::nn::BCEWithLogitsLoss loss_func_;
    int epochs_;
    int batch_size_;
};
