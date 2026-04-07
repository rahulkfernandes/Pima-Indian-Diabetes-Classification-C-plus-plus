#pragma once
#include <torch/torch.h>

class SimpleMLP : public torch::nn::Module {
public:
    SimpleMLP(int in_size, int hidden_size, int out_size, float dropout_rate);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    float dropout_rate_;
};