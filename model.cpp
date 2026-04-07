#include "model.h"

SimpleMLP::SimpleMLP(int in_size, int hidden_size, int out_size, float dropout_rate)
    : dropout_rate_(dropout_rate) {
    fc1 = register_module("fc1", torch::nn::Linear(in_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size / 2));
    fc3 = register_module("fc3", torch::nn::Linear(hidden_size / 2, out_size));
}

torch::Tensor SimpleMLP::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, dropout_rate_, is_training());
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
}