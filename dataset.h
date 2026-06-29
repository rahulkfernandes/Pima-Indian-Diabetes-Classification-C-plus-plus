// dataset.h
#pragma once
#include <torch/torch.h>

class DiabetesDataset : public torch::data::Dataset<DiabetesDataset> {
public:
    DiabetesDataset(torch::Tensor features, torch::Tensor labels)
        : features_(features), labels_(labels) {}

    // Return a single sample (Example containing feature tensor and label tensor)
    torch::data::Example<> get(size_t index) override {
        return {features_[index], labels_[index]};
    }

    // Return the number of samples
    torch::optional<size_t> size() const override {
        return features_.size(0);
    }

private:
    torch::Tensor features_;
    torch::Tensor labels_;
};