#include "SparseAdam.h"

#include <torch/torch.h>
#include <stdexcept>
#include <sstream>

namespace torch {
    namespace optim {
        // ===== SparseAdamOptions Implementation =====

        SparseAdamOptions::SparseAdamOptions(double lr) {
            // Use the generated setter to assign the learning rate.
            this->lr_ = lr;
        }

        void SparseAdamOptions::serialize(serialize::InputArchive &archive) {
        }

        void SparseAdamOptions::serialize(serialize::OutputArchive &archive) const {
        }

        double SparseAdamOptions::get_lr() const {
            return lr_;
        }

        void SparseAdamOptions::set_lr(const double lr) {
            TORCH_CHECK(lr > 0, "Invalid learning rate: ", lr);
            lr_ = lr;
        }

        bool operator==(const SparseAdamOptions &lhs, const SparseAdamOptions &rhs) {
            return lhs.lr() == rhs.lr() &&
                   lhs.betas() == rhs.betas() &&
                   lhs.eps() == rhs.eps() &&
                   lhs.maximize() == rhs.maximize();
        }


        // ===== SparseAdamParamState Implementation =====
        bool operator==(const SparseAdamParamState &lhs, const SparseAdamParamState &rhs) {
            return lhs.step() == rhs.step() &&
                   torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
                   torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
                   torch::equal(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());
            torch::equal(lhs.row_steps(), rhs.row_steps());
        }


        // ===== SparseAdam Optimizer Implementation =====
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <iomanip>  // for setting precision
#include <unordered_map>

        // Ensure you have this namespace for indexing.
        using namespace torch::indexing;

        torch::Tensor SparseAdam::step(LossClosure closure) {
            torch::NoGradGuard no_grad;
            torch::Tensor loss = {};

            if (closure) {
                torch::AutoGradMode enable_grad(true);
                loss = closure();
            }

            // Set high precision for debug output.
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "[DEBUG] Optimizer step called with " << param_groups_.size() << " param groups." << std::endl;

            // For each parameter group
            for (auto &group: param_groups_) {
                auto &group_options = static_cast<SparseAdamOptions &>(group.options());
                const double lr = group_options.lr();
                const double eps = group_options.eps();
                const bool maximize = group_options.maximize();
                double beta1, beta2;
                std::tie(beta1, beta2) = group_options.betas();

                std::cout << "[DEBUG] Processing param group with lr: " << lr
                        << ", eps: " << eps
                        << ", beta1: " << beta1
                        << ", beta2: " << beta2
                        << ", maximize: " << maximize << std::endl;

                for (auto &p: group.params()) {
                    if (!p.defined() || !p.grad().defined()) {
                        std::cout << "[DEBUG] Skipping parameter because it or its grad is not defined." << std::endl;
                        continue; // skip
                    }

                    TORCH_CHECK(!p.is_sparse(), "SparseAdam: parameter is sparse. Only dense params supported.");
                    TORCH_CHECK(p.grad().is_sparse(), "SparseAdam: gradient is not sparse (use Adam instead).");

                    // Get or create state
                    auto param_key = p.unsafeGetTensorImpl();
                    if (state_.find(param_key) == state_.end()) {
                        std::cout << "[DEBUG] Creating optimizer state for new parameter." << std::endl;
                        auto state = std::make_unique<SparseAdamParamState>();
                        state->step() = 0;
                        state->exp_avg() = torch::zeros_like(
                            p, p.options().memory_format(torch::MemoryFormat::Preserve));
                        state->exp_avg_sq() = torch::zeros_like(
                            p, p.options().memory_format(torch::MemoryFormat::Preserve));
                        state_[param_key] = std::move(state);
                    } else {
                        std::cout << "[DEBUG] Using existing optimizer state for parameter." << std::endl;
                    }

                    auto &param_state = static_cast<SparseAdamParamState &>(*state_.at(param_key));
                    auto &exp_avg = param_state.exp_avg();
                    auto &exp_avg_sq = param_state.exp_avg_sq();

                    // Determine the sign for the gradient update.
                    const double sign = maximize ? 1.0 : -1.0;
                    //const double sign = maximize ? -1.0 : 1.0;
                    std::cout << "[DEBUG] Maximize: 0 Minimize: 1, res:" << maximize << std::endl;

                    // Coalesce the gradient so indices are unique.
                    auto grad_sparse = p.grad().coalesce();
                    auto grad_indices = grad_sparse._indices(); // shape: [dims, nnz]
                    auto grad_values = grad_sparse._values(); // shape: [nnz, ...]
                    if (grad_values.numel() == 0) {
                        std::cout << "[DEBUG] Skipping parameter update; gradient tensor is empty." << std::endl;
                        continue;
                    }
                    // Increment the step counter and print it.
                    param_state.step(param_state.step() + 1);
                    const int64_t step = param_state.step();
                    std::cout << "[DEBUG] Parameter step count: " << step << std::endl;

                    // Bias corrections and step_size calculation.
                    const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
                    const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
                    const double step_size = (lr * std::sqrt(bias_correction2)) / bias_correction1;
                    std::cout << "[DEBUG] Bias correction 1: " << bias_correction1
                            << ", Bias correction 2: " << bias_correction2
                            << ", Step size: " << step_size << std::endl;

                    const int64_t nnz = grad_values.size(0);
                    std::cout << "[DEBUG] Number of nonzero gradient entries: " << nnz << std::endl;

                    // Update each element individually.
                    for (int64_t i = 0; i < nnz; i++) {
                        // Build full index vector from the sparse gradient indices.
                        std::vector<torch::indexing::TensorIndex> indices;
                        for (int64_t d = 0; d < grad_indices.size(0); d++) {
                            int64_t idx = grad_indices[d][i].item<int64_t>();
                            indices.push_back(idx);
                        }
                        // Debug print the indices being updated.
                        std::cout << "[DEBUG] Updating element at indices: ";
                        for (auto idx: indices) {
                            // If possible, print as int. (Note: if using TensorIndex wrapper, you may need to convert.)
                            std::cout << idx << " ";
                        }
                        std::cout << std::endl;

                        // Retrieve the parameter element and the corresponding state elements.
                        auto p_elem = p.index(indices);
                        auto exp_avg_elem = exp_avg.index(indices);
                        auto exp_avg_sq_elem = exp_avg_sq.index(indices);

                        // Print the current value.
                        std::cout << "[DEBUG] Parameter value before update: " << p_elem << std::endl;

                        // Get the gradient for this element.
                        auto g = grad_values[i].mul(sign);
                        std::cout << "[DEBUG] Gradient value: " << g << std::endl;

                        // Update the first moment: exp_avg = beta1 * exp_avg + (1 - beta1) * g
                        exp_avg_elem.mul_(beta1);
                        exp_avg_elem.add_(g, 1 - beta1);

                        // Update the second moment: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g^2
                        exp_avg_sq_elem.mul_(beta2);
                        exp_avg_sq_elem.addcmul_(g, g, 1 - beta2);

                        // Compute the denominator: sqrt(exp_avg_sq) + eps
                        auto denom = exp_avg_sq_elem.sqrt().add_(eps);

                        // Update the parameter element: p_elem -= step_size * (exp_avg_elem / denom)
                        p_elem.addcdiv_(exp_avg_elem, denom, -step_size);

                        // Debug print the updated parameter value.
                        std::cout << "[DEBUG] Parameter value after update: " << p_elem << std::endl;
                    }
                } // end for each param
            } // end for each param group

            return loss;
        }
    } // namespace optim
} // namespace torch
