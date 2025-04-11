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

  void SparseAdamOptions::serialize(serialize::InputArchive& archive) {

  }

  void SparseAdamOptions::serialize(serialize::OutputArchive& archive) const {

  }

  double SparseAdamOptions::get_lr() const {
    return lr_;
  }

  void SparseAdamOptions::set_lr(const double lr) {
    TORCH_CHECK(lr > 0, "Invalid learning rate: ", lr);
    lr_ = lr;
  }

  bool operator==(const SparseAdamOptions& lhs, const SparseAdamOptions& rhs) {
    return lhs.lr() == rhs.lr() &&
           lhs.betas() == rhs.betas() &&
           lhs.eps() == rhs.eps() &&
           lhs.maximize() == rhs.maximize();
  }


// ===== SparseAdamParamState Implementation =====
bool operator==(const SparseAdamParamState& lhs, const SparseAdamParamState& rhs) {
    return lhs.step() == rhs.step() &&
           torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
           torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
           torch::equal(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());
           torch::equal(lhs.row_steps(), rhs.row_steps());
}



// ===== SparseAdam Optimizer Implementation =====
torch::Tensor SparseAdam::step(LossClosure closure) {
  // Disable gradient tracking for the update; enable only if calling closure.
  at::NoGradGuard no_grad;
  Tensor loss = {};

  if (closure != nullptr) {
    // Enable gradients within the closure.
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }

  // Iterate over each parameter group.
  for (auto& group : param_groups_) {
    // Retrieve options for this group. (Each group can override defaults if needed.)
    auto& group_options = static_cast<SparseAdamOptions&>(group.options());
    double lr     = group_options.lr();
    double eps    = group_options.eps();
    bool maximize = group_options.maximize();
    double beta1, beta2;
    std::tie(beta1, beta2) = group_options.betas();

    // For each parameter in that group:
    for (auto& p : group.params()) {
      // If param is not defined or has no gradient, skip.
      if (!p.defined() || !p.grad().defined()) {
        continue;
      }

      // Check that the parameter itself is dense (like Python's check).
      if (p.is_sparse()) {
        TORCH_CHECK(false, "SparseAdam requires parameters to be dense. Parameter is sparse.");
      }
      // Check that the gradient is actually sparse.
      if (!p.grad().is_sparse()) {
        TORCH_CHECK(false, "SparseAdam requires sparse gradients, please consider Adam instead.");
      }

      // Grab the gradient
      auto grad = p.grad();

      // State initialization if missing
      auto param_key = p.unsafeGetTensorImpl();
      if (state_.find(param_key) == state_.end()) {
        auto state = std::make_unique<SparseAdamParamState>();
        // We'll keep the single "global" step if you want, or skip it entirely:
        state->step(0);

        // Make exp_avg, exp_avg_sq:
        state->exp_avg() = torch::zeros_like(p, p.options().memory_format(torch::MemoryFormat::Preserve));
        state->exp_avg_sq() = torch::zeros_like(p, p.options().memory_format(torch::MemoryFormat::Preserve));

        // row_steps is 1D, length p.size(0):
        auto row_steps_opts = torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(p.device());
        state->row_steps() = torch::zeros({p.size(0)}, row_steps_opts);

        state_[param_key] = std::move(state);
      }

      auto& param_state = static_cast<SparseAdamParamState&>(*state_.at(param_key));
      auto& exp_avg = param_state.exp_avg();
      auto& exp_avg_sq = param_state.exp_avg_sq();
      auto& row_steps = param_state.row_steps(); // 1D, length nRows
      auto row_steps_accessor = row_steps.accessor<int64_t, 1>();

      // Update step
      param_state.step(param_state.step() + 1);

      // We only update the indices/values that appear in the sparse gradient
      auto indices = grad._indices();  // shape: [rows, nnz]
      auto values  = grad._values();   // shape: [nnz, ...] or [nnz] if 1D

      // Number of non-zero entries in the gradient
      int64_t nnz = values.size(0);

      // For typical nn.Embedding, the 0th dim is the "row" index.
      // We loop through each row in the gradient:
      for (int64_t i = 0; i < nnz; i++) {
        // The row in the first dimension:
        int64_t row = indices[0][i].item<int64_t>();
        row_steps_accessor[row] += 1;
        int64_t local_step_for_this_row = row_steps_accessor[row];

        // Select that row from param, exp_avg, exp_avg_sq.
        // This works for typical 2D Embedding shapes:
        auto param_slice = p.select(0, row);
        auto exp_avg_slice = exp_avg.select(0, row);
        auto exp_avg_sq_slice = exp_avg_sq.select(0, row);

        // The gradient row
        auto grad_slice = values[i];

        // Update the first moment exp_avg
        exp_avg_slice.mul_(beta1).add_(grad_slice, 1 - beta1);

        // Update the second moment exp_avg_sq
        exp_avg_sq_slice.mul_(beta2).addcmul_(grad_slice, grad_slice, 1 - beta2);

        // Compute the denominator = sqrt(...) + eps
        auto denom = exp_avg_sq_slice.sqrt().add_(eps);

        double bias_correction1 = 1.0 - std::pow(beta1, local_step_for_this_row);
        double bias_correction2 = 1.0 - std::pow(beta2, local_step_for_this_row);
        double step_size = lr * std::sqrt(bias_correction2) / bias_correction1;

        // The actual update
        //  update = exp_avg_slice / denom
        // Then param_slice -= step_size * update   (for normal Adam)
        // or   param_slice += step_size * update   (for 'maximize' mode)
        if (maximize) {
          param_slice.addcdiv_(exp_avg_slice, denom, step_size);
        } else {
          param_slice.addcdiv_(exp_avg_slice, denom, -step_size);
        }
      }
    } // end for each param in group
  }   // end for each group
  return loss;
}


} // namespace optim
} // namespace torch
