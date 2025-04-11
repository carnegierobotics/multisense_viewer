#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

/// Options for the SparseAdam optimizer.
struct TORCH_API SparseAdamOptions : public OptimizerCloneableOptions<SparseAdamOptions> {
  /// Constructs a SparseAdamOptions with the given learning rate.
  SparseAdamOptions(double lr = 1e-3);

  TORCH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(bool, maximize) = false;

public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(const SparseAdamOptions& lhs, const SparseAdamOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

  struct TORCH_API SparseAdamParamState
    : public OptimizerCloneableParamState<SparseAdamParamState> {
    TORCH_ARG(int64_t, step) = 0;
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);
    TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};
    TORCH_ARG(torch::Tensor, row_steps);

  public:
    //void serialize(torch::serialize::InputArchive& archive) override;
    //void serialize(torch::serialize::OutputArchive& archive) const override;
    TORCH_API friend bool operator==(
        const SparseAdamParamState& lhs,
        const SparseAdamParamState& rhs);
  };


/// Implements the SparseAdam algorithm.
/// SparseAdam only updates a subset of parameters corresponding to non-zero indices
/// in the sparse gradient (typically produced by a module like nn::Embedding with sparse=true).
class TORCH_API SparseAdam : public Optimizer {
public:
  explicit SparseAdam(
      std::vector<OptimizerParamGroup> param_groups,
      SparseAdamOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<SparseAdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() > 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() > 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ", std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ", std::get<1>(betas));
  }
  explicit SparseAdam(std::vector<Tensor> params, SparseAdamOptions defaults = {})
      : SparseAdam({OptimizerParamGroup(std::move(params))}, defaults) {}


  /// Performs a single optimization step.
  torch::Tensor step(LossClosure closure = nullptr) override;
  //void save(serialize::OutputArchive& archive) const override;
  //void load(serialize::InputArchive& archive) override;

private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SparseAdam);
  }
};

} // namespace optim
} // namespace torch
