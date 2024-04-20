// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_PLANNERS_MPPI_OPTIMIZER_H_
#define MJPC_PLANNERS_MPPI_OPTIMIZER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/MPPI/policy.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// sampling planner limits
inline constexpr int MinMPPISamplingSplinePoints = 1;
inline constexpr int MaxMPPISamplingSplinePoints = 36;
inline constexpr int MinMPPISamplingSplinePower = 1;
inline constexpr int MaxMPPISamplingSplinePower = 5;
inline constexpr double MinMPPINoiseStdDev = 0.0;
inline constexpr double MaxMPPINoiseStdDev = 1.0;
inline constexpr double MinMPPIinvTemp = 0.0;
inline constexpr double MaxMPPIinvTemp = 40.0;
inline constexpr double MinMPPIGamma = 0.0;
inline constexpr double MaxMPPIGamma = 1.0;
inline constexpr double MinMPPIAlphamu = 0.0;
inline constexpr double MaxMPPIAlphamu = 1.0;
inline constexpr double MinMPPIAlphasig = 0.0;
inline constexpr double MaxMPPIAlphasig = 1.0;
// inline constexpr double MinminMPPIetha = 0.0001;
// inline constexpr double MinmaxMPPIetha = 1;
// inline constexpr double MaxminMPPIetha = 0.1;
// inline constexpr double MaxmaxMPPIetha = 3.0;

class MPPIPlanner : public RankedPlanner {
 public:
  // constructor
  MPPIPlanner() = default;

  // destructor
  ~MPPIPlanner() override = default;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy using random sampling
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // compute trajectory using MPPI policy
  void MPPI_NominalTrajectory(int horizon, ThreadPool& pool, int ith);

  // set action from policy
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override;

  // resample nominal policy
  void UpdateNominalPolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(int i, int horizon);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // optimizes policies, but rather than picking the best, generate up to
  // ncandidates. returns number of candidates created.
  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;
  // returns the total return for the nth candidate (or another score to
  // minimize)
  double CandidateScore(int candidate) const override;

  // set action from candidate policy
  void ActionFromCandidatePolicy(double* action, int candidate,
                                 const double* state, double time) override;

  void CopyCandidateToPolicy(int candidate) override;

  void Cal_MPPI_candidate(double lambda_candidate, double min_return, int num_trajectory, int ith_lambda);

  // Added by me !! 
  void setCurrentCost(double cost) override{
    current_cost = cost;
  }

  int Optimize_Lambda(int horizon);
  int best_lambda;
  double current_cost;

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  MPPIPolicy policy;  // (Guarded by mtx_)
  MPPIPolicy candidate_policy[kMaxTrajectory];
  MPPIPolicy MPPI_candidate_policy[5];
  MPPIPolicy previous_policy;

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> cov_parameters_scratch;
  std::vector<double> times_scratch;
  std::vector<double> tmp_params_scratch;
  std::vector<double> tmp_times_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];
  Trajectory MPPI_trajectory[kMaxTrajectory];

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // rollout parameters
  double timestep_power;

  // ----- noise ----- //
  double noise_exploration;  // standard deviation for sampling normal: N(0,
                             // exploration)
  std::vector<double> noise;

  // gradient
  std::vector<double> noise_gradient;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  int num_trajectory_;
  int num_spline_points_;

  mutable std::shared_mutex mtx_;

  // inverse temperature
  double lambda;

  // discount factor
  double gamma;

  // importance sample weight
  std::vector<double> importance_weight;

  // etha for the denominator of importance sample weight
  double etha;

  //alpha_mu # from STORM paper.
  double alpha_mu;

  //alpha_sig # from STORM paper.
  double alpha_sig;

  // previous min_trajectory cost
  double previous_min_return;

  //if cost have increased compared to previous_min_return, it have true
  bool cost_trigger = false; 

  // Parameter for storing average cost of n/8 of previous trajectory cost
  double previous_avg_return;

  // Parameter for current n/8 of previous trajectory cost
  double current_avg_return;

  // Parameter to keep past 8 action cost
  
  double past_cost[32] = {0.0,};
  double moving_avg;
  double prev_moving_avg;
  double moving_variance;
  double lambda_objective;
  double prev_lambda_objective;
  double prev_32_avg;
  double prev_32_var;
  int cnt = 0;

  int lambda_choice; // if you increased Lambda previously, +1, if you stay still, 0 , if you decreased Lambda previously, -1
  int lambda_num = 5;
  double lambda_list[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
  // double Minetha;
  // double Maxetha;
  double lam_coeff;
  double lam_power;
  double cov_weight;

  
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_MPPI_OPTIMIZER_H_
