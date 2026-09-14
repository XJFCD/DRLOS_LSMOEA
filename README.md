# DRLOS-LSMOEA

**A Deep Reinforcement Learning-Assisted Operator Selection for Large-Scale Multi-Objective Optimization**

DRLOS-LSMOEA is the MATLAB implementation of a DRL-assisted operator selection framework for
large-scale multi-objective optimization problems (LSMOPs). It is implemented as an algorithm
plug-in for the [PlatEMO](https://github.com/BIMK/PlatEMO) platform and consists of three main
components:

1. **A heterogeneous operator pool** of eight reproduction operators, covering both
   large-scale-specific and general-purpose search behaviours.
2. **A lightweight DRL agent** that is trained online to predict the reward of each operator
   for the current search state, and selects the operator greedily.
3. **A dimension-control mechanism (DCM)** that, in later evolutionary stages, applies the
   selected operator to only a subset of variable groups to control the effective update
   dimensionality.

## Requirements

- MATLAB (any recent release; R2020a or later recommended).
- [PlatEMO](https://github.com/BIMK/PlatEMO) (v3.x / v4.x).
- Deep Learning Toolbox (for `mapminmax`).
- Parallel Computing Toolbox (optional; used for GPU acceleration via `canUseGPU` / `gpuArray`).

## Installation

1. Install and launch [PlatEMO](https://github.com/BIMK/PlatEMO) once so that the framework is set up.
2. Copy this folder into the PlatEMO `Algorithms` directory, keeping the subfolders:

   ```
   PlatEMO/Algorithms/DRLOS-LSMOEA/
   ├── DRLOS_LSMOEA.m
   ├── EnvironmentalSelection.m
   ├── Normalization.m
   ├── CalFitness.m
   ├── DRL/
   └── Operators/
   ```

3. Run `platemo` (or start the PlatEMO app). The algorithm is listed as **DRLOS-LSMOEA** under
   multi-/many-objective, real-valued, large-scale algorithms.

## Usage

From the PlatEMO GUI, select `DRLOS-LSMOEA` as the algorithm and any scalable benchmark as the
problem. From the command line:

```matlab
% 2-objective LSMOP1, D = 500, N = 300, maxFE = 1e5
platemo('algorithm', @DRLOS_LSMOEA, 'problem', @LSMOP1, 'M', 2, 'D', 500, 'N', 300, 'maxFE', 1e5);

% 2-objective UF1, D = 1000
platemo('algorithm', @DRLOS_LSMOEA, 'problem', @UF1,  'M', 2, 'D', 1000, 'N', 300, 'maxFE', 1e5);
```

## Repository layout

| Path | Description |
| --- | --- |
| `DRLOS_LSMOEA.m` | Main algorithm class (`classdef DRLOS_LSMOEA < ALGORITHM`). Contains the DRL loop, the state/reward computation, the operator dispatch, and the variable-grouping helper. |
| `EnvironmentalSelection.m` | Environmental selection used by the algorithm (local copy). |
| `Normalization.m` | Objective normalization helper. |
| `CalFitness.m` | Fitness (convergence/diversity) metric used by the competitive-learning operator. |
| `DRL/` | Lightweight fully-connected value network implemented from scratch in MATLAB (no Deep Learning Toolbox network objects). |
| `DRL/iniA.m` | Weight/bias initialization. |
| `DRL/MgaussRandom.m` | Gaussian random number generation for initialization. |
| `DRL/dropout.m` | Dropout helper. |
| `DRL/trainmodel.m` | Builds and trains the network from the experience buffer. |
| `DRL/trainNet.m` | One training step (forward pass, backpropagation, SGD update with weight decay). |
| `DRL/testNet.m` | Forward pass used for inference. |
| `DRL/updatemodel.m` | Incremental re-training of an existing network. |
| `DRL/Estimate.m` | Network/estimation helper. |
| `Operators/GA/` | GA operators (`OperatorGAhalf`, `OperatorGAhalfonly`). |
| `Operators/SBX/` | Simulated binary crossover (`Operator_SBX`, `Operator_SBXonly`). |
| `Operators/PM/` | Polynomial mutation (`Operator_PM`, `Operator_PMonly`). |
| `Operators/DE/` | Differential evolution (`OperatorDE`, `OperatorDEonly`). |
| `Operators/MOPSO/` | Multi-objective particle swarm operator and its support files. |
| `Operators/LMOEADS/` | LMOEA/D-Style decomposition-based operator. |
| `Operators/CompetitiveLearning/` | Competitive-learning operator (LMOCSO-style). |
| `Operators/TwoStageSample/` | Two-stage directed sampling operator for large-scale problems. |

The eight operators above correspond to actions `1..8` of the DRL agent in `DRLOS_LSMOEA.m`.

## Method overview

- **State.** A compact two-dimensional descriptor of the current population:
  `average_f = log(mean(sum(objs, 2)) + eps)` (convergence) and
  `average_d = sum(prctile(objs, 90) - prctile(objs, 10))` (diversity/spread).
- **Reward.** The normalized improvement of the two descriptors, combined as
  `reward = 0.7 * tanh(Δf / σf) + 0.3 * tanh(Δd / σd)`, where `σf`, `σd` are the standard
  deviations of the recent deltas (z-score normalization over a sliding history).
- **Agent.** A 3-layer MLP (input → 40 → 40 → reward), ReLU + tanh activations, dropout,
  trained with SGD and weight decay. The action is the concatenated `[state, operator_index]`
  and the network predicts a scalar reward; the operator with the largest predicted reward is
  selected (ε-greedy with `greedy = 0.95`, discount factor `gama = 0.9`).
- **Schedule.** Operators are sampled uniformly during the first `flag1 = 20%` of the function
  evaluations; the network is built after that from the experience buffer and periodically
  re-trained.
- **Two stages.** For the first `flag2 = 30%` of evaluations the selected operator acts on the
  full decision vector. Afterwards, the DCM applies the operator to a random subset of variable
  groups (`G = 6` groups; `Ksel = round(alpha * G)` groups are varied, `alpha = 0.3`); the
  remaining dimensions are copied from the parents.

## Default parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `Ns1`, `Ns2` | 30, 15 | Sample counts of the two-stage sampling operator. |
| `Nw`, `Ns` | 10, 30 | LMOEA/D-style operator neighbourhood / sampling parameters. |
| `Rate` | 0.8 | Learning rate factor of the competitive-learning operator. |
| `div` | 10 | MOPSO diversity parameter. |
| `proC`, `disC` | 1, 20 | SBX crossover probability / distribution index. |
| `proM`, `disM` | 1, 20 | Polynomial mutation probability / distribution index. |
| `greedy` | 0.95 | Greedy (exploitation) probability of the agent. |
| `gama` | 0.9 | Discount factor. |
| `capacity` | 200 | Size of the experience replay buffer. |
| `alpha` | 0.3 | Fraction of variable groups updated by the DCM. |
| `G` | 6 | Number of variable groups. |

The first set of parameters is exposed through `Algorithm.ParameterSet` and can be overridden
from the command line by passing extra arguments after `@DRLOS_LSMOEA`.

## Reproducing the experiments

The experimental setting reported in the paper is:

- Test suites: LSMOP and UF.
- Number of objectives: `M ∈ {2, 3}`.
- Number of decision variables: `D ∈ {500, 1000, 2000, 5000, 8000}`.
- Population size `N = 300`; maximum function evaluations `maxFE = 100000`.
- 20 independent runs per instance; performance measured by IGD and HV.

Example (one instance, one run):

```matlab
platemo('algorithm', @DRLOS_LSMOEA, 'problem', @LSMOP1, 'M', 2, 'D', 2000, ...
        'N', 300, 'maxFE', 1e5, 'metric', {@IGD, @HV});
```

## Notes

- GPU acceleration is used automatically when `canUseGPU` reports an available GPU; otherwise
  the network is trained on the CPU. `canUseGPU` and `gpuArray` are provided by the Parallel
  Computing Toolbox, so that toolbox must be available for the script to run as-is.
- `EnvironmentalSelection.m`, `Normalization.m`, and `CalFitness.m` are local copies of utility
  functions. They are included so the algorithm folder is self-contained; you may remove them
  and rely on the PlatEMO versions if preferred.

## Acknowledgements

This work builds on the [PlatEMO](https://github.com/BIMK/PlatEMO) platform (BIMK Group) and
reuses or adapts several of its operator implementations. If you use this code, please
acknowledge PlatEMO and cite:

```bibtex
@article{Tian2017PlatEMO,
  title   = {PlatEMO: A MATLAB platform for evolutionary multi-objective optimization [educational forum]},
  author  = {Tian, Ye and Cheng, Ran and Zhang, Xingyi and Jin, Yaochu},
  journal = {IEEE Computational Intelligence Magazine},
  volume  = {12},
  number  = {4},
  pages   = {73--87},
  year    = {2017},
  publisher = {IEEE}
}
```

## Citation

If you use DRLOS-LSMOEA in your research, please cite:

```bibtex
@article{drlos-lsmoa,
  title  = {A Deep Reinforcement Learning-Assisted Operator Selection for Large-Scale Multi-Objective Optimization},
  author = {TODO: add authors},
  year   = {2027},
  note   = {Under review}
}
```

## License

The code in this repository is released for academic and research use. The algorithm plug-in is
distributed together with (and derives from) PlatEMO, whose original copyright and license terms
apply to the framework and to the reused/adapted operator implementations. Please observe the
PlatEMO license when redistributing or modifying this code.
