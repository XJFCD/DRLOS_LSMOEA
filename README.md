# DRLOS-LSMOEA

**A Deep Reinforcement Learning-Assisted Operator Selection for Large-Scale Multi-Objective Optimization**

DRLOS-LSMOEA is the MATLAB implementation of a DRL-assisted operator selection framework for
large-scale multi-objective optimization problems (LSMOPs). It is implemented based on the
[PlatEMO](https://github.com/BIMK/PlatEMO) platform and consists of three main components:

1. **A heterogeneous operator pool** of eight reproduction operators, covering both
   large-scale-specific and general-purpose search behaviours.
2. **A lightweight DRL agent** that is trained online to predict the reward of each operator
   for the current search state, and selects the operator greedily.
3. **A dimension-control mechanism (DCM)** that, in later evolutionary stages, applies the
   selected operator to only a subset of variable groups to control the effective update
   dimensionality.

## Requirements

- MATLAB R2023b.
- [PlatEMO](https://github.com/BIMK/PlatEMO) v4.12.
- Deep Learning Toolbox (for `mapminmax`).
- Parallel Computing Toolbox (optional; used for GPU acceleration via `canUseGPU` / `gpuArray`).

## Usage

From the PlatEMO GUI, select `DRLOS-LSMOEA` as the algorithm and any scalable benchmark as the
problem. From the command line:

```matlab
% 2-objective LSMOP1, D = 500, N = 300, maxFE = 1e5
platemo('algorithm', @DRLOS_LSMOEA, 'problem', @LSMOP1, 'M', 2, 'D', 500, 'N', 300, 'maxFE', 100000);

% 2-objective UF1, D = 1000
platemo('algorithm', @DRLOS_LSMOEA, 'problem', @UF1,  'M', 2, 'D', 1000, 'N', 300, 'maxFE', 100000);
```

## Repository layout

| Path | Description |
| --- | --- |
| `DRLOS_LSMOEA.m` | Main algorithm class (`classdef DRLOS_LSMOEA < ALGORITHM`). Contains the DRL loop, the state/reward computation, the operator dispatch, and the variable-grouping helper. |
| `EnvironmentalSelection.m` | Environmental selection used by the algorithm. |
| `Normalization.m` | Objective normalization helper. |
| `CalFitness.m` | Fitness (convergence/diversity) metric used by the competitive-learning operator. |
| `DRL/` | The DRL implementation is based on the code of Ming et al. (2024). |
| `Operators/GA/` | GA operators. |
| `Operators/SBX/` | Simulated binary crossover. |
| `Operators/PM/` | Polynomial mutation. |
| `Operators/DE/` | Differential evolution. |
| `Operators/MOPSO/` | Multi-objective particle swarm operator and its support files. |
| `Operators/LMOEADS/` | LMOEA/D-Style decomposition-based operator. |
| `Operators/CompetitiveLearning/` | Competitive-learning operator. |
| `Operators/TwoStageSample/` | Two-stage directed sampling operator for large-scale problems. |

The eight operators above correspond to actions `1...8` of the DRL agent in `DRLOS_LSMOEA.m`.
