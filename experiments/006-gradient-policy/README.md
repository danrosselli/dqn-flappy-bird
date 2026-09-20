# REINFORCE (Policy Gradient)

## Objective

Replace the value-based DQN approach with a policy-gradient method that learns a stochastic policy directly. REINFORCE optimizes the policy from complete episode returns, without a replay buffer or a target network.

## Hypothesis

A softmax policy trained on normalized discounted returns can learn a competent flapping policy from the same 8-dimensional state, trading DQN's sample efficiency for a simpler, unbiased update. The main risk is gradient variance, since every update depends on a full episode.

## Approach

- **State (8-dim)**: horizontal distance and gap alignment for the current and next pipe (`dx`, `dy`, `gap`, `dxNext`, `dyNext`, `gapNext`), bird vertical velocity (`velY`) and current pipe speed (`speed`), all normalized to [-1, +1].
- **Architecture**: a single policy network 8→64→64→2 with a softmax output — each unit is the probability of an action.
- **Update rule**: episodic REINFORCE.
  - Discounted returns: `G_t = r_t + γ·G_{t+1}`
  - Returns are z-score normalized (mean/std) for a more stable update.
  - Loss: `-G_t · log π(a_t | s_t)`, averaged over the trajectory.
- **Trajectory**: the current episode is collected in memory and consumed by a single update when the bird dies. The buffer is capped at 10,000 steps, keeping the most recent ones.
- **Rewards**: Gaussian proximity shaping toward the current gap center (sigma 0.5, no offset), velocity penalty (-0.05 when |velY| > 700), flap penalty (-0.1 per flap), pipe passage (+10), collision (-20). No per-frame survival bonus.
- **Exploration**: actions are sampled from the policy's softmax distribution — no epsilon-greedy.

## Changes

- Replaced `DQNAgent` with `PolicyGradientAgent` (softmax policy, episodic update).
- Removed the replay buffer and the target network.
- Added trajectory collection and a single end-of-episode gradient update.
- Persistence metadata is tagged with `algorithm: 'reinforce'`, so a previously saved DQN is discarded instead of being loaded into the policy network.
- HUD now shows action probabilities (P-Idle / P-Flap) instead of Q-values.

## Training

Episodes: TBD
Learning rate: 0.003
Gamma: 0.99
Max trajectory: 10000 steps
Update timing: end of episode
Return normalization: z-score

## Results

TBD

## Conclusion

TBD
