# dqn-flappy-bird
Deep Q-Learning Flappy Bird Game using Phaser Engine

## Arquitetura do Agent

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. COLETA DE EXPERIÊNCIA (Game.js)                             │
│     A cada frame: estado → ação → recompensa → próximo estado   │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. ARMAZENAMENTO (StratifiedPER)                               │
│     Guarda transições em 3 estratos (fácil/médio/difícil)         │
│     Prioriza amostragem balanceada (40%/40%/20%)                │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. AMOSTRAGEM (DQNAgent.train)                                 │
│     Pega BATCH_SIZE (64) transições do buffer                   │
│     Separa em tensores: states, actions, rewards, nextStates    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. CÁLCULO DO TARGET (DDQN - Double DQN)                       │
│     a) Q(s', a*) usando model (online)                          │
│     b) a* = argmax Q(s')                                        │
│     c) Q(s', a*) usando targetModel (estável)                   │
│     d) target = r + γ * Q_target(s', a*) * (1 - done)          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. NESTED TRAINING (NestedModel.trainStep) ← O CORAÇÃO         │
│     Atualiza cada camada em frequências diferentes               │
└─────────────────────────────────────────────────────────────────┘
```
