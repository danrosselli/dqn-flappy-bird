# Insights e Estratégias para Mitigar Catastrophic Forgetting em DQN (Flappy Bird)

**Documento gerado para o projeto de Deep Q-Network com TensorFlow.js**  
**Data:** Agosto 2026  
**Contexto:** Agente DQN online contínuo para Flappy Bird com estado contínuo `[dx, dy, velY, gap, dxNext, dyNext, gapNext, speed]`, replay buffer híbrido (reservoir + recent) e target network.

---

## 1. O Problema: Catastrophic Forgetting

O **esquecimento catastrófico** (catastrophic forgetting) ocorre quando a rede neural sobrescreve pesos importantes para situações antigas enquanto aprende situações novas.  

No Flappy Bird isso se manifesta porque:
- A distribuição de estados muda continuamente (velocidade dos pipes aumenta com o score, gaps variam).
- O agente encontra situações cada vez mais difíceis ao longo das gerações.
- Experiências antigas (fáceis) e novas (difíceis) competem pelos mesmos parâmetros da rede.

Seu código já possui várias boas práticas:
- Target network com update periódico
- Experience Replay híbrido (reservoir + recent)
- Throttling de treino
- Persistência completa (modelo + buffers + metadata)
- Recompensas densas de proximidade + bônus de score + penalidade de morte

Ainda assim, o aprendizado contínuo sofre com interferência entre experiências antigas e novas.

---

## 2. Estratégias Avaliadas e Descartadas (por enquanto)

### 2.1 LSTM / Deep Recurrent Q-Network (DRQN)
- **O que faz:** Substitui a primeira camada fully-connected por uma LSTM, permitindo que a rede tenha memória temporal.
- **Quando ajuda:** Principalmente em POMDPs (observações parciais), onde o agente precisa integrar informação ao longo do tempo.
- **No Flappy Bird:** O estado já é relativamente Markoviano (você inclui dx, dy, velY, gaps atuais e próximos + speed). LSTM pode suavizar a política, mas **não resolve o core do catastrophic forgetting**.
- **Custo:** Treino mais lento, mais memória e necessidade de gerenciar o estado oculto da LSTM.
- **Decisão:** Deixar de lado por enquanto.

### 2.2 GRAPES (Group Responsibility for Adjusting the Propagation of Error Signals)
- Paper: *Learning in Deep Neural Networks Using a Biologically Inspired Optimizer* (Nature Communications, 2022) — [arXiv:2104.11604](https://arxiv.org/abs/2104.11604)
- **O que faz:** Otimizador biologicamente inspirado que modula o sinal de erro com base na *importância do nó* (derivada da distribuição de pesos na camada).
- **Benefícios reportados:** Melhora convergência, acurácia e **mitiga catastrophic forgetting** em benchmarks de sequential tasks.
- **No seu caso:** Interessante, mas exige modificar o fluxo de backpropagation/otimização no TensorFlow.js (não é plug-and-play).
- **Decisão:** Deixar de lado por enquanto para focar em soluções mais práticas.

---

## 3. Abordagens Avançadas de Continual RL Avaliadas

### 3.1 Forget-and-Grow (FoG) — **Mais recomendada**
**Paper:** *A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control* (2025) — [arXiv:2507.02712](https://arxiv.org/abs/2507.02712)

Dois mecanismos complementares:

1. **Experience Replay Decay (ER Decay)**  
   Reduz gradualmente a probabilidade de amostrar transições muito antigas no buffer.  
   Combate o *primacy bias* (a rede fica viciada nas experiências iniciais fáceis e perde plasticidade).

2. **Network Expansion**  
   Adiciona novos parâmetros/neurônios dinamicamente ao longo do treinamento, restaurando plasticidade.

**Por que é excelente para o Flappy Bird:**
- Ambiente contínuo e non-stationary (dificuldade sobe suavemente com o score).
- Combina perfeitamente com o replay buffer híbrido que você já possui.
- ER Decay é relativamente simples de implementar.
- Network Expansion também é factível em TF.js.

**Dificuldade de implementação:** Média-baixa (começar só pelo ER Decay).

---

### 3.2 Meta-Experience Replay (MER) — **Muito boa e já validada em Flappy Bird**
**Paper:** *Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference* (ICLR 2019) — [arXiv:1810.11910](https://arxiv.org/abs/1810.11910)

- Combina Experience Replay com meta-learning estilo **Reptile**.
- O update é feito de forma a maximizar transferência e minimizar interferência entre o exemplo atual e os do buffer.
- O paper original **testou explicitamente em variantes de Flappy Bird** (gaps que diminuem) e mostrou superioridade sobre experience replay normal, especialmente com buffer limitado e ambiente non-stationary.

**Por que é boa:**
- Diretamente aplicável a DQN.
- Você já tem um bom buffer → MER só muda *como* o batch é usado no treino (micro-updates internos + meta-update).

**Dificuldade:** Média (exige modificar o loop de `train()`).

---

### 3.3 Gradient Boosted DQN (GB-DQN)
**Paper:** *GB-DQN: Gradient Boosted DQN Models for Non-stationary Reinforcement Learning* (2025) — [arXiv:2512.17034](https://arxiv.org/abs/2512.17034)

- Em vez de sobrescrever a mesma rede, congela a rede atual e treina uma nova rede residual que aprende a corrigir o erro de Bellman (residual) da ensemble anterior.
- A Q-function final é a soma das redes.

**Vantagens:**
- Preserva conhecimento antigo de forma explícita.
- Se adapta bem a mudanças de regime (ex.: quando a velocidade dos pipes sobe bastante).
- Recuperação mais rápida após drifts e menos catastrophic forgetting.

**Dificuldade:** Média-alta (gerenciar ensemble de redes + decidir quando adicionar novo “weak learner”).

---

### 3.4 PackNet / Progressive Networks — **Menos adequadas**
- **Progressive Networks:** Cria uma coluna nova inteira para cada “task” e congela as anteriores + conexões laterais.
- **PackNet:** Faz pruning iterativo, congela pesos importantes da task atual e libera o resto para a próxima.

**Problema no seu caso:**
- Foram projetadas para **sequências de tasks discretas e bem separadas**.
- No Flappy Bird a dificuldade muda de forma contínua e suave → não existem “task boundaries” claras.
- Progressive Networks crescem a memória linearmente.
- PackNet precisa de um critério de task boundary que você não possui naturalmente.

**Conclusão:** Úteis em lifelong learning multi-task, mas overkill e pouco naturais aqui. Deixar por último (ou ignorar por enquanto).

---

## 4. Ranking Prático de Prioridade

| Posição | Método                          | Adequação ao Flappy Bird | Dificuldade | Retorno esperado |
|---------|---------------------------------|---------------------------|-------------|------------------|
| 1       | Forget-and-Grow (ER Decay)      | Excelente                 | Baixa       | Alto             |
| 2       | Meta-Experience Replay (MER)    | Excelente                 | Média       | Alto             |
| 3       | Gradient Boosted DQN            | Boa                       | Média-Alta  | Médio-Alto       |
| 4       | PackNet / Progressive Networks  | Baixa                     | Alta        | Baixo            |

---

## 5. Outras Estratégias Complementares Recomendadas

Além das abordagens avançadas acima, estas técnicas clássicas/modernas também são valiosas:

1. **Melhorar ainda mais o Experience Replay**
   - Prioritized Experience Replay (PER) baseado em TD-error.
   - Mais foco em trajetórias de crash recentes (você já tem `sampleHybridWithCrashFocus`).
   - Combinar com ER Decay.

2. **Regularização de estabilidade-plasticidade**
   - Elastic Weight Consolidation (EWC) ou variantes.
   - Synaptic Intelligence.
   - L2 regularization / weight decay leve.

3. **Manter plasticidade da rede**
   - CReLU (Concatenated ReLU) para evitar dormant neurons.
   - ReDo / Continual Backprop / targeted resets de neurônios inativos.
   - Soft resets periódicos de partes da rede.

4. **Melhorias de arquitetura e hiperparâmetros**
   - Soft target update (Polyak averaging) em vez de hard copy.
   - Dueling DQN.
   - Double DQN (se ainda não estiver usando).
   - Learning rate schedule.
   - Rede um pouco maior ou com residual connections.

---

## 6. Caminho de Experimentação Sugerido

1. **Implementar ER Decay** no `ReplayBuffer` (maior retorno / esforço).
2. **Testar Meta-Experience Replay (MER)** no loop de treino.
3. Se ainda precisar de mais plasticidade → adicionar Network Expansion ou experimentar residual boosting (GB-DQN).
4. Só depois considerar LSTM, GRAPES ou PackNet.

---

## 7. Referências Principais

- **FoG:** Kang et al. (2025). *A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control*. arXiv:2507.02712
- **MER:** Riemer et al. (2019). *Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference*. ICLR 2019. arXiv:1810.11910
- **GB-DQN:** Lee & Lee (2025). *GB-DQN: Gradient Boosted DQN Models for Non-stationary Reinforcement Learning*. arXiv:2512.17034
- **GRAPES:** Dellaferrera et al. (2022). *Learning in Deep Neural Networks Using a Biologically Inspired Optimizer*. Nature Communications. arXiv:2104.11604
- **DRQN:** Hausknecht & Stone (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs*. arXiv:1507.06527
- **Loss of Plasticity:** Abbas et al. (2023). *Loss of Plasticity in Continual Deep Reinforcement Learning*. arXiv:2303.07507

---

## 8. Observações Finais

O maior ganho costuma vir de **melhor balanceamento no replay + preservação de plasticidade**, não necessariamente de trocar a arquitetura inteira por LSTM ou GRAPES.

Seu setup atual (replay híbrido + target network + recompensas densas) já é sólido. As melhorias prioritárias (ER Decay e MER) devem trazer ganhos perceptíveis com esforço moderado.

Este documento serve como referência interna do projeto. Pode ser atualizado conforme novos experimentos forem realizados.