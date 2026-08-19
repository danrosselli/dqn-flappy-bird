# Forget-and-Grow (FoG) — Implementação para DQN Flappy Bird

**Objetivo deste documento:**  
Documentar, de forma técnica e completa, a técnica **Forget-and-Grow (FoG)** aplicada ao experimento, servindo como referência de especificação e de implementação das modificações no código-fonte existente (`src/`).

**Arquivos principais envolvidos:**
- `replayBuffer.js` → principal ponto de alteração (ER Decay)
- `dqn.js` → Network Expansion + integração
- `Game.js` → possível ajuste de chamada de treino (mínimo)
- `persistenceManager.js` → salvar/carregar novos estados se necessário

---

## 1. Teoria do Forget-and-Grow (FoG)

### 1.1 Problema que resolve

Em Deep Reinforcement Learning contínuo (especialmente DQN online), dois problemas relacionados aparecem:

1. **Primacy Bias**  
   A rede tende a overfitting nas experiências iniciais (mais fáceis) armazenadas no replay buffer. Experiências antigas continuam sendo amostradas com a mesma probabilidade, mesmo quando a distribuição de estados já mudou significativamente (ex.: pipes mais rápidos, gaps mais difíceis).

2. **Loss of Plasticity**  
   Com o tempo, muitos neurônios se tornam "dormant" (ativação próxima de zero). A rede perde capacidade de aprender novas situações porque a maior parte dos parâmetros está saturada ou inativa.

### 1.2 Inspiração biológica

O paper se inspira em dois processos observados no cérebro humano:

- **Infantile Amnesia**: formação de novos neurônios (neurogênese) no hipocampo interrompe traços de memória antigos, fazendo o cérebro "esquecer" experiências muito iniciais.
- **Crescimento neural**: adição de novos neurônios aumenta a capacidade representacional e restaura plasticidade.

### 1.3 Os dois mecanismos do FoG

| Mecanismo              | Nome no paper          | Função principal                                      | Analogia biológica          |
|------------------------|------------------------|-------------------------------------------------------|-----------------------------|
| 1. Experience Replay Decay | ER Decay              | Reduz gradualmente a probabilidade de amostrar experiências antigas | Infantile Amnesia (esquecer o passado distante) |
| 2. Network Expansion   | Network Expansion     | Adiciona novos parâmetros/neurônios dinamicamente    | Neurogênese (crescer capacidade) |

**FoG = ER Decay + Network Expansion**

O paper original aplica isso principalmente em continuous control (MuJoCo, DMControl, etc.), mas os princípios se transferem muito bem para DQN em Flappy Bird, pois o ambiente é non-stationary de forma contínua (dificuldade sobe com o score).

---

## 2. Análise do Código Atual (o que já existe)

### 2.1 ReplayBuffer atual (`replayBuffer.js`)

Você já possui uma estrutura híbrida excelente:

```js
constructor(recentSize = 10000, reservoirSize = 40000)
```

- `recentBuffer` → circular buffer das experiências mais recentes
- `reservoirBuffer` → reservoir sampling (memória de longo prazo uniforme)
- `buffer` → buffer legado (circular)
- Métodos de amostragem: `sampleRandomBasic`, `sampleHybrid`, `sampleHybridWithRecentSeq`, `sampleHybridWithCrashFocus`

**Ponto-chave:** O `sampleHybrid` e variantes tratam recent e reservoir com proporções fixas.  
**O que falta:** um mecanismo de *decay* que reduza a influência (probabilidade de amostragem) das experiências antigas do reservoir ao longo do tempo.

### 2.2 DQNAgent atual (`dqn.js`)

- Rede fixa: 8 → 64 → 64 → 2
- Target network com hard update a cada `TARGET_UPDATE_FREQ`
- Treino throttled (`TRAIN_THROTTLE`)
- Epsilon decay clássico
- Persistência completa via `PersistenceManager`

**O que falta:** capacidade de expandir a rede dinamicamente (adicionar unidades ou camadas) sem destruir o conhecimento já aprendido.

---

## 3. Especificação Técnica das Modificações

### 3.1 Experience Replay Decay (ER Decay)

#### Ideia central
Cada transição no buffer passa a ter uma **idade** (ou um *decay factor*).  
Quanto mais antiga a transição, menor a probabilidade de ela ser amostrada.

#### Implementação recomendada (simples e eficaz)

**Opção A — Decay por contagem de amostragens (recomendada para começar):**

1. Adicionar em cada transição um campo `timesSampled = 0`.
2. Toda vez que uma transição for amostrada, incrementar `timesSampled`.
3. Na hora de amostrar, usar probabilidade inversamente proporcional a `(1 + timesSampled)` ou uma função de decay exponencial.

**Opção B — Decay temporal / por step global (mais alinhada com o paper):**

1. Manter um contador global `totalSteps` ou `totalTransitionsSeen`.
2. Cada transição guarda o `stepAdded` (quando foi inserida).
3. Probabilidade de amostragem ∝ `decayFactor ^ (currentStep - stepAdded)`  
   onde `decayFactor` é um hiperparâmetro (ex.: 0.99995 ou 0.9999).

**Opção C — Híbrida (recomendada para o seu código):**

- Manter o `recentBuffer` intacto (sempre amostrar com alta prioridade as experiências recentes).
- Aplicar o decay **apenas no `reservoirBuffer`**.
- Assim você preserva o foco em experiências recentes + crash focus, e só “esquece” gradualmente a memória de longo prazo.

#### Pseudocódigo do novo método de amostragem

```js
sampleHybridWithDecay(batchSize, recentRatio = 0.35) {
  const batch = [];
  const numRecent = Math.floor(batchSize * recentRatio);
  const numReservoir = batchSize - numRecent;

  // 1. Amostrar do recentBuffer (sem decay)
  // ... (igual ao sampleHybrid atual)

  // 2. Amostrar do reservoirBuffer COM decay
  if (numReservoir > 0 && this.reservoirBuffer.length > 0) {
    // Calcular pesos de amostragem com decay
    const weights = this.reservoirBuffer.map(t => {
      const age = this.totalSeen - (t.stepAdded || 0);
      return Math.pow(this.decayFactor, age);   // decayFactor < 1
    });

    // Amostragem ponderada (roulette wheel ou similar)
    for (let i = 0; i < numReservoir; i++) {
      const idx = weightedRandomIndex(weights);
      batch.push(this.reservoirBuffer[idx]);
      // Opcional: incrementar timesSampled
    }
  }

  // 3. Shuffle final
  // ...
  return batch;
}
```

#### Hiperparâmetros sugeridos

```js
const ER_DECAY_FACTOR = 0.99992;   // valores típicos: 0.9999 ~ 0.99995
const ER_DECAY_MIN_WEIGHT = 0.05;  // peso mínimo para não zerar completamente
```

#### Integração no `add()`

No método `add()` do ReplayBuffer, ao criar a transição:

```js
const transition = { 
  state, action, reward, nextState, done,
  stepAdded: this.totalSeen,   // ou Date.now() / stepCount global
  timesSampled: 0
};
```

---

### 3.2 Network Expansion

#### Ideia central
Periodicamente (ou quando detectar perda de plasticidade), adicionar capacidade nova à rede:

- Opção mais simples: aumentar o número de unidades nas camadas ocultas.
- Opção mais sofisticada: adicionar uma nova camada residual ou um bloco novo e treinar preferencialmente os novos parâmetros no início.

#### Estratégia recomendada para TF.js + DQN (prática)

1. **Expandir as camadas densas existentes** (mais fácil de implementar e manter compatibilidade com target network e persistência).

2. Frequência de expansão:
   - A cada N gerações (ex.: a cada 50 ou 100 gerações), **ou**
   - Quando o score médio das últimas K gerações parar de melhorar significativamente.

3. Como expandir:

```js
expandNetwork(extraUnits = 16) {
  // 1. Criar nova rede com mais unidades
  const newModel = this.createExpandedModel(extraUnits);
  
  // 2. Copiar pesos das camadas antigas para as posições correspondentes
  //    (as novas unidades ficam com inicialização aleatória / Xavier)
  
  // 3. Compilar novamente
  // 4. Atualizar this.model e this.targetModel
  // 5. (Opcional) Congelar temporariamente as unidades antigas por alguns steps
}
```

#### Cuidados importantes

- A `STATE_SIZE` e o número de ações **não mudam**.
- A target network deve ser expandida da mesma forma e receber os pesos imediatamente (ou via soft update).
- Persistência: o `PersistenceManager` precisa ser capaz de salvar/carregar redes de tamanhos diferentes (já existe um check de `modelInputSize`, mas agora o número de unidades ocultas também pode variar).
- Após expansão, é recomendável um pequeno período de “warm-up” com learning rate um pouco maior apenas nos novos parâmetros (se possível).

#### Alternativa mais simples (recomendação inicial)

Em vez de expandir dinamicamente durante o treino, começar com uma rede um pouco maior (ex.: 96 ou 128 unidades) e aplicar **apenas o ER Decay**.  
Muitos dos benefícios de plasticidade já aparecem só com o decay. Network Expansion pode ser adicionada numa segunda fase.

---

## 4. Plano de Implementação Passo a Passo

### Fase 1 — ER Decay (prioritária)

1. No `ReplayBuffer`:
   - Adicionar `this.decayFactor = 0.99992` (ou tornar configurável).
   - No `add()`: gravar `stepAdded: this.totalSeen` e `timesSampled: 0` em cada transição.
   - Criar método auxiliar `weightedRandomIndex(weights)`.
   - Criar novo método `sampleHybridWithDecay(batchSize, recentRatio = 0.35)`.
   - Manter os métodos antigos intactos (para comparação).

2. No `DQNAgent.train()`:
   - Trocar a chamada de amostragem para `this.replayBuffer.sampleHybridWithDecay(BATCH_SIZE)`.

3. (Opcional) Expor o `decayFactor` como hiperparâmetro exportado.

### Fase 2 — Network Expansion (opcional / segunda etapa)

1. Adicionar método `createExpandedModel(extraUnits)` em `DQNAgent`.
2. Adicionar método `expandNetwork(extraUnits = 16)`.
3. Decidir trigger de expansão (ex.: a cada 80 gerações, ou quando highScore não melhora por X gerações).
4. Atualizar `saveBrain` / `loadBrain` e `PersistenceManager` para lidar com redes de tamanhos diferentes (salvar também a arquitetura atual).
5. Após expansão, resetar parcialmente o epsilon ou aumentar temporariamente o exploration.

### Fase 3 — Integração e Logging

- Adicionar no HUD (Game.js) a informação do decay atual ou do tamanho da rede.
- Logar periodicamente a idade média das amostras do reservoir e a proporção de pesos baixos.

---

## 5. Hiperparâmetros Recomendados (ponto de partida)

```js
// ER Decay
export const ER_DECAY_FACTOR = 0.99992;
export const ER_DECAY_MIN_WEIGHT = 0.05;
export const ER_RECENT_RATIO = 0.35;          // manter próximo do que você já usa

// Network Expansion (se implementar)
export const EXPAND_EVERY_GENERATIONS = 80;
export const EXPAND_UNITS = 16;               // unidades adicionadas por expansão
export const MAX_HIDDEN_UNITS = 192;          // limite superior
```

---

## 6. Critérios de Sucesso

Após implementar o ER Decay, espera-se observar:

- Melhor retenção de performance em scores altos (o agente não “esquece” como jogar em velocidades baixas/médias).
- Recuperação mais rápida após quedas de performance.
- Menor overfitting nas primeiras gerações.
- Score médio e highScore mais estáveis ao longo de muitas gerações.

---

## 7. Ordem de Prioridade de Implementação

1. **ER Decay no reservoir** (maior impacto / menor risco)
2. Logging e monitoramento do decay
3. Network Expansion (só depois que o decay estiver estável)
4. Soft target update + possíveis melhorias de arquitetura (dueling, etc.)

---

## 8. Observações Finais de Implementação

- **Não quebre a compatibilidade** com os métodos de amostragem antigos. Mantenha-os.
- O `sampleHybridWithCrashFocus` e `sampleHybridWithRecentSeq` podem continuar sendo usados; o decay deve ser aplicado preferencialmente na parte do reservoir.
- Respeite o `trainingInProgress` flag e o throttle existentes.
- Teste primeiro com `decayFactor` bem próximo de 1.0 (ex.: 0.99995) para mudanças suaves.
- Documente no código qualquer novo hiperparâmetro.

---

**Referência principal:**  
Kang et al. (2025). *A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control*. arXiv:2507.02712

Este README deve ser suficiente para uma implementação fiel e segura do Forget-and-Grow no código atual do projeto.