# Explicação Completa do DQN no Flappy Bird

Este documento reúne as explicações sobre a Equação de Bellman, Experience Replay, Target Network, Q-values, tensores e o fluxo completo de treinamento do agente DQN.

---

## 1. Visão Geral

A equação de Bellman (na forma usada no Q-Learning / DQN) aparece **somente no agente DQN**.

### Localização exata

Dentro do método `train()` da classe `DQNAgent`:

```javascript
const nextQValues = this.targetModel.predict(nextStateTensor);
const maxNextQ = nextQValues.max(1);
const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));
```

### O que essa linha representa

Essa é a implementação da equação de Bellman para o target do DQN:

$$
Q_{\text{target}}(s, a) = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})
$$

- `rewardTensor` → $r$
- `maxNextQ` → $\max_{a'} Q(s', a')$ (calculado com a target network)
- `gamma` → fator de desconto ($0.99$)
- `notDone` → multiplica por $0$ quando o episódio termina (estado terminal)

Depois o código usa esse `targetQ` para calcular o TD-error e fazer o `model.fit`.

### No arquivo do jogo (Phaser)

Não existe a equação de Bellman. Esse arquivo só:

- observa o estado,
- calcula a recompensa,
- chama `agent.replayBuffer.add(...)` e `agent.train()`,
- escolhe a ação.

A matemática de Bellman fica concentrada **exclusivamente** no método `train()` do agente DQN.

---

## 2. O lado do jogo: método `update()`

O método `update()` do jogo (Phaser) é responsável por:

1. Observar o estado atual
2. Calcular a recompensa
3. Armazenar a transição no buffer
4. Treinar o agente
5. Escolher e executar a próxima ação

### Fluxo resumido do frame

```
1. Observa o estado atual → currentState
2. Se já existia um estado e ação anteriores:
   - Calcula a recompensa da transição anterior
   - Guarda a transição no buffer: (lastState, lastAction, reward, currentState, done)
   - Chama train()
3. Escolhe a nova ação com base no currentState
4. Executa a ação (flap ou não)
5. Salva currentState e a ação escolhida como lastState e lastAction para o próximo frame
```

### Trecho de armazenamento da transição

```javascript
// 3. Armazenar Transição
if (this.lastState !== null && this.lastAction !== null) {
  const reward = survivalReward + this.proximityReward + velPenalty + this.bonusReward;

  this.agent.replayBuffer.add(
    this.lastState,   // s  (estado anterior)
    this.lastAction,  // a  (ação tomada)
    reward,           // r  (recompensa recebida)
    currentState,     // s' (novo estado)
    this.gameOver     // done (se o episódio terminou)
  );
  await this.agent.train();
}
this.bonusReward = 0;
```

---

## 3. Experience Replay Buffer

O **Replay Buffer** (Experience Replay) é o coração do aprendizado no DQN.

### O que é

No DQN clássico a gente **não** treina a rede neural com a transição que acabou de acontecer. Em vez disso:

1. Guardamos a transição `(s, a, r, s', done)` em um buffer (uma fila/memória).
2. Durante o treinamento, amostramos **aleatoriamente** um mini-batch desse buffer.
3. Treinamos a rede com esse mini-batch.

Isso se chama **Experience Replay**.

### Por que usar buffer?

| Motivo                      | Explicação                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| Quebra a correlação temporal| Experiências consecutivas são muito parecidas. Treinar em sequência deixa a rede instável. |
| Reutiliza experiências      | Uma boa (ou ruim) experiência pode ser usada várias vezes.                 |
| Estabiliza o treinamento    | Uma das principais razões do DQN funcionar bem.                            |

### O que cada parâmetro significa

| Parâmetro        | Significado                                     | Exemplo no Flappy           |
|------------------|-------------------------------------------------|-----------------------------|
| `this.lastState` | Estado antes de tomar a ação                    | `[dx, dy, velY, gap, ...]`  |
| `this.lastAction`| Ação que o agente escolheu (0 = IDLE, 1 = FLAP) | `0` ou `1`                  |
| `reward`         | Recompensa que recebeu por essa ação            | survival + proximity + etc. |
| `currentState`   | Estado depois da ação                           | novo `[dx, dy, ...]`        |
| `this.gameOver`  | Se o episódio terminou (true = morte)           | `true` / `false`            |

### Observação importante

No código o `train()` é chamado **todo frame** (depois de adicionar uma experiência). Isso é comum em implementações online, mas o ideal é:

- Adicionar a experiência sempre
- Treinar só de vez em quando (ex: a cada N frames) **ou**
- Treinar só quando o buffer tiver um tamanho mínimo de experiências

Se o buffer ainda estiver muito pequeno no começo, o treinamento fica ruim.

**Resumo simples:**  
O buffer é a **memória** do agente. Ele guarda as experiências passadas (estado, ação, recompensa, próximo estado) e depois treina a rede neural pegando amostras aleatórias dessa memória, em vez de treinar só com o que acabou de acontecer.

---

## 4. Como os dados do buffer são usados no treinamento

### Visão geral do fluxo

```
Buffer (memória) → Sample (amostra aleatória) → Calcula Target Q → Compara com Q atual → Treina a rede
```

### O método `train()`

```javascript
async train() {
  // 1. Só treina de vez em quando (throttle)
  if (this.stepCount % TRAIN_THROTTLE !== 0) return;

  // 2. Só treina se tiver experiências suficientes
  if (this.replayBuffer.size() < BATCH_SIZE) return;

  // 3. Amostra um mini-batch aleatório do buffer
  const batch = this.replayBuffer.sampleRandomBasic(BATCH_SIZE);
  // ...
}
```

### 1. Amostragem do Buffer

```javascript
const batch = this.replayBuffer.sampleRandomBasic(BATCH_SIZE); // BATCH_SIZE = 64
```

Isso pega **64 transições aleatórias** do tipo:

```javascript
{
  state:      [dx, dy, velY, gap, dxNext, ...],  // s
  action:     0 ou 1,                           // a
  reward:     número,                           // r
  nextState:  [...],                            // s'
  done:       true/false                        // se morreu
}
```

Depois o código separa tudo em arrays:

```javascript
states, actions, rewards, nextStates, dones
```

### 2. Cálculo do Target (o “valor correto”)

```javascript
const nextQValues = this.targetModel.predict(nextStateTensor);  // Q(s', a') usando a Target Network
const maxNextQ = nextQValues.max(1);                            // max_a' Q(s', a')

const targetQ = reward + gamma * maxNextQ * (1 - done);
```

Em matemática clássica do DQN:

$$
y = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \quad \text{(se não terminou)}
$$

$$
y = r \quad \text{(se terminou / done = true)}
$$

- `gamma = 0.99` → valoriza recompensas futuras
- Usa a **Target Network** (`this.targetModel`) para estabilizar o treinamento
- Se `done = true` (pássaro morreu), o target fica só a recompensa (não tem futuro)

### 3. Comparação com o Q atual (TD-Error)

```javascript
const qValues = this.model.predict(stateTensor);  // Q atual da rede principal

// Para cada transição do batch:
const currentQ = qValuesArray[i][actions[i]];     // Q(s, a) que a rede previu
const target   = targetQArray[i];                 // valor "correto" calculado acima

const absError = Math.abs(target - currentQ);     // TD-Error
```

O **TD-Error** mede o quanto a rede está errada naquela transição.

Você ainda normaliza esse erro:

```javascript
const normError = absError / (Math.abs(target) + 1e-6);
```

E só treina se o erro médio normalizado for grande o suficiente:

```javascript
if (meanNormTDError >= TD_NORM_THRESHOLD) {  // 0.04
  // Treina
} else {
  // Pula o treino (economia de tempo)
}
```

### 4. Atualização da rede (o treino de fato)

```javascript
// Substitui apenas o Q da ação que foi tomada pelo target
qValuesArray[i][actions[i]] = target;

// Treina a rede para aproximar esse novo valor
await this.model.fit(stateTensor, targets, {
  epochs: 1,
  batchSize: BATCH_SIZE,
  verbose: 0
});
```

**Resumo do que a rede está aprendendo:**

> “Quando eu estava no estado $s$ e tomei a ação $a$, o valor correto de $Q$ deveria ser $y$ (o target).”

A loss usada é **Mean Squared Error** entre o Q previsto e o target.

### 5. Atualização da Target Network

```javascript
if (this.stepCount % TARGET_UPDATE_FREQ === 0) {  // a cada 1000 steps
  this.targetModel.setWeights(this.model.getWeights());
}
```

A Target Network é uma cópia “congelada” da rede principal. Ela só é atualizada de tempos em tempos para evitar que o target fique se movendo o tempo todo (isso estabiliza muito o treinamento).

### Resumo visual do processo

```
1. Buffer tem milhares de transições antigas
          ↓
2. Pega 64 aleatórias
          ↓
3. Para cada uma:
   - Calcula Q(s', ·) com a Target Network
   - Calcula target = r + γ * max Q(s')
   - Compara com Q(s, a) da rede principal
          ↓
4. Se o erro for grande o suficiente → treina a rede principal
          ↓
5. De vez em quando copia os pesos da rede principal → Target Network
```

### Observações sobre o código

| Ponto               | Comentário                                              |
|---------------------|---------------------------------------------------------|
| `TRAIN_THROTTLE = 2`| Só tenta treinar a cada 2 frames                        |
| `TD_NORM_THRESHOLD` | Só treina se o erro for relevante (evita treinar à toa) |
| Target Network      | Atualizada a cada 1000 steps                            |
| Gamma 0.99          | Boa escolha                                             |
| Batch size 64       | Padrão razoável                                         |

---

## 5. As duas redes: `model` e `targetModel`

Existem **duas redes neurais** com a mesma arquitetura:

| Rede              | Função                          | Atualizada quando?             |
|-------------------|---------------------------------|--------------------------------|
| `this.model`      | Rede principal (online network) | A cada treinamento (`model.fit`) |
| `this.targetModel`| Rede alvo (target network)      | A cada 1000 steps              |

### Como funciona a sincronização

```javascript
if (this.stepCount % TARGET_UPDATE_FREQ === 0) {  // TARGET_UPDATE_FREQ = 1000
  this.targetModel.setWeights(this.model.getWeights());
}
```

A cada 1000 passos, os pesos da `model` são copiados para a `targetModel`.

### Por que fazer isso?

Se usássemos a **mesma rede** para calcular o Q atual e o target, o valor alvo ficaria se movendo o tempo todo (o que a rede está tentando aprender muda enquanto ela aprende). Isso deixa o treinamento muito instável.

A Target Network resolve isso:

- A `model` aprende e muda frequentemente.
- A `targetModel` fica “congelada” por 1000 steps, fornecendo targets mais estáveis.
- Depois de 1000 steps, ela é atualizada com os novos pesos da `model`.

Esse é um dos truques clássicos do DQN original (DeepMind, 2015) para estabilizar o aprendizado.

---

## 6. Q-values e a função `chooseAction`

### O que a rede neural devolve

A rede tem **2 neurônios** na saída (`units: 2` e `activation: 'linear'`):

```javascript
model.add(tf.layers.dense({
  units: 2,
  activation: 'linear'
}));
```

Ela devolve um vetor com **2 Q-values**:

```javascript
[Q(state, IDLE),  Q(state, FLAP)]
//     índice 0         índice 1
```

Esses valores representam o quanto a rede acredita que é bom tomar cada ação naquele estado.

### Como a ação é escolhida (`chooseAction`)

```javascript
async chooseAction(state) {
  // 1. Exploração (epsilon-greedy)
  if (Math.random() < epsilon) {
    return Math.random() < 0.05 ? ACTION_FLAP : ACTION_IDLE;
  }

  // 2. Exploitation (escolhe a melhor ação segundo a rede)
  return tf.tidy(() => {
    const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
    const qValues = this.model.predict(stateTensor);   // → [Q_idle, Q_flap]
    const action = qValues.argMax(1).dataSync()[0];     // pega o índice do maior valor
    return action;  // 0 ou 1
  });
}
```

| Situação                  | O que acontece                                    | Resultado |
|---------------------------|---------------------------------------------------|-----------|
| `Math.random() < epsilon` | Escolhe aleatoriamente (com viés forte para IDLE) | 0 ou 1    |
| Caso contrário            | Olha os 2 Q-values e pega o maior (`argMax`)      | 0 ou 1    |

- Se `Q_idle > Q_flap` → devolve `0` (IDLE)
- Se `Q_flap > Q_idle` → devolve `1` (FLAP)

**Em outras palavras:**  
A rede neural **não** decide diretamente “bater a asa ou não”.  
Ela estima dois números (os Q-values) e a função `chooseAction` simplesmente escolhe a ação que tem o maior valor estimado (com um pouco de aleatoriedade no começo por causa do epsilon).

### `argMax` e `dataSync`

Ambos são métodos do TensorFlow.js.

**`argMax`:**

```javascript
qValues.argMax(1)
```

- Método de tensor.
- Retorna o **índice** do maior valor ao longo de um eixo.
- `qValues` tem shape `[1, 2]` → `.argMax(1)` procura o maior no eixo das ações → resultado: tensor com índice `0` ou `1`.

**`dataSync`:**

```javascript
.dataSync()[0]
```

- Copia os dados do tensor (que está na GPU/WebGL) para a memória da CPU de forma **síncrona**.
- Retorna um `TypedArray`.
- O `[0]` pega o primeiro (e único) valor.

```javascript
const action = qValues.argMax(1).dataSync()[0];
```

1. `argMax(1)` → cria um tensor com o índice da melhor ação  
2. `.dataSync()` → traz esse valor para o JavaScript normal  
3. `[0]` → pega o número (`0` ou `1`)

---

## 7. Como o treinamento realmente funciona (detalhe fino)

### Passo a passo no treino

1. Pega um batch do buffer (64 experiências antigas).
2. Calcula os Q-values com as **duas** redes:
   - `this.model.predict(states)` → Q-values atuais da rede principal → `[Q(s, IDLE), Q(s, FLAP)]`
   - `this.targetModel.predict(nextStates)` → Q-values da target network → usa só o maior: $\max Q(s', a')$
3. Monta o target:

```javascript
target = reward + gamma * maxQ_target(s')   // se não morreu
target = reward                             // se morreu
```

4. Substitui **apenas** o Q da ação que foi realmente tomada:

```javascript
// Exemplo: se o pássaro fez FLAP (action = 1)
qValuesArray[i][1] = target;   // só mexe no índice da ação tomada
// o Q do IDLE continua igual
```

5. Treina a rede principal (`model`) para que ela se aproxime desse novo vetor de Q-values:

```javascript
model.fit(states, qValuesArray_modificado)
```

| O que a rede vê        | O que ela deve aprender                                      |
|------------------------|--------------------------------------------------------------|
| Estado $s$             | Os dois Q-values                                             |
| Ação que o pássaro fez | Só o Q dessa ação é corrigido para o target                  |
| Target vem de          | $r + \gamma \cdot \max(Q$ da targetModel no próximo estado) |

A `targetModel` **não** é treinada diretamente. Ela só serve para calcular um target mais estável. Quem realmente aprende (atualiza os pesos) é só a `model`.

### Os Q-values são calculados diretamente com base nas recompensas?

**Não exatamente.**

Os Q-values **não** são calculados diretamente a partir das recompensas. Eles são **estimativas** que a rede neural aprende ao longo do tempo.

1. A rede neural **prevê** os Q-values:

```javascript
qValues = model.predict(state)  // → [Q(s, IDLE), Q(s, FLAP)]
```

No começo esses valores são quase aleatórios.

2. Durante o treino, a gente calcula um **target** (valor desejado) usando a equação de Bellman:

```
target = recompensa + γ * max(Q da targetModel no próximo estado)
```

A recompensa entra sim, mas **só como uma parte** do cálculo. A outra parte vem da própria estimativa futura da rede (via targetModel).

3. A rede é treinada para que o Q-value da ação tomada se aproxime desse target.

| Conceito          | O que é de fato                                                         |
|-------------------|-------------------------------------------------------------------------|
| Q-value           | Estimativa da rede do “valor” de tomar uma ação                         |
| Target            | Valor calculado com $r + \gamma \cdot \max Q_{\text{futuro}}$           |
| Treinamento       | Faz o Q previsto se aproximar do target                                 |
| Ao longo do tempo | Os Q-values vão se tornando boas estimativas do retorno futuro esperado |

A recompensa **influencia** os Q-values (através do target), mas os Q-values em si são **saídas da rede neural**, não um cálculo direto tipo “$Q$ = soma das recompensas”.

---

## 8. Tensores vs Arrays

O TensorFlow.js **não** trabalha diretamente com arrays ou números normais do JavaScript. Ele trabalha com **tensors**.

### Exemplos de conversão no código

Na hora de escolher a ação:

```javascript
const stateTensor = tf.tensor2d([state], [1, STATE_SIZE]);
const qValues = this.model.predict(stateTensor);
```

Na hora de treinar:

```javascript
const stateTensor     = tf.tensor2d(states, [BATCH_SIZE, STATE_SIZE]);
const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, STATE_SIZE]);
const rewardTensor    = tf.tensor1d(rewards);
const doneTensor      = tf.tensor1d(dones);
```

| Função             | O que faz                     | Exemplo de entrada       |
|--------------------|-------------------------------|--------------------------|
| `tf.tensor2d(...)` | Converte array 2D → tensor 2D | lista de estados         |
| `tf.tensor1d(...)` | Converte array 1D → tensor 1D | lista de rewards / dones |
| `tf.tensor(...)`   | Versão mais genérica          | qualquer formato         |

Quando você precisa do valor de volta para o JavaScript normal:

```javascript
.dataSync()     // traz os dados do tensor para um TypedArray
.arraySync()    // traz como array JavaScript normal
```

**Resumo:**

- Tudo que entra na rede (`predict`, `fit`) precisa ser tensor.
- Tudo que sai da rede também é tensor.
- Você converte JS → tensor na entrada e tensor → JS na saída.

### Diferença entre Array e Tensor

| Aspecto                  | Array JS               | Tensor                                    |
|--------------------------|------------------------|-------------------------------------------|
| Onde fica a memória      | CPU                    | CPU ou GPU (WebGL)                        |
| Velocidade               | Lento para muita conta | Muito mais rápido (especialmente com GPU) |
| Operações                | Manuais / loops        | Otimizadas (`matMul`, `add`, `relu`...)   |
| Diferenciação automática | Não                    | Sim (autograd)                            |
| Formato                  | Flexível               | Tipado e com shape fixo                   |
| Gerenciamento            | Automático (GC)        | Manual (precisa de `.dispose()`)          |

**Por que o TensorFlow usa tensor?**

- **Performance**: operações em lote (batch) são executadas de forma muito mais eficiente, especialmente com o backend WebGL (GPU do navegador).
- **Operações matemáticas avançadas**: multiplicação de matrizes, broadcast, redução, etc.
- **Compatibilidade com o modelo**: as camadas da rede neural esperam tensors.
- **Possibilidade de rodar na GPU**.

### O `stateTensor` no batch

```javascript
batch.forEach(transition => {
  states.push(transition.state);
  actions.push(transition.action);
  rewards.push(transition.reward);
  nextStates.push(transition.nextState);
  dones.push(transition.done ? 1 : 0);
});

const stateTensor = tf.tensor2d(states, [BATCH_SIZE, STATE_SIZE]);
```

- `batch` é um array com 64 transições.
- O `forEach` separa cada campo em arrays separados.
- `states` fica assim:

```javascript
states = [
  [dx1, dy1, velY1, ...],  // estado 1
  [dx2, dy2, velY2, ...],  // estado 2
  // ... total: 64 arrays
]
```

- Depois vira um **único tensor 2D** de shape `[64, 8]`:
  - 64 → quantidade de experiências no batch
  - 8 → tamanho de cada estado (`STATE_SIZE`)

A rede neural consegue processar os 64 estados **de uma vez só** (em paralelo), o que é bem mais eficiente.

---

## 9. Explicação detalhada do cálculo do Target (Bellman)

```javascript
const nextQValues = this.targetModel.predict(nextStateTensor);
const maxNextQ = nextQValues.max(1);
const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));
```

### 1. Previsão dos Q-values do próximo estado

```javascript
const nextQValues = this.targetModel.predict(nextStateTensor);
```

- `nextStateTensor` tem shape `[64, 8]` (os 64 próximos estados do batch)
- A target network prevê os Q-values de cada um deles
- Resultado: tensor de shape `[64, 2]`  
  (para cada estado futuro: `[Q(s', IDLE), Q(s', FLAP)]`)

### 2. Pega o maior Q-value de cada linha

```javascript
const maxNextQ = nextQValues.max(1);
```

- `.max(1)` pega o maior valor ao longo do eixo 1 (o eixo das ações)
- Resultado: tensor de shape `[64]`  
  → um único valor por experiência: $\max_{a'} Q_{\text{target}}(s', a')$

Isso representa a **melhor ação possível** no próximo estado (segundo a target network).

### 3. Cria uma máscara para saber se o episódio terminou

```javascript
const notDone = tf.logicalNot(tf.cast(doneTensor, 'bool'));
```

- `doneTensor` tem `0` ou `1` (1 = pássaro morreu)
- Converte para booleano e inverte:
  - Se `done = 1` (morreu) → `notDone = false` (0)
  - Se `done = 0` (ainda vivo) → `notDone = true` (1)

Isso serve para **zerar o valor futuro** quando o jogo já acabou.

### 4. Calcula o target final (Equação de Bellman)

```javascript
const targetQ = rewardTensor.add(maxNextQ.mul(gamma).mul(notDone));
```

Implementa a fórmula clássica do Q-Learning:

$$
y = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})
$$

- `rewardTensor` → recompensa imediata $r$
- `maxNextQ.mul(gamma)` → valor futuro descontado $\gamma \cdot \max Q(s')$
- `.mul(notDone)` → zera o futuro se o pássaro morreu
- `.add(...)` → soma tudo

**Resultado final:**  
`targetQ` é um tensor de shape `[64]` contendo, para cada experiência do batch, o valor que a rede deveria ter previsto para a ação que foi tomada.

---

## 10. Por que usa o máximo dos Qs de $s'$ (e não a ação real)?

Porque este código está implementando **Q-Learning (off-policy)**, e a equação de Bellman ótima usa o **máximo** sobre as ações futuras.

A transição armazenada é:

```
(state, action, reward, nextState, done)
```

Na equação de Bellman do Q-Learning:

$$
Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a')
$$

O $\max_{a'}$ significa:

> “Qual o melhor valor que eu posso obter a partir de $s'$, assumindo que daqui pra frente eu vou agir de forma ótima?”

Por isso:

```javascript
const nextQValues = this.targetModel.predict(nextStateTensor);  // Q de todas as ações em s'
const maxNextQ = nextQValues.max(1);                            // pega o maior
```

Ele **não** usa a ação que o agente realmente tomou em `nextState`, porque:

1. Queremos estimar a função Q ótima ($Q^*$), não a política que estava sendo usada na coleta.
2. O agente pode ter tomado uma ação ruim em $s'$ (por causa do $\varepsilon$-greedy). Usar essa ação contaminaria o target.
3. O $\max$ permite aprendizado **off-policy**: aprender a política ótima enquanto exploramos com outra política.

### Se usasse a ação real → seria SARSA (on-policy)

$$
Q(s, a) \leftarrow r + \gamma \cdot Q(s', a')
$$

| Algoritmo        | Target                             | Tipo de aprendizado | Usa a ação real de $s'$? |
|------------------|------------------------------------|---------------------|--------------------------|
| Q-Learning / DQN | $r + \gamma \max_{a'} Q(s', a')$   | Off-policy          | Não                      |
| SARSA            | $r + \gamma Q(s', a')$             | On-policy           | Sim                      |

No código (DQN) ele corretamente usa o $\max$.

---

## 11. O treinamento acontece no estado $s$, usando o max Q de $s'$

Sim.

### Fluxo resumido

1. Pega o max Q do estado seguinte $s'$ (usando a target network)
2. Monta o target:

$$
\text{target} = r + \gamma \cdot \max_{a'} Q(s', a')
$$

3. Treina a rede principal no estado atual $s$:

$$
Q(s, a) \approx \text{target}
$$

(onde $a$ é a ação que realmente foi tomada em $s$)

```
Transição:   s  --(a)-->  r, s'
                   ↓
              target = r + γ · max Q(s')
                   ↓
         Atualiza Q(s, a)  ←  target
```

O valor de $s'$ só serve para calcular o alvo.  
O aprendizado (o fit) acontece em $s$.

---

## 12. Como o `fit` funciona com a rede de 2 saídas

A rede termina com 2 valores (Q-Idle e Q-Flap).

O truque do DQN é como montam o target:

```javascript
// Prepara o target para o fit
qValuesArray[i][actions[i]] = target;
```

Depois:

```javascript
const targetsTensor = tf.tensor2d(qValuesArray, [BATCH_SIZE, 2]);

await this.model.fit(stateTensor, targets, {
  epochs: 1,
  batchSize: BATCH_SIZE,
  verbose: 0
});
```

### O que acontece

1. A rede prevê 2 valores: `[Q_Idle, Q_Flap]`
2. Copiam esses 2 valores para o target.
3. **Só substituem** a posição da ação que realmente foi tomada pelo valor de Bellman.

Exemplo:

| Ação tomada | Q-valores atuais da rede | Target que vai pro fit |
|-------------|--------------------------|------------------------|
| FLAP (1)    | [3.2, 5.1]               | [3.2, 8.7]             |
| IDLE (0)    | [4.0, 2.8]               | [7.3, 2.8]             |

4. O loss (`meanSquaredError`) é calculado entre a predição e esse target de 2 posições.

### Por que funciona?

- Posição da ação **não** tomada → target = predição atual → erro ≈ 0
- Posição da ação **tomada** → target = valor de Bellman → erro = TD-error
- O backpropagation só “puxa” o Q da ação correta.

---

## 13. Treinamento por batch e o threshold de TD-error

A decisão é no nível do **batch inteiro**, não amostra por amostra.

1. Amostra 64 transições aleatórias.
2. Calcula o TD-error normalizado de cada uma.
3. Tira a média (`meanNormTDError`).
4. Se a média ≥ `TD_NORM_THRESHOLD` (0.04) → treina o **batch inteiro**.
5. Se a média for menor → pula o **batch inteiro**.

Mesmo que só algumas amostras tenham TD-error alto, se a média passar do threshold, **todas as 64** são usadas no fit.

Isso é diferente do **Prioritized Experience Replay (PER)** clássico, onde cada transição tem prioridade individual e a amostragem é proporcional a essa prioridade. Aqui a amostragem continua uniforme e a filtragem é só uma decisão binária “treina ou não treina este batch”.

É uma heurística simples e barata de “só treina quando o batch parece ter informação útil em média”.

---

## 14. Resumo final do fluxo completo

```
FRAME DO JOGO
─────────────
1. Observa estado atual (currentState)
2. Se existe lastState/lastAction:
   - Calcula reward
   - Adiciona (lastState, lastAction, reward, currentState, done) no Buffer
   - Chama agent.train()
3. Escolhe ação (epsilon-greedy ou argMax dos Q-values)
4. Executa ação (flap ou idle)
5. Salva currentState e action como lastState/lastAction

TREINO (agent.train)
────────────────────
1. Throttle + checa tamanho do buffer
2. Amostra 64 transições aleatórias do Buffer
3. Converte para tensors
4. Calcula target com a equação de Bellman (usando targetModel)
5. Calcula TD-error normalizado
6. Se erro médio >= threshold:
   - Monta targets (só corrige o Q da ação tomada)
   - model.fit(...)
   - Decai epsilon
7. De tempos em tempos: copia pesos model → targetModel
```
