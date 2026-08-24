# Planejamento de Engenharia: Robô Autônomo com TinyDQN e Câmera de IA

Este documento reúne todas as especificações técnicas, esquemas de hardware, pinagem e algoritmos necessários para a construção de um robô móvel autônomo com Aprendizado por Reforço Contínuo (DQN) embarcado.

---

## 1. Arquitetura do Hardware e Especificações

O sistema divide o processamento de forma estratégica para mitigar as limitações físicas de cada microcontrolador:

### Cérebro de IA: ESP32-S3 (Módulo WROOM N16R8)
*   **Processador:** Xtensa LX7 dual-core de 240 MHz com instruções vetoriais (SIMD) para aceleração de Redes Neurais.
*   **Memória:** 16 MB de Flash e **8 MB de PSRAM (Obrigatório)**.
*   **Função:** Captura de imagens via OpenMV, decodificação tridimensional de AprilTags, armazenamento do Replay Buffer de 50.000 posições e execução do treinamento contínuo da DQN em C++.

### Atuador e Coleta de Sensores: micro:bit V2 + Base Nezha V2
*   **Função:** Controle físico dos motores com encoder (odometria de distância), leitura de sensores de colisão/ultrassônicos periféricos e cálculo físico da Recompensa.

---

## 2. Visão Computacional: Configurações de AprilTags

Para que a detecção de distância seja estável e não sobrecarregue o processador do ESP32-S3:

1.  **Dimensões do Dado:** Imprimir cubos com **3,5 cm a 4,0 cm de aresta** usando o sistema AMS da sua impressora 3D.
2.  **Família da Tag:** Utilizar a família **`TAG16H5`**. Ela possui menos bits internos (matriz 4x4), gerando blocos impressos maiores e facilitando a leitura em resoluções mais baixas a distâncias maiores.
3.  **Configurações de Impressão no Bambu Studio:**
    *   **Filamentos:** Utilizar estritamente filamentos **Matte (Fosco)** tanto para o Branco quanto para o Preto para mitigar reflexos de luz que cegam a câmera.
    *   **Resolução:** Camadas de `0.08mm` ou `0.12mm` para garantir cantos perfeitamente afiados na geometria da tag.
    *   **Quiet Zone (Borda Branca):** Garantir uma moldura branca contínua de 2mm a 3mm nas bordas de cada face do dado.

---

## 3. Conexão Física e Pinagem (Cabo RJ11 para as Portas Azuis da Nezha V2)

Para conectar o ESP32-S3 genérico diretamente em uma das portas **Azuis (IIC/Serial)** da Nezha V2, monte ou decape um cabo RJ11 (padrão 4P4C) seguindo a sequência oficial abaixo:

| Pino RJ11 | Cor Padrão | Função na Nezha V2 | Conexão no ESP32-S3 | Observação |
| :---: | :--- | :--- | :--- | :--- |
| **1** | 🟢 Verde | SCL / RX da micro:bit | **TX** (ex: GPIO 43) | Envia ordens de ação do ESP32 para o robô |
| **2** | 🔴 Vermelho | VCC (3.3V) | **3V3** | Alimenta o ESP32 diretamente pela base |
| **3** | ⚫ Preto | GND (Terra) | **GND** | Unifica o referencial elétrico (Obrigatório) |
| **4** | 🟡 Amarelo | SDA / TX da micro:bit | *Não Conectado* (ou RX) | Opcional (se houver telemetria inversa direta) |

*Atenção: A Nezha V2 fornece 3.3V regulados nas portas azuis, o que elimina a necessidade de conversores de nível lógico entre as duas placas.*

---

## 4. Gerenciamento de Memória Dinâmica no ESP32-S3 (Alocação de 4.4MB na PSRAM)

Um Replay Buffer de 50.000 transições com 10 estados contínuos exige cerca de 4.4 MB de RAM. O código abaixo demonstra como forçar o compilador C++ a alocar essa estrutura gigantesca na **PSRAM externa** usando `ps_malloc()`, evitando o travamento da memória interna do chip.

```cpp
#include <Arduino.h>
#include <math.h>

// Configurações da Arquitetura DQN Expandida
const int STATE_SIZE = 10;       // Vetor de 10 estados contínuos
const int HIDDEN_SIZE = 64;      // Camada oculta expandida
const int ACTION_SIZE = 2;       // Número de ações possíveis
const int BATCH_SIZE = 32;       // Lote de treino
const int BUFFER_MAX = 50000;    // Exigência do experimento: 50k posições

const float GAMMA = 0.99;
float epsilon = 1.0;
const float LEARNING_RATE = 0.001;

// Estrutura de cada passo de experiência
struct Transition {
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    bool done;
};

// Ponteiro para o Replay Buffer Alocado na PSRAM
Transition* replayBuffer = nullptr;
int bufferSize = 0;
int bufferIndex = 0;

// Matrizes da Rede Neural (Pesos e Biases)
float pesos_ocultos[STATE_SIZE][HIDDEN_SIZE];
float bias_ocultos[HIDDEN_SIZE];
float pesos_saida[HIDDEN_SIZE][ACTION_SIZE];
float bias_saida[ACTION_SIZE];

void inicializarRede() {
    float limite_l1 = sqrt(6.0 / (STATE_SIZE + HIDDEN_SIZE));
    for(int i = 0; i < STATE_SIZE; i++)
        for(int j = 0; j < HIDDEN_SIZE; j++)
            pesos_ocultos[i][j] = ((float)rand() / RAND_MAX * 2.0 - 1.0) * limite_l1;
            
    for(int j = 0; j < HIDDEN_SIZE; j++) bias_ocultos[j] = 0.0;

    float limite_l2 = sqrt(6.0 / (HIDDEN_SIZE + ACTION_SIZE));
    for(int i = 0; i < HIDDEN_SIZE; i++)
        for(int j = 0; j < ACTION_SIZE; j++)
            pesos_saida[i][j] = ((float)rand() / RAND_MAX * 2.0 - 1.0) * limite_l2;
            
    for(int j = 0; j < ACTION_SIZE; j++) bias_saida[j] = 0.0;
}

void setup() {
    Serial.begin(115200);
    Serial1.begin(115200, SERIAL_8N1, 44, 43); // RX=44, TX=43 ligados na Nezha

    // Inicializa e valida a PSRAM física de 8MB (Módulo R8)
    if (!psramInit()) {
        Serial.println("ERRO: PSRAM nao encontrada!");
        while (true) delay(1000);
    }

    // Alocação dinâmica e segura de 4.4MB na RAM Externa
    replayBuffer = (Transition*) ps_malloc(BUFFER_MAX * sizeof(Transition));
    if (replayBuffer == nullptr) {
        Serial.println("ERRO: Falha crítica de alocação na PSRAM!");
        while (true) delay(1000);
    }

    Serial.println("Replay Buffer de 50.000 posições configurado com sucesso.");
    inicializarRede();
}
```

---

## 5. Protocolo de Comunicação Síncrona (Mestre-Escravo)

Para mitigar qualquer atraso ou latência na transmissão serial, o loop do robô adota um comportamento de **Espera Síncrona**. O ESP32-S3 congela temporariamente os passos lógicos da rede neural enquanto a micro:bit move fisicamente os motores da Nezha e calcula o ambiente, garantindo integridade matemática total na DQN.

```cpp
// Execução do Backpropagation manual com SGD simplificado
void treinarDQN() {
    if (bufferSize < BATCH_SIZE) return;

    for (int b = 0; b < BATCH_SIZE; b++) {
        int idx = rand() % bufferSize;
        Transition t = replayBuffer[idx];

        // Forward Pass Atual e Próximo (ReLU Oculta, Saída Linear)
        // [Implementação interna do Forward Pass Otimizado]
        
        // Equação de Bellman e cálculo do gradiente
        // [Cálculo do TD-Error e Ajuste dos pesos_ocultos e pesos_saida]
    }
    if (epsilon > 0.05) epsilon *= 0.9995;
}

void loop() {
    float estado_atual[STATE_SIZE] = {0.0}; 
    // [Coleta os 10 estados da Câmera local e variáveis internas]

    // 1. Escolha de Ação (Epsilon-Greedy)
    int acao = ( ((float)rand() / RAND_MAX) < epsilon ) ? (rand() % ACTION_SIZE) : 0; 

    // 2. Envio síncrono para a micro:bit via Serial
    Serial1.printf("A:%d\n", acao);

    // 3. BLOQUEIO DE ESPERA: Aguarda a micro:bit mover o robô e devolver a Recompensa e Sensores
    while(!Serial1.available()) { delay(1); }
    
    // Leitura da string contendo o Feedback (Ex: "R:-5,S1:23,S2:12...")
    String feedback = Serial1.readStringUntil('\n');
    
    float recompensa = 0.0;
    bool terminado = false;
    float proximo_estado[STATE_SIZE] = {0.0};
    
    // [Parseamento manual da string recebida para preencher a recompensa e os sensores da Nezha V2]

    // 4. Salva a transição de forma estável na PSRAM e executa o passo de treino contínuo
    // salvarExperiencia(estado_atual, acao, recompensa, proximo_estado, terminado);
    treinarDQN();
}
```

---

## 6. Lógica de Blocos no Microsoft MakeCode (micro:bit)

No ambiente de blocos ou JavaScript do MakeCode, configure o robô para agir como escravo do barramento:

1.  **No Iniciar (On Start):**
    *   Redirecionar a porta serial para os pinos correspondentes ao slot RJ11 conectado (`P14` e `P15` ou equivalentes da sua pinagem) na velocidade de **115200 bps**.
2.  **Ao Receber Linha Serial (On Serial Received `\n`):**
    *   Ler a string de comando vinda do ESP32-S3.
    *   Se o comando começar com `"A:1"`, acionar os motores da Nezha V2 usando o controle de odometria para andar uma distância fixa (ex: `Mover motor M1 e M2 por 10 cm`).
    *   Após o robô terminar fisicamente de se mover, ler os sensores de distância/colisão plugados na Nezha.
    *   Calcular a recompensa: Atribuir valor positivo (ex: `+10`) se o robô se deslocou livremente, ou valor negativo pesado (ex: `-100`) caso o sensor de colisão da base Nezha tenha sido ativado.
    *   Montar a string de resposta com todos os dados dos sensores e disparar de volta usando o bloco `Serial Gravar Linha (Serial Write Line)`.
