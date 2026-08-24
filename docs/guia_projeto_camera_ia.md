# Guia de Projeto: Robô Inteligente com ESP32-S3 AI-Camera e micro:bit (Nezha V2)

Este documento contém o planejamento completo, especificações de hardware, esquemas de ligação e códigos necessários para construir um robô autônomo capaz de detectar pequenos objetos (dados), calcular coordenadas tridimensionais de distância e se orientar no espaço utilizando marcadores fiduciários **AprilTags**.

---

## 1. Visão Geral do Sistema

O robô utiliza uma arquitetura de processamento distribuído (Edge AI + Atuação):
1. **ESP32-S3 (Subsistema de Visão):** Atua como o "cérebro visual". Roda o firmware OpenMV em MicroPython para processar as imagens da câmera em tempo real, identificar as AprilTags nas faces dos dados ou no ambiente, e calcular a distância e orientação 3D.
2. **micro:bit + Nezha V2 (Subsistema de Controle e Movimento):** Recebe as coordenadas espaciais tratadas via comunicação serial (UART) e controla os motores, servos e atuadores do robô para interagir com os objetos.

---

## 2. Especificações de Hardware

### 2.1 Placa de Desenvolvimento de Visão
* **Modelo Recomendado:** ESP32-S3 WROOM (Versão **N16R8**).
* **Requisito Obrigatório:** Deve possuir **8 MB de PSRAM** (indicado pelo sufixo R8) para suportar o armazenamento dos frames de imagem e o processamento matemático das matrizes tridimensionais.
* **Sensor de Imagem:** OV2640 (2 Megapixels), configurado em resolução **QVGA (320x240)** ou **QQVGA (160x120)** para manter uma alta taxa de quadros (FPS) e evitar desfoque por movimento (*motion blur*).

### 2.2 Controle e Expansão
* **Placa Principal:** BBC micro:bit (V1 ou V2).
* **Base de Expansão:** Elecfreaks Nezha V2 (fornece conexões estruturadas para motores e sensores da linha PlanetX através de portas RJ11).

---

## 3. Confecção do Cubo e Tags Fiduciárias (AMS)

Para que o sistema meça distâncias reais através de uma única lente, os objetos devem possuir dimensões conhecidas.

### 3.1 Configurações de Impressão 3D (Bambu Studio / AMS)
* **Tamanho do Dado:** Recomenda-se um cubo de **35 mm a 40 mm** de aresta. Dados muito pequenos (ex: 15 mm) limitam a distância de leitura a poucos centímetros na resolução QVGA.
* **Família da AprilTag:** Utilize a família **`TAG16H5`**. Ela possui uma grade interna menor (4x4), gerando blocos pretos e brancos maiores e muito mais fáceis de serem focados pela câmera de longe.
* **Filamento:** Utilize obrigatoriamente filamentos **Matte (Fosco)** para o preto e para o branco. Plásticos brilhantes (PLA comum ou Silk) geram reflexos de luz que invalidam a leitura do algoritmo.
* **Fatiamento:** Configure a altura de camada para `0.12 mm` ou inferior para garantir cantos geométricos afiados nas tags. Mantenha uma borda branca (*Quiet Zone*) de pelo menos 2 a 3 mm nas arestas do dado isolando o código preto.

### 3.2 Mapeamento das Faces do Dado
Imprima uma tag diferente para cada face para identificar o valor do dado e a sua rotação:
* **ID 0:** Face do número 1
* **ID 1:** Face do número 2
* **ID 2:** Face do número 3
* **ID 3:** Face do número 4
* **ID 4:** Face do número 5
* **ID 5:** Face do número 6
* **IDs 10+:** Podem ser impressos e espalhados pelo cenário como "Âncoras" fixas de localização global.

---

## 4. Conexões Elétricas (Cabo Customizado RJ11)

A conexão entre a ESP32-S3 e a Nezha V2 é feita utilizando uma das **Portas Azuis (IIC/Serial)** da base expansora. Os pinos digitais internos dessa porta serão reconfigurados no software para comunicação Serial UART de 3.3V.

Olhando o conector RJ11 (padrão 4P4C) de frente, com os contatos metálicos para cima e a trava para trás, a pinagem deve ser soldada nos pinos correspondentes da ESP32-S3:

| Pino RJ11 | Cor Padrão Elecfreaks | Função na Porta Azul (Nezha) | Conexão na ESP32-S3 | Nota de Segurança |
| :---: | :--- | :--- | :--- | :--- |
| **1** | 🟢 Verde | Sinal 1 (RX da micro:bit) | **TX** (ex: GPIO 43) | Transporta os dados de visão para o robô |
| **2** | 🔴 Vermelho | VCC (Alimentação 3.3V) | **3V3** ou **VIN** | *Verificar se a sua porta fornece 3.3V estável antes de alimentar* |
| **3** | ⚫ Preto | GND (Terra) | **GND** | Obrigatório para unificar o ponto de referência |
| **4** | 🟡 Amarelo | Sinal 2 (TX da micro:bit) | *Não Conectado* | Dispensável para comunicação unidirecional |

---

## 5. Implementação do Software

O desenvolvimento na ESP32-S3 dispensa o uso obrigatório da OpenMV IDE para programação, podendo ser feito inteiramente através do **VS Code** utilizando extensões de MicroPython como a **MicroPico** para sincronizar os arquivos na memória flash do chip.

### 5.1 Código da ESP32-S3 (`main.py` - MicroPython)

Este script deve ser salvo na raiz da memória da placa. Ele inicializa a câmera, localiza as tags da família `TAG16H5`, calcula a distância Z e envia uma string compacta via UART formatada como `ID,Distancia,Inclinacao\n`.

```python
import sensor, image, time
from machine import UART

# 1. Inicialização da Porta Serial (UART 1)
# Configure a velocidade para 115200 bps e mapeie os pinos TX/RX da sua placa física
uart = UART(1, baudrate=115200, tx=43, rx=44)

# 2. Inicialização do Sensor de Câmera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # Resolução equilibrada para velocidade e RAM (320x240)
sensor.skip_frames(time = 2000)     # Aguarda estabilização da exposição automática

while(True):
    img = sensor.snapshot()
    
    # Executa a busca por AprilTags da família correta
    for tag in img.find_apriltags(families=image.TAG16H5):
        
        tag_id = tag.id()
        
        # O método z_translation devolve a distância estimada. 
        # O valor é multiplicado por -1 para calibração de orientação dependendo da lente.
        # IMPORTANTE: No OpenMV IDE original, passa-se o tamanho real no método se necessário,
        # ou aplica-se a proporção linear com base no tamanho físico da tag impressa.
        distancia_z = int(tag.z_translation() * -1)
        
        # Converte a rotação Y (Yaw) de radianos para graus inteiros
        rotacao_y = int(tag.y_rotation() * (180 / 3.14159))
        
        # Empacota os dados de forma compacta delimitados por vírgula
        # Exemplo de saída: "2,250,15
" (Dado ID 2, a 250mm de distância, inclinado 15 graus)
        payload = f"{tag_id},{distancia_z},{rotacao_y}
"
        
        # Transmite fisicamente o pacote de dados para a micro:bit através do cabo RJ11
        uart.write(payload)
        
    time.sleep_ms(50) # Evita sobrecarregar o buffer de recepção da micro:bit
```

### 5.2 Lógica de Recepção na micro:bit (MakeCode)

Para programar a recepção dos dados na micro:bit no bloco do robô, use as diretrizes de blocos do MakeCode estruturadas na seguinte lógica:

1. **No Iniciar (On Start):**
   * Use o bloco `serial redirecionar para` (Serial redirect to).
   * Defina os pinos correspondentes à porta RJ11 conectada (ex: `TX = P14`, `RX = P13`, dependendo do mapeamento físico da porta azul escolhida na Nezha).
   * Defina o Baud Rate para `115200`.

2. **Ao Receber Linha Serial (On Serial Data Received):**
   * Crie uma variável chamada `linha_recebida` e atribua a ela o comando `serial ler linha` (serial read line).
   * Crie uma lista chamada `dados_fatiados` e atribua o bloco de texto `separar linha_recebida em ","` (split string).
   * Atribua as variáveis do seu robô indexando os elementos da lista:
     * `ID_Objeto` = `dados_fatiados[0]` (obtém o ID do dado/âncora)
     * `Distancia` = `dados_fatiados[1]` (obtém a distância em milímetros)
     * `Angulo` = `dados_fatiados[2]` (obtém a rotação)

3. **Tomada de Decisão do Robô (Exemplo Algorítmico):**
   * *Se `ID_Objeto == "0"` e `Distancia < "150"`:* Acione os motores da Nezha V2 para recuar (Objeto muito próximo).
   * *Se `ID_Objeto == "10"` (Âncora da Mesa):* Use o valor de `Angulo` para alinhar as rodas do robô paralelamente à borda da mesa, garantindo navegação reta no cenário.

---
Guia gerado automaticamente para documentação técnica de robótica educacional e computação embarcada.
