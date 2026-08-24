# ESP32-CAM Setup Guide — Robô de Entrega

Guia prático para instalar e configurar o ESP32-CAM (AI-Thinker, câmera OV2640) para rodar OpenCV, AprilTag ou YOLO, e comunicar-se com o Micro:bit.

---

## 1. Ambiente de Desenvolvimento

### Arduino IDE (recomendado para iniciantes)

1. **Instale o Arduino IDE** (versão 1.8.15+): https://www.arduino.cc/en/software
2. **Adicione o gerenciador de boards ESP32**:
   - `File > Preferences` → coloque na caixa de "Additional Boards Manager URLs":
     ```
     https://dl.espressif.com/dl/package_esp32_index.json
     ```
   - Reinicie o IDE.
   - `Tools > Board > Boards Manager` → instale **ESP32** (versão 2.0.0 ou superior).
3. **Selecione o board**:
   - `Tools > Board` → **AI-Thinker ESP32-CAM**
   - `Tools > Partition Scheme` → **Default 1.9MB APP / 1.9MB SPIFFS** (ou **16MB Flash** se precisar de mais espaço)
   - `Tools > CPU Frequency` → **240 MHz**
   - `Tools > Flash Frequency` → **80 MHz**
   - `Tools > Upload Speed` → **115200**
   - **Importante**: Marque **"USB CDC On Boot"** em `Tools > USB CDC On Boot` → **ON** (para ver Serial Monitor).

### PlatformIO (para usuários avançados)

- Instale o PlatformIO IDE (VS Code extension ou standalone).
- Crie um novo projeto: `PlatformIO > New Project` → board **AI-Thinker ESP32-CAM**.

---

## 2. Conexões Elétricas

| ESP32-CAM | Micro:bit | Função |
|---|---|---|
| **GND** | GND | Terra comum |
| **TX (GPIO 17)** | **RX (P0)** | Dados ESP32-CAM → Micro:bit |
| **RX (GPIO 16)** | **TX (P1)** | Dados Micro:bit → ESP32-CAM |
| **3.3V** | **3.3V** | Alimentação (comum) |
| **GPIO 12** | — | LED onboard (opcional) |
| **GPIO 2** | — | LED flash (opcional) |

**Aviso**: TX do ESP32-CAM (3.3V) → RX do Micro:bit (3.3V) é compatível. Se usar 5V no Micro:bit, adicione um divisor de tensão (10k + 20k) ou um level shifter.

---

## 3. Flashar o Firmware

### Opção A: Firmware AI-Thinker (padrão, já vem)

- O ESP32-CAM já vem com firmware AI-Thinker que permite:
  - Capturar imagens (JPEG, RAW)
  - Acessar via WiFi (http://192.168.4.1/)
  - Enviar via UART

### Opção B: Firmware Personalizado (OpenCV/AprilTag/YOLO)

Baixe e flash um firmware compatível:

| Firmware | O que roda | Link |
|---|---|---|
| **AI-Thinker AI-Camera** | Detecção de rosto, QR code, cor | https://ai-thinker.com/esp32-cam |
| **Espressif AI-Camera** | Detecção de rosto, cor, AprilTag | https://github.com/espressif/esp-who |
| **OpenCV Lite** | Detecção de cor, contornos, contagem | https://github.com/opencv/opencv |
| **YOLOv5-nano (NCNN)** | Detecção de objetos em tempo real | https://github.com/ultralytics/ultralytics |

**Como flashar**:
1. Baixe o firmware (.bin) desejado.
2. No Arduino IDE: `Tools > Sketch > Upload` (ou use `pio run --target upload` no PlatformIO).
3. Aguarde a conclusão e reinicie o ESP32-CAM.

---

## 4. Instalar Bibliotecas

### Bibliotecas essenciais para Arduino IDE

Instale via `Sketch > Include Libraries > Manage Libraries`:

| Biblioteca | Versão | Para |
|---|---|---|
| **esp32-camera** | 1.0.1+ | Captura de imagem |
| **esp32-fs** | 1.0.0+ | Sistema de arquivos |
| **ArduinoJson** | 6.21.0+ | Parse JSON |
| **esp-tflite-micro** | 2.0.0+ | Inferência do DQN (se quiser rodar no ESP32) |
| **apriltag** | 0.0.1+ | Detecção de AprilTag |

### Bibliotecas alternativas

| Biblioteca | Para |
|---|---|
| **OpenCV** (ESP32 port) | Processamento de imagem |
| **TensorFlow Lite** | Modelos ML leves |
| **NCNN** | YOLOv5-nano otimizado |

---

## 5. Código Básico — Captura e Envio via UART

### 5.1. Captura de imagem e detecção de cor (exemplo)

```cpp
#include <esp_camera.h>
#include <ArduinoJson.h>
#include <Wire.h>

// Pin definitions (AI-Thinker ESP32-CAM)
#define PWDN 32
#define RESET -1
#define XCLK 0
#define SIOD 26
#define SIOC 27
#define Y9 19
#define Y8 18
#define Y7 17
#define Y6 16
#define Y6 16
#define Y5 12
#define Y4 11
#define Y3 10
#define Y2 8
#define Y1 9
#define VSYNC 25
#define HREF 23
#define PCLK 21

static camera_config_t camera_config = {
  .pin_pwdn = PWDN,
  .pin_reset = RESET,
  .pin_xclk = XCLK,
  .pin_sioda = SIOD,
  .pin_sioc = SIOC,
  .pin_y9 = Y9,
  .pin_y8 = Y8,
  .pin_y7 = Y7,
  .pin_y6 = Y6,
  .pin_y5 = Y5,
  .pin_y4 = Y4,
  .pin_y3 = Y3,
  .pin_y2 = Y2,
  .pin_y1 = Y1,
  .pin_vsync = VSYNC,
  .pin_href = HREF,
  .pin_pclk = PCLK,
  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_JPEG,
  .frame_size = FRAMESIZE_QVGA, // 320x240
  .jpeg_quality = 12,
  .fb_count = 1
};

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17); // UART para Micro:bit

  // Inicializa câmera
  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.println("Erro ao inicializar câmera");
    return;
  }
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Falha ao capturar imagem");
    return;
  }

  // Processa imagem (ex.: detecção de cor)
  // Aqui você implementaria OpenCV ou YOLO

  // Exemplo: Envia dados via UART
  StaticJsonDocument<200> doc;
  doc["t"] = "obj";
  doc["x"] = 0.5;  // Coordenada relativa
  doc["y"] = -0.3;
  doc["d"] = 0.4;  // Distância
  doc["a"] = 0.2;  // Ângulo
  doc["c"] = 0.92; // Confiança

  char buffer[256];
  serializeJson(doc, buffer);
  Serial2.println(buffer);

  esp_camera_fb_return(fb);
  delay(100); // ~10 FPS
}
```

### 5.2. Detecção de AprilTag (exemplo)

```cpp
#include <apriltag.h>
#include <apriltag36h11.h>

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17);

  // Inicializa AprilTag
  apriltag_detector_t *td = apriltag36h11_create();
  apriltag_detector_add_family(td, apriltag36h11_create());

  // Inicializa câmera (como no exemplo anterior)
  // ...
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  // Converte para imagem em grayscale
  image_u8_t im = {
    .width = fb->width,
    .height = fb->height,
    .buf = fb->buf,
    .buf_size = fb->len
  };

  // Detecta tags
  zarray_t *detections = apriltag_detector_detect(td, &im);
  for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t det;
    zarray_get(detections, i, &det);

    // Envia dados via UART
    StaticJsonDocument<200> doc;
    doc["t"] = "tag";
    doc["id"] = det->id;
    doc["c"] = det->c[0];  // Centro x
    doc["c"] = det->c[1];  // Centro y
    doc["d"] = det->d;     // Distância
    doc["a"] = det->a;     // Ângulo

    char buffer[256];
    serializeJson(doc, buffer);
    Serial2.println(buffer);
  }
  apriltag_detections_destroy(detections);

  esp_camera_fb_return(fb);
  delay(100);
}
```

---

## 6. Comunicação com o Micro:bit

### 6.1. Dados enviados (JSON)

```json
{"t":"obj","id":"dice_red","x":0.12,"y":-0.03,"d":0.45,"a":0.2,"c":0.92}
{"t":"tag","id":42,"x":1.20,"y":0.05,"d":1.20,"a":-0.05}
```

### 6.2. Micro:bit recebe e processa

```javascript
// Micro:bit (MakeCode ou Python)
serial.onDataReceived(ReceivedDataString, () => {
  let data = JSON.parse(ReceivedDataString)
  if (data.t == "obj") {
    // Atualiza estado do robô
    robotState.distObj = data.d
    robotState.angleObj = data.a
  } else if (data.t == "tag") {
    robotState.distDel = data.d
    robotState.angleDel = data.a
  }
})
```

---

## 7. Limitações e Dicas

| Limitação | Mitigação |
|---|---|
| **Memória limitada** (520KB RAM) | Use modelos leves (YOLOv3-tiny, OpenCV Lite) |
| **FPS baixo** (10-15 FPS) | Ajuste o loop, use `delay(100)` |
| **Ruído de imagem** | Use filtros (Gaussian blur, Canny) |
| **Calibração de câmera** | Use `camera_calibrate` para corrigir distorções |
| **Conexão USB** | Use cabo USB OTG ou adaptador para programação |

---

## 8. Recursos Úteis

- [Documentação do ESP32-CAM](https://ai-thinker.com/esp32-cam)
- [Biblioteca OpenCV para ESP32](https://github.com/opencv/opencv)
- [YOLOv5-nano para ESP32](https://github.com/ultralytics/ultralytics)
- [AprilTag no ESP32](https://github.com/4DTech/apriltag)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32 Camera Library](https://github.com/espressif/esp32-camera)

---

## 9. Próximos Passos

1. **Flashar o firmware** desejado (AI-Thinker, OpenCV, ou YOLO).
2. **Testar a câmera** (captura e envio via UART).
3. **Integrar com o Micro:bit** (recebe JSON, processa, controla motores).
4. **Treinar o DQN** na simulação (Phaser) com ruído/latência simulada.
5. **Deployar no hardware** (exportar pesos do DQN para C/TFLite Micro).

Se precisar de ajuda com código específico (ex.: detecção de cor, AprilTag, ou comunicação UART), posso fornecer exemplos detalhados!