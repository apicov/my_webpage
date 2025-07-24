# TinyML Tutorial: Keras 3.0 + Google Edge AI + ESP32

## 📚 Introduction

This comprehensive tutorial covers TinyML development using Keras 3.0, Google's Edge AI tools, and deployment on ESP32. You'll learn to build, optimize, and deploy machine learning models for edge devices.

**What you'll learn:**
- Keras 3.0 multi-backend development
- Model optimization techniques (quantization, pruning)
- TensorFlow Lite conversion and deployment
- ESP32 IDF development with TinyML
- Real-world IoT applications

---

## 🎯 Prerequisites

### Hardware Requirements:
- **ESP32 development board** (ESP32-WROOM-32 recommended)
- **USB cable** for programming
- **Computer** with Python 3.8+

### Software Requirements:
- **Python 3.8+**
- **ESP-IDF** (Espressif IoT Development Framework)
- **Keras 3.0**
- **TensorFlow 2.x**
- **Git**

---

## 🏗️ Chapter 1: Keras 3.0 Fundamentals

### Installation

```bash
# Install Keras 3.0 with TensorFlow backend
pip install keras tensorflow

# Install with JAX backend (faster training)
pip install keras jax jaxlib

# Install with PyTorch backend
pip install keras torch
```

### Multi-Backend Setup

```python
import keras

# Check available backends
print(keras.backend.backends())  # ['tensorflow', 'jax', 'torch']

# Set backend
keras.config.set_backend('jax')  # For faster training
keras.config.set_backend('tensorflow')  # For deployment
```

### Basic Model Building

```python
import keras
from keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile and train
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🎯 Chapter 2: Model Optimization Techniques

### Quantization

**Post-Training Quantization:**
```python
import tensorflow as tf
import keras

# Build and train model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Representative dataset for calibration
def representative_dataset():
    for data in x_train[:100]:
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
```

**Quantization-Aware Training:**
```python
import tensorflow_model_optimization as tfmot

# Apply quantization to layers
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# Train with quantization awareness
q_aware_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
q_aware_model.fit(x_train, y_train, epochs=10)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Pruning

**Magnitude-Based Pruning:**
```python
import tensorflow_model_optimization as tfmot

# Apply pruning to model
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Define pruning schedule
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.8,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Compile and train
model_for_pruning.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add pruning callback
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
]

model_for_pruning.fit(
    x_train, y_train,
    callbacks=callbacks,
    epochs=10
)

# Strip pruning for deployment
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

**Structured Pruning:**
```python
# Apply structured pruning to specific layers
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Prune only dense layers
for layer in model.layers:
    if isinstance(layer, keras.layers.Dense):
        layer = prune_low_magnitude(layer, **pruning_params)
```

### Knowledge Distillation

```python
# Teacher model (large)
teacher_model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Student model (small)
student_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Custom distillation loss
def distillation_loss(y_true, y_pred, temperature=3.0):
    # Soft targets from teacher
    soft_targets = teacher_model.predict(x_train)
    soft_targets = soft_targets / temperature
    
    # Hard targets
    hard_targets = y_true
    
    # Combined loss
    soft_loss = keras.losses.categorical_crossentropy(soft_targets, y_pred)
    hard_loss = keras.losses.categorical_crossentropy(hard_targets, y_pred)
    
    return 0.7 * soft_loss + 0.3 * hard_loss

# Train student model
student_model.compile(
    optimizer='adam',
    loss=distillation_loss,
    metrics=['accuracy']
)
student_model.fit(x_train, y_train, epochs=10)
```

---

## 🚀 Chapter 3: TensorFlow Lite Conversion

### Basic Conversion

```python
import tensorflow as tf
import keras

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Advanced Conversion Options

```python
# Conversion with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set target specs
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Enable experimental features
converter.experimental_new_converter = True

# Convert
tflite_model = converter.convert()
```

### Model Analysis

```python
import tensorflow as tf

# Analyze TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Get model size
import os
model_size = os.path.getsize("model.tflite")
print(f"Model size: {model_size / 1024:.2f} KB")
```

---

## 🔧 Chapter 4: ESP32 Development Setup

### ESP-IDF Installation

```bash
# Clone ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf

# Install ESP-IDF
./install.sh

# Set up environment
source export.sh
```

### Project Structure

```
esp32_tinyml/
├── CMakeLists.txt
├── main/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── model.h
│   └── model.tflite
├── components/
│   └── tflite-micro/
└── sdkconfig
```

### CMakeLists.txt Configuration

```cmake
# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_tinyml)
```

```cmake
# main/CMakeLists.txt
idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES tflite-micro
)
```

### Basic ESP32 Application

```cpp
// main.cpp
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Global variables
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// Tensor arena
static constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void)
{
    // Initialize error reporter
    error_reporter->Report("ESP32 TinyML Starting...");

    // Load model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema mismatch!");
        return;
    }

    // Create resolver
    static tflite::AllOpsResolver resolver;

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        return;
    }

    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Main loop
    while (true) {
        // Prepare input data
        for (int i = 0; i < input->dims->data[1]; i++) {
            input->data.f[i] = 0.5f;  // Example input
        }

        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed!");
            return;
        }

        // Process output
        float max_value = 0;
        int max_index = 0;
        for (int i = 0; i < output->dims->data[1]; i++) {
            if (output->data.f[i] > max_value) {
                max_value = output->data.f[i];
                max_index = i;
            }
        }

        printf("Prediction: %d (confidence: %.2f)\n", max_index, max_value);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
```

---

## 🎯 Chapter 5: Advanced Optimization Techniques

### Model Architecture Optimization

```python
# MobileNet-style architecture
def create_mobile_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Depthwise separable convolutions
    for filters in [64, 128, 256]:
        x = layers.DepthwiseConv2D(3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Mixed Precision Training

```python
# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Build model (will use float16 where possible)
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Training will be faster and use less memory
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

### Model Compression Pipeline

```python
def optimize_model_for_edge(model, x_train, y_train):
    """Complete optimization pipeline"""
    
    # 1. Pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.7,
            begin_step=0,
            end_step=1000
        )
    }
    
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy')
    
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_for_pruning.fit(x_train, y_train, callbacks=callbacks, epochs=5)
    
    # 2. Strip pruning
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    # 3. Quantization
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model_for_export)
    
    q_aware_model.compile(optimizer='adam', loss='categorical_crossentropy')
    q_aware_model.fit(x_train, y_train, epochs=3)
    
    # 4. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset
    def representative_dataset():
        for data in x_train[:100]:
            yield [data.astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    
    return tflite_model
```

---

## 🔧 Chapter 6: ESP32 Advanced Features

### Sensor Integration

```cpp
// main.cpp with sensor integration
#include "driver/adc.h"
#include "esp_adc_cal.h"

// ADC configuration
#define ADC1_CHANNEL (ADC1_CHANNEL_6)
#define ADC_ATTEN ADC_ATTEN_DB_11
#define ADC_WIDTH ADC_WIDTH_BIT_12

// Calibration
static esp_adc_cal_characteristics_t adc_chars;

void setup_adc() {
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN, ADC_WIDTH, 1100, &adc_chars);
    adc1_config_width(ADC_WIDTH);
    adc1_config_channel_atten(ADC1_CHANNEL, ADC_ATTEN);
}

uint32_t read_sensor() {
    uint32_t adc_reading = 0;
    for (int i = 0; i < 10; i++) {
        adc_reading += adc1_get_raw(ADC1_CHANNEL);
    }
    adc_reading /= 10;
    return esp_adc_cal_raw_to_voltage(adc_reading, &adc_chars);
}
```

### WiFi Integration

```cpp
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"

static const char* TAG = "TinyML";

void wifi_init_sta() {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "YOUR_WIFI_SSID",
            .password = "YOUR_WIFI_PASSWORD",
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");
}
```

### Real-time Inference

```cpp
// Real-time inference task
void inference_task(void* parameter) {
    while (true) {
        // Read sensor data
        uint32_t sensor_value = read_sensor();
        
        // Preprocess data
        float input_data[1] = {(float)sensor_value / 4095.0f};
        
        // Copy to input tensor
        for (int i = 0; i < input->dims->data[1]; i++) {
            input->data.f[i] = input_data[i];
        }
        
        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status == kTfLiteOk) {
            // Process results
            float prediction = output->data.f[0];
            ESP_LOGI(TAG, "Prediction: %.3f", prediction);
            
            // Send results via WiFi if needed
            // send_prediction(prediction);
        }
        
        vTaskDelay(pdMS_TO_TICKS(100));  // 10Hz inference
    }
}
```

---

## 🚀 Chapter 7: Deployment and Testing

### Model Validation

```python
# Validate TFLite model
import tensorflow as tf
import numpy as np

def validate_tflite_model(tflite_path, test_data, test_labels):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test predictions
    correct = 0
    total = len(test_data)
    
    for i in range(total):
        # Prepare input
        input_data = test_data[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data[0])
        true_label = np.argmax(test_labels[i])
        
        if prediction == true_label:
            correct += 1
    
    accuracy = correct / total
    print(f"TFLite Model Accuracy: {accuracy:.4f}")
    return accuracy
```

### Performance Benchmarking

```python
import time

def benchmark_model(tflite_path, test_data, num_runs=1000):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_data[0:1])
        interpreter.invoke()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], test_data[0:1])
        interpreter.invoke()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"Average inference time: {avg_time:.2f} ms")
    return avg_time
```

### ESP32 Performance Monitoring

```cpp
// Performance monitoring
#include "esp_timer.h"

void benchmark_inference() {
    const int num_runs = 1000;
    uint64_t total_time = 0;
    
    for (int i = 0; i < num_runs; i++) {
        uint64_t start_time = esp_timer_get_time();
        
        // Run inference
        interpreter->Invoke();
        
        uint64_t end_time = esp_timer_get_time();
        total_time += (end_time - start_time);
    }
    
    float avg_time = (float)total_time / num_runs / 1000.0f;  // Convert to ms
    ESP_LOGI(TAG, "Average inference time: %.2f ms", avg_time);
}
```

---

## 🎯 Chapter 8: Real-World Applications

### Audio Classification

```python
# Audio classification model
def create_audio_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # 1D convolutions for audio
    x = layers.Conv1D(32, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Gesture Recognition

```python
# Gesture recognition with accelerometer data
def create_gesture_model(input_shape, num_gestures):
    inputs = keras.Input(shape=input_shape)  # (time_steps, 3) for x,y,z
    
    # LSTM for temporal patterns
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_gestures, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Anomaly Detection

```python
# Autoencoder for anomaly detection
def create_anomaly_detector(input_shape):
    # Encoder
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(16, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.Dense(64, activation='relu')(x)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(x)
    
    return keras.Model(inputs, decoded)
```

---

## 📚 Chapter 9: Best Practices

### Memory Management

```cpp
// Optimize memory usage
#define TENSOR_ARENA_SIZE (50 * 1024)  // Adjust based on model
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

// Use PSRAM if available
#ifdef CONFIG_SPIRAM_SUPPORT
    static uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(
        TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
#endif
```

### Power Optimization

```cpp
// Power management
#include "esp_pm.h"
#include "esp_sleep.h"

void setup_power_management() {
    // Configure dynamic frequency scaling
    esp_pm_config_esp32_t pm_config = {
        .max_freq_mhz = 240,
        .min_freq_mhz = 10,
        .light_sleep_enable = true
    };
    esp_pm_configure(&pm_config);
}

// Sleep between inferences
void power_save_mode() {
    esp_sleep_enable_timer_wakeup(1000000);  // 1 second
    esp_light_sleep_start();
}
```

### Error Handling

```cpp
// Robust error handling
TfLiteStatus run_inference(float* input_data, float* output_data) {
    // Validate inputs
    if (!input_data || !output_data) {
        ESP_LOGE(TAG, "Invalid input/output data");
        return kTfLiteError;
    }
    
    // Set input tensor
    TfLiteStatus status = interpreter->input(0)->data.f = input_data;
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to set input tensor");
        return status;
    }
    
    // Run inference
    status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return status;
    }
    
    // Get output
    memcpy(output_data, interpreter->output(0)->data.f, 
           sizeof(float) * interpreter->output(0)->dims->data[1]);
    
    return kTfLiteOk;
}
```

---

## 🚀 Chapter 10: Advanced Techniques

### Model Ensembling

```python
# Ensemble multiple small models
def create_ensemble(models, weights=None):
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(x):
        predictions = []
        for model in models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    return ensemble_predict
```

### Transfer Learning

```python
# Transfer learning for edge devices
def create_transfer_model(base_model, num_classes):
    # Freeze base model
    base_model.trainable = False
    
    # Add classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(base_model.input, outputs)
    
    # Fine-tune last few layers
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    return model
```

### Custom Operations

```cpp
// Custom TFLite operation
TfLiteRegistration* Register_CUSTOM_OP() {
    static TfLiteRegistration r = {
        .init = CustomOpInit,
        .free = CustomOpFree,
        .prepare = CustomOpPrepare,
        .invoke = CustomOpEval,
    };
    return &r;
}

TfLiteStatus CustomOpEval(TfLiteContext* context, TfLiteNode* node) {
    // Custom operation implementation
    const TfLiteTensor* input = GetInput(context, node, 0);
    TfLiteTensor* output = GetOutput(context, node, 0);
    
    // Your custom logic here
    for (int i = 0; i < input->dims->data[0]; i++) {
        output->data.f[i] = input->data.f[i] * 2.0f;  // Example: multiply by 2
    }
    
    return kTfLiteOk;
}
```

---

## 🎉 Conclusion

You now have a comprehensive understanding of:

✅ **Keras 3.0** - Multi-backend development  
✅ **Model optimization** - Quantization, pruning, distillation  
✅ **TensorFlow Lite** - Conversion and deployment  
✅ **ESP32 development** - TinyML on microcontrollers  
✅ **Real-world applications** - Audio, gesture, anomaly detection  
✅ **Best practices** - Memory, power, error handling  

### Next Steps:

1. **Start with simple models** - Build confidence with basic examples
2. **Experiment with optimization** - Try different techniques
3. **Deploy to ESP32** - Get hands-on experience
4. **Build real applications** - Apply to your IoT projects

### Resources:

- **TensorFlow Lite Micro**: https://github.com/tensorflow/tflite-micro
- **ESP-IDF**: https://docs.espressif.com/projects/esp-idf/
- **Keras 3.0**: https://keras.io/
- **Model Optimization**: https://www.tensorflow.org/model_optimization

**Happy TinyML development!** 🚀

---

*Build amazing edge AI applications!* 🎯 