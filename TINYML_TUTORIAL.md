# TinyML Tutorial: Keras 3.0 + Google Edge AI + ESP32

## 📚 Introduction

This comprehensive tutorial covers TinyML development using Keras 3.0, Google's Edge AI tools, and deployment on ESP32. You'll learn to build, optimize, and deploy machine learning models for edge devices.

### What is TinyML?

**TinyML** is the field of machine learning that focuses on running ML models on resource-constrained devices like microcontrollers, sensors, and edge devices. It brings AI to the "edge" of the network, enabling:

- **Real-time inference** without cloud connectivity
- **Privacy-preserving** AI (data stays on device)
- **Low-power operation** for battery-powered devices
- **Low-latency responses** for time-critical applications
- **Cost-effective deployment** without expensive cloud infrastructure

### Why TinyML Matters

Traditional ML requires powerful computers or cloud servers, but many applications need AI to run on small, inexpensive devices:

1. **IoT Devices**: Smart sensors, wearables, home automation
2. **Embedded Systems**: Industrial monitoring, automotive, medical devices
3. **Edge Computing**: Local processing for privacy and speed
4. **Battery-Powered Devices**: Long battery life with AI capabilities

### The TinyML Challenge

Running ML on microcontrollers is challenging because they have:
- **Limited Memory**: Often less than 1MB RAM
- **Limited Storage**: Flash memory constraints
- **Limited Processing Power**: Single-core, low-frequency CPUs
- **Power Constraints**: Battery life requirements
- **Real-time Requirements**: Predictable timing

### Keras 3.0 + TinyML Advantages

Keras 3.0 brings several benefits to TinyML development:

1. **Multi-backend Support**: Choose the best backend for your workflow
   - **TensorFlow**: Best for deployment and TensorFlow Lite
   - **JAX**: Fastest for training and research
   - **PyTorch**: Popular in research community

2. **Unified API**: Same code works across backends
3. **Better Performance**: Optimized for modern hardware
4. **Easier Deployment**: Simplified model conversion
5. **Integration**: Works seamlessly with TensorFlow Lite

### The ESP32 Platform

The **ESP32** is an ideal platform for TinyML because it:
- **Cost-effective**: ~$5-10 per board
- **Feature-rich**: WiFi, Bluetooth, multiple cores
- **Well-supported**: Extensive documentation and community
- **Power-efficient**: Low-power modes available
- **Real-time capable**: Predictable timing

### What You'll Build

Throughout this tutorial, you'll create:
1. **Keyword Spotting**: "Hey Alexa" detection
2. **Gesture Recognition**: Hand gesture classification
3. **Anomaly Detection**: Sensor data monitoring
4. **Image Classification**: Ultra-lightweight vision models

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

### Understanding Keras 3.0 Multi-Backend Architecture

Keras 3.0 introduces a revolutionary multi-backend architecture that allows you to choose the best backend for your specific needs:

#### **Backend Options:**

1. **TensorFlow Backend** (Recommended for TinyML)
   - **Best for**: Deployment, TensorFlow Lite conversion
   - **Advantages**: Mature ecosystem, excellent TinyML support
   - **Use when**: Building models for edge deployment

2. **JAX Backend** (Best for Training)
   - **Best for**: Fast training, research, experimentation
   - **Advantages**: Compilation optimization, parallel processing
   - **Use when**: Training large models or doing research

3. **PyTorch Backend** (Alternative)
   - **Best for**: Research, PyTorch ecosystem integration
   - **Advantages**: Dynamic computation, research community
   - **Use when**: Working with PyTorch-based workflows

#### **Why Multi-Backend Matters for TinyML:**

- **Training**: Use JAX for fast model development
- **Optimization**: Use TensorFlow for quantization and pruning
- **Deployment**: Use TensorFlow for TensorFlow Lite conversion
- **Flexibility**: Switch backends without changing code

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

#### **Understanding Backend Configuration**

The multi-backend setup allows you to seamlessly switch between different computation engines. This is particularly powerful for TinyML development where you might want to:
- **Train** with JAX for speed
- **Optimize** with TensorFlow for quantization
- **Deploy** with TensorFlow for TensorFlow Lite

#### **Backend Selection Strategy:**

1. **Development Phase**: Use JAX for fast iteration
2. **Optimization Phase**: Switch to TensorFlow for quantization
3. **Deployment Phase**: Use TensorFlow for TensorFlow Lite conversion

```python
import keras

# Check available backends
print(keras.backend.backends())  # ['tensorflow', 'jax', 'torch']

# Set backend for different phases of development
keras.config.set_backend('jax')  # For faster training
keras.config.set_backend('tensorflow')  # For deployment

# You can also check which backend is currently active
print(f"Current backend: {keras.config.backend()}")

# And get backend-specific information
if keras.config.backend() == 'tensorflow':
    print("Using TensorFlow backend - optimal for TinyML deployment")
elif keras.config.backend() == 'jax':
    print("Using JAX backend - optimal for fast training")
elif keras.config.backend() == 'torch':
    print("Using PyTorch backend - good for research")
```

### Basic Model Building

#### **Understanding Model Architectures for TinyML**

When building models for TinyML, you need to consider several constraints:

1. **Memory Constraints**: Models must fit in limited RAM
2. **Storage Constraints**: Models must fit in flash memory
3. **Power Constraints**: Models must be computationally efficient
4. **Real-time Constraints**: Models must run within timing requirements

#### **TinyML Model Design Principles:**

- **Start Small**: Begin with minimal architectures
- **Quantize Early**: Use quantization from the start
- **Profile Memory**: Monitor memory usage during development
- **Test on Target**: Validate on actual hardware early

#### **Sequential vs Functional API:**

- **Sequential**: Simpler, good for linear architectures
- **Functional**: More flexible, good for complex models
- **For TinyML**: Sequential is often sufficient and more memory-efficient

```python
import keras
from keras import layers

# Sequential model - Good for TinyML
# This creates a simple feedforward neural network
model = keras.Sequential([
    # Input layer: 784 features (e.g., flattened 28x28 image)
    layers.Dense(128, activation='relu', input_shape=(784,)),
    # Dropout for regularization (reduces overfitting)
    layers.Dropout(0.2),
    # Hidden layer: 64 neurons
    layers.Dense(64, activation='relu'),
    # Output layer: 10 classes (e.g., digits 0-9)
    layers.Dense(10, activation='softmax')
])

# Functional API - More flexible
# This allows for more complex architectures
inputs = keras.Input(shape=(784,))  # Define input shape
x = layers.Dense(128, activation='relu')(inputs)  # First layer
x = layers.Dropout(0.2)(x)  # Regularization
x = layers.Dense(64, activation='relu')(x)  # Hidden layer
outputs = layers.Dense(10, activation='softmax')(x)  # Output layer
model = keras.Model(inputs, outputs)  # Create model

# Compile and train
# For TinyML, consider using smaller learning rates
model.compile(
    optimizer='adam',  # Adaptive learning rate optimizer
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Model summary - Important for TinyML to understand model size
model.summary()
```

### 🔍 **Understanding Model Building for TinyML**

Now that we've covered the basics, let's understand the key considerations for TinyML model development:

#### **1. Model Size Analysis**

The `model.summary()` output is crucial for TinyML because it shows:
- **Total Parameters**: Number of weights and biases
- **Trainable Parameters**: Parameters that will be updated during training
- **Model Size**: Memory footprint of the model

**Example Output Analysis:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
dense (Dense)               (None, 128)              100,480   
dropout (Dropout)           (None, 128)              0         
dense_1 (Dense)             (None, 64)               8,256     
dense_2 (Dense)             (None, 10)               650       
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
```

**TinyML Considerations:**
- **109,386 parameters** = ~437KB in float32 (4 bytes per parameter)
- **After quantization** = ~109KB in int8 (1 byte per parameter)
- **ESP32 RAM**: 520KB total, so this model fits comfortably

#### **2. Memory Usage Breakdown**

**Training Memory (Development):**
- Model parameters: 437KB
- Gradients: 437KB
- Optimizer state: ~874KB
- **Total**: ~1.7MB (too large for ESP32)

**Inference Memory (Deployment):**
- Model parameters: 109KB (quantized)
- Activation buffers: ~50KB
- **Total**: ~160KB (fits in ESP32 RAM)

#### **3. TinyML Design Patterns**

**Layer Selection for TinyML:**
- **Dense Layers**: Good for small models, simple computations
- **Convolutional Layers**: Efficient for image processing
- **LSTM/GRU**: Avoid for TinyML (too memory-intensive)
- **Attention**: Too complex for most microcontrollers

**Activation Functions:**
- **ReLU**: Fast, good for TinyML
- **Sigmoid/Tanh**: Slower, avoid if possible
- **Softmax**: Only for output layer

**Regularization:**
- **Dropout**: Good for training, removed for inference
- **BatchNorm**: Can be fused with layers for efficiency
- **L1/L2**: Built into optimizer, minimal overhead

#### **4. Backend Selection Strategy**

**Development Workflow:**
1. **Training**: Use JAX backend for speed
2. **Optimization**: Switch to TensorFlow for quantization
3. **Deployment**: Use TensorFlow for TensorFlow Lite

**Code Example:**
```python
# Phase 1: Training with JAX
keras.config.set_backend('jax')
model.fit(x_train, y_train, epochs=10)

# Phase 2: Optimization with TensorFlow
keras.config.set_backend('tensorflow')
# Apply quantization, pruning, etc.

# Phase 3: Deployment
# Convert to TensorFlow Lite
```

#### **5. Model Architecture Guidelines**

**For ESP32 (520KB RAM):**
- **Parameters**: < 100K (quantized)
- **Layers**: < 10 layers
- **Input size**: < 10KB per inference
- **Output size**: < 1KB per inference

**Memory Budget Example:**
- Model parameters: 100KB
- Activation buffers: 50KB
- Input/output buffers: 20KB
- System overhead: 50KB
- **Total**: 220KB (42% of ESP32 RAM)

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

## 🔧 **Troubleshooting & Common Issues**

### **Issue 1: Model Too Large for ESP32**

**Problem**: Model exceeds available memory.

**Symptoms**:
- Compilation errors about memory
- Runtime crashes during inference
- Model doesn't fit in flash memory

**Solutions**:
```python
# 1. Reduce model size
model = keras.Sequential([
    layers.Dense(32, activation='relu'),  # Smaller layers
    layers.Dense(16, activation='relu'),  # Fewer neurons
    layers.Dense(1, activation='sigmoid')
])

# 2. Use aggressive quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Use float16

# 3. Apply pruning
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)
```

### **Issue 2: Poor Model Accuracy After Quantization**

**Problem**: Model accuracy drops significantly after quantization.

**Solutions**:
```python
# 1. Use quantization-aware training
qat_model = tfmot.quantization.keras.quantize_model(model)
qat_model.compile(optimizer='adam', loss='binary_crossentropy')
qat_model.fit(x_train, y_train, epochs=10)

# 2. Use mixed-precision quantization
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# 3. Calibrate with representative dataset
def representative_dataset():
    for data in calibration_data:
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset
```

### **Issue 3: ESP32 Inference Too Slow**

**Problem**: Model inference takes too long.

**Solutions**:
```cpp
// 1. Enable ESP32 optimizations
#define TFLITE_MICRO_USE_ESP32_OPTIMIZED_KERNELS 1

// 2. Use smaller data types
#define TFLITE_MICRO_USE_INT8 1

// 3. Optimize memory allocation
#define TFLITE_MICRO_USE_STATIC_MEMORY 1
```

### **Issue 4: Memory Fragmentation**

**Problem**: ESP32 runs out of memory during operation.

**Solutions**:
```cpp
// 1. Use static memory allocation
static uint8_t tensor_arena[CONFIG_SPIRAM_SUPPORT ? 1024 * 1024 : 100 * 1024];

// 2. Implement memory pooling
void* allocate_memory(size_t size) {
    static uint8_t memory_pool[8192];
    static size_t used = 0;
    
    if (used + size <= sizeof(memory_pool)) {
        void* ptr = &memory_pool[used];
        used += size;
        return ptr;
    }
    return NULL;
}
```

---

## 🎯 **Advanced Learning Challenges**

### **Challenge 1: Multi-Modal TinyML**

Create a system that combines audio and sensor data:

```python
# Audio + Accelerometer fusion
def create_multimodal_model():
    # Audio branch
    audio_input = layers.Input(shape=(audio_length,))
    audio_features = layers.Dense(64, activation='relu')(audio_input)
    
    # Sensor branch
    sensor_input = layers.Input(shape=(sensor_length, 3))
    sensor_features = layers.LSTM(32)(sensor_input)
    
    # Fusion
    combined = layers.Concatenate()([audio_features, sensor_features])
    output = layers.Dense(num_classes, activation='softmax')(combined)
    
    return keras.Model([audio_input, sensor_input], output)
```

### **Challenge 2: Continual Learning on Edge**

Implement a system that learns from new data without forgetting:

```python
def continual_learning_update(model, new_data, new_labels):
    # Store important weights
    important_weights = model.get_weights()
    
    # Train on new data
    model.fit(new_data, new_labels, epochs=5)
    
    # Apply elastic weight consolidation
    for i, (old_w, new_w) in enumerate(zip(important_weights, model.get_weights())):
        # Prevent large changes to important weights
        model.layers[i].set_weights([new_w * 0.9 + old_w * 0.1])
```

### **Challenge 3: Federated Learning on ESP32**

Implement federated learning where multiple ESP32 devices collaborate:

```cpp
// Federated learning implementation
class FederatedLearning {
private:
    float local_weights[MAX_WEIGHTS];
    float global_weights[MAX_WEIGHTS];
    
public:
    void train_local_model(float* data, float* labels, int num_samples) {
        // Train model on local data
        for (int i = 0; i < num_samples; i++) {
            // Gradient descent update
            update_weights(data[i], labels[i]);
        }
    }
    
    void aggregate_weights(float* other_weights) {
        // Average weights with other devices
        for (int i = 0; i < MAX_WEIGHTS; i++) {
            global_weights[i] = (local_weights[i] + other_weights[i]) / 2.0f;
        }
    }
};
```

### **Challenge 4: Real-time Anomaly Detection**

Create a system that detects anomalies in real-time sensor data:

```python
def create_anomaly_detector():
    # Autoencoder for anomaly detection
    encoder = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu')
    ])
    
    decoder = keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Combine encoder and decoder
    input_data = layers.Input(shape=(input_dim,))
    encoded = encoder(input_data)
    decoded = decoder(encoded)
    
    return keras.Model(input_data, decoded)
```

### **Challenge 5: Energy-Efficient Inference**

Optimize your model for minimal power consumption:

```cpp
// Power management for ESP32
class PowerManager {
private:
    int inference_count = 0;
    float battery_level = 100.0f;
    
public:
    void enter_deep_sleep() {
        // Enter deep sleep mode to save power
        esp_deep_sleep_start();
    }
    
    void adaptive_inference_frequency() {
        // Adjust inference frequency based on battery level
        if (battery_level < 20.0f) {
            // Reduce inference frequency
            vTaskDelay(pdMS_TO_TICKS(10000));  // 10 seconds
        } else {
            // Normal frequency
            vTaskDelay(pdMS_TO_TICKS(1000));   // 1 second
        }
    }
};
```

---

## 🚀 **Self-Assessment Checkpoints**

### **Checkpoint 1: Keras 3.0 Fundamentals**
- [ ] I understand multi-backend architecture
- [ ] I can choose appropriate backends for different tasks
- [ ] I can build models for resource-constrained devices
- [ ] I understand model size and memory considerations

### **Checkpoint 2: Model Optimization**
- [ ] I can apply quantization techniques
- [ ] I understand pruning and its effects
- [ ] I can use quantization-aware training
- [ ] I know how to balance accuracy vs. size

### **Checkpoint 3: TensorFlow Lite**
- [ ] I can convert Keras models to TFLite
- [ ] I understand TFLite Micro limitations
- [ ] I can optimize models for edge deployment
- [ ] I know how to handle custom operations

### **Checkpoint 4: ESP32 Development**
- [ ] I can set up ESP-IDF development environment
- [ ] I understand ESP32 memory constraints
- [ ] I can implement real-time inference
- [ ] I know how to handle power management

### **Checkpoint 5: Real-World Applications**
- [ ] I can build audio classification models
- [ ] I understand sensor data processing
- [ ] I can implement anomaly detection
- [ ] I know how to optimize for production

---

## 🎯 **Real-World Project Ideas**

### **Beginner Projects**
1. **Voice Command Recognition** - "Hey ESP32" detection
2. **Gesture Recognition** - Hand gesture classification
3. **Environmental Monitoring** - Temperature/humidity anomaly detection
4. **Activity Recognition** - Walking, running, sitting detection

### **Intermediate Projects**
1. **Smart Home Controller** - Voice + gesture control
2. **Industrial Monitoring** - Equipment health monitoring
3. **Wearable Health Monitor** - Heart rate, activity tracking
4. **Agricultural IoT** - Soil moisture, crop health monitoring

### **Advanced Projects**
1. **Autonomous Robot** - Obstacle detection and navigation
2. **Smart Manufacturing** - Quality control and predictive maintenance
3. **Medical Device** - Patient monitoring and alerting
4. **Edge AI Network** - Multiple devices collaborating

---

*Build amazing edge AI applications!* 🎯 