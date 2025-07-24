# TinyML Advanced Tutorial: Part 2 - Deep Dive into Edge AI

## 🚀 Advanced Optimization Techniques

### Understanding Advanced TinyML Optimization

This tutorial covers **advanced optimization techniques** that push the boundaries of what's possible with TinyML. These techniques are essential for deploying sophisticated AI models on the most resource-constrained devices.

#### **Why Advanced Optimization Matters:**

**Resource Constraints:**
- **Memory**: Microcontrollers often have < 1MB RAM
- **Storage**: Flash memory limited to < 4MB
- **Power**: Battery life requirements for IoT devices
- **Latency**: Real-time response requirements

**Performance Requirements:**
- **Accuracy**: Maintain high accuracy despite optimization
- **Speed**: Fast inference for real-time applications
- **Efficiency**: Minimal power consumption
- **Reliability**: Stable performance across conditions

#### **Advanced Techniques Overview:**

1. **Neural Architecture Search (NAS)**: Automatically find optimal model architectures
2. **Advanced Quantization**: Mixed-precision and per-channel quantization
3. **Structured Pruning**: Remove entire neurons/filters for efficiency
4. **Progressive Quantization**: Gradual precision reduction
5. **Memory Optimization**: Advanced memory management techniques
6. **Multi-task Learning**: Single model for multiple tasks
7. **Real-time Sensor Fusion**: Combine multiple sensor inputs
8. **Performance Monitoring**: Real-time performance tracking

#### **When to Use Advanced Techniques:**

**Use Advanced Optimization When:**
- Standard quantization isn't sufficient
- You need maximum performance on limited hardware
- Accuracy requirements are very high
- Power consumption is critical
- You're targeting the most constrained devices

**Considerations:**
- **Development Time**: Advanced techniques require more time
- **Complexity**: More complex to implement and debug
- **Maintenance**: Harder to maintain and update
- **Trade-offs**: Balance between performance and complexity

### Neural Architecture Search (NAS) for TinyML

#### **Understanding Neural Architecture Search**

**Neural Architecture Search (NAS)** is an automated approach to finding optimal neural network architectures. For TinyML, NAS is particularly valuable because:

**Why NAS for TinyML:**
- **Automated Design**: Finds architectures optimized for specific constraints
- **Constraint-Aware**: Can optimize for memory, latency, and power
- **Performance**: Often finds better architectures than manual design
- **Efficiency**: Reduces design time and improves results

**NAS Components:**
1. **Search Space**: Define possible architectures
2. **Search Strategy**: How to explore the search space
3. **Evaluation**: How to measure architecture quality
4. **Constraints**: Memory, latency, power requirements

#### **TinyML-Specific NAS Considerations:**

**Search Space Design:**
- **Layer Types**: Conv2D, DepthwiseConv2D, Dense
- **Layer Parameters**: Filters, kernel size, expansion ratios
- **Connectivity**: Skip connections, branching
- **Activation Functions**: ReLU, ReLU6, Swish

**Evaluation Metrics:**
- **Model Size**: Parameters and memory footprint
- **Inference Speed**: Latency on target hardware
- **Power Consumption**: Energy per inference
- **Accuracy**: Task-specific performance

```python
import keras
from keras import layers
import numpy as np

class TinyMLNAS:
    """
    Neural Architecture Search for TinyML
    
    This class implements a simplified NAS approach specifically designed
    for TinyML applications. It searches for architectures that are:
    - Memory efficient (small parameter count)
    - Fast (low latency)
    - Accurate (high performance)
    - Power efficient (low energy consumption)
    
    The search space is constrained to operations that work well on
    microcontrollers and edge devices.
    
    Parameters:
    - input_shape: Shape of input data (e.g., (32, 32, 3) for images)
    - num_classes: Number of output classes
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_mobile_block(self, x, filters, expansion_ratio=6):
        """MobileNet-style block with searchable parameters"""
        expanded_filters = int(x.shape[-1] * expansion_ratio)
        
        # Pointwise expansion
        x = layers.Conv2D(expanded_filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)  # ReLU6 for quantization
        
        # Depthwise convolution
        x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        
        # Pointwise linear
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        return x
    
    def search_architecture(self, search_space):
        """Search for optimal architecture"""
        best_model = None
        best_score = float('inf')
        
        for config in search_space:
            model = self.build_model(config)
            score = self.evaluate_model(model)
            
            if score < best_score:
                best_score = score
                best_model = model
                
        return best_model
    
    def build_model(self, config):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        
        # Add searchable blocks
        for block_config in config['blocks']:
            x = self.create_mobile_block(x, **block_config)
            
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return keras.Model(inputs, outputs)
```

### Advanced Quantization Techniques

#### Mixed-Precision Quantization

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def mixed_precision_quantization(model, x_train):
    """Apply different quantization to different layers"""
    
    # Define quantization configs
    quantize_config = tfmot.quantization.keras.QuantizeConfig(
        weight_quantizer=tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, per_axis=True, symmetric=True
        ),
        activation_quantizer=tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False
        )
    )
    
    # Apply quantization selectively
    quantized_model = tfmot.quantization.keras.quantize_apply(
        model, 
        quantize_config,
        layer_names=['dense_1', 'dense_2']  # Only quantize specific layers
    )
    
    return quantized_model

def dynamic_range_quantization(model):
    """Dynamic range quantization for better accuracy"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Enable experimental features
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    
    return converter.convert()
```

#### Per-Channel Quantization

```python
def per_channel_quantization(model, x_train):
    """Per-channel quantization for better accuracy"""
    
    def representative_dataset():
        for data in x_train[:100]:
            yield [data.astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Per-channel quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16
    ]
    
    return converter.convert()
```

### Advanced Pruning Techniques

#### Structured Pruning

```python
def structured_pruning(model, sparsity_pattern):
    """Apply structured pruning patterns"""
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.8,
            begin_step=0,
            end_step=1000
        ),
        'block_size': (1, 4),  # Prune in 1x4 blocks
        'block_pooling_type': 'AVG'
    }
    
    # Apply to specific layers
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            layer = tfmot.sparsity.keras.prune_low_magnitude(
                layer, **pruning_params
            )
    
    return model

def magnitude_pruning_with_momentum(model, momentum=0.9):
    """Pruning with momentum for better convergence"""
    
    class MomentumPruningSchedule(tfmot.sparsity.keras.PruningSchedule):
        def __init__(self, initial_sparsity, final_sparsity, momentum=0.9):
            self.initial_sparsity = initial_sparsity
            self.final_sparsity = final_sparsity
            self.momentum = momentum
            self.current_sparsity = initial_sparsity
            
        def __call__(self, step):
            # Update with momentum
            target_sparsity = self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (step / 1000)
            
            self.current_sparsity = (
                self.momentum * self.current_sparsity + 
                (1 - self.momentum) * target_sparsity
            )
            
            return self.current_sparsity
    
    pruning_params = {
        'pruning_schedule': MomentumPruningSchedule(0.0, 0.8, momentum)
    }
    
    return tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

## 🎯 Microcontroller-Specific Models

### 1. Keyword Spotting Model

```python
def create_keyword_spotting_model(input_shape, num_keywords):
    """Lightweight keyword spotting for voice commands"""
    
    inputs = keras.Input(shape=input_shape)  # (time_steps, features)
    
    # 1D convolutions for audio features
    x = layers.Conv1D(64, 10, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Dropout(0.2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_keywords, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Training with audio data
def train_keyword_model(model, audio_data, labels):
    """Train keyword spotting model"""
    
    # Data augmentation
    augmented_data = []
    for audio in audio_data:
        # Add noise
        noise = np.random.normal(0, 0.01, audio.shape)
        augmented_data.append(audio + noise)
        
        # Time shift
        shift = np.random.randint(-5, 6)
        shifted = np.roll(audio, shift, axis=0)
        augmented_data.append(shifted)
    
    # Combine original and augmented data
    all_data = np.concatenate([audio_data, augmented_data])
    all_labels = np.concatenate([labels, labels])
    
    # Train model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model.fit(all_data, all_labels, epochs=50, validation_split=0.2)
```

### 2. Gesture Recognition Model

```python
def create_gesture_model(input_shape, num_gestures):
    """Gesture recognition with accelerometer/gyroscope data"""
    
    inputs = keras.Input(shape=input_shape)  # (time_steps, 6) for accel+gyro
    
    # LSTM for temporal patterns
    x = layers.LSTM(32, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(16)(x)
    x = layers.Dropout(0.2)(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu')(x)
    
    outputs = layers.Dense(num_gestures, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Data preprocessing for gestures
def preprocess_gesture_data(raw_data):
    """Preprocess accelerometer/gyroscope data"""
    
    # Normalize data
    normalized_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
    
    # Apply low-pass filter
    from scipy import signal
    b, a = signal.butter(3, 0.1, 'low')
    filtered_data = signal.filtfilt(b, a, normalized_data, axis=0)
    
    # Segment into windows
    window_size = 50  # 50 samples per gesture
    windows = []
    
    for i in range(0, len(filtered_data) - window_size, window_size // 2):
        window = filtered_data[i:i + window_size]
        windows.append(window)
    
    return np.array(windows)
```

### 3. Anomaly Detection Model

```python
def create_anomaly_detector(input_shape, encoding_dim=8):
    """Autoencoder for anomaly detection"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(16, activation='relu')(encoded)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(x)
    
    # Create models
    autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)
    
    return autoencoder, encoder

def detect_anomalies(autoencoder, data, threshold=0.1):
    """Detect anomalies using reconstruction error"""
    
    # Get reconstructions
    reconstructions = autoencoder.predict(data)
    
    # Calculate reconstruction error
    mse = np.mean(np.square(data - reconstructions), axis=1)
    
    # Detect anomalies
    anomalies = mse > threshold
    
    return anomalies, mse
```

### 4. Image Classification for Microcontrollers

```python
def create_tiny_image_classifier(input_shape, num_classes):
    """Ultra-lightweight image classifier"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Very small CNN
    x = layers.Conv2D(8, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(16, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

## 🔧 Advanced ESP32 Techniques

### Memory-Optimized Inference

```cpp
// Advanced memory management for ESP32
#include "esp_heap_caps.h"
#include "esp_log.h"

class OptimizedTFLiteInterpreter {
private:
    static constexpr int kTensorArenaSize = 30 * 1024;  // 30KB
    uint8_t* tensor_arena;
    tflite::MicroInterpreter* interpreter;
    const tflite::Model* model;
    
public:
    OptimizedTFLiteInterpreter() {
        // Allocate from PSRAM if available
        #ifdef CONFIG_SPIRAM_SUPPORT
            tensor_arena = (uint8_t*)heap_caps_malloc(
                kTensorArenaSize, 
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
            );
        #else
            tensor_arena = (uint8_t*)heap_caps_malloc(
                kTensorArenaSize, 
                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
            );
        #endif
        
        if (!tensor_arena) {
            ESP_LOGE("TFLite", "Failed to allocate tensor arena");
        }
    }
    
    ~OptimizedTFLiteInterpreter() {
        if (tensor_arena) {
            heap_caps_free(tensor_arena);
        }
    }
    
    TfLiteStatus Initialize(const uint8_t* model_data) {
        model = tflite::GetModel(model_data);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE("TFLite", "Model schema mismatch");
            return kTfLiteError;
        }
        
        static tflite::AllOpsResolver resolver;
        static tflite::MicroInterpreter static_interpreter(
            model, resolver, tensor_arena, kTensorArenaSize, nullptr);
        
        interpreter = &static_interpreter;
        
        TfLiteStatus status = interpreter->AllocateTensors();
        if (status != kTfLiteOk) {
            ESP_LOGE("TFLite", "AllocateTensors() failed");
            return status;
        }
        
        return kTfLiteOk;
    }
    
    TfLiteStatus Invoke() {
        return interpreter->Invoke();
    }
    
    TfLiteTensor* input(int index) {
        return interpreter->input(index);
    }
    
    TfLiteTensor* output(int index) {
        return interpreter->output(index);
    }
};
```

### Multi-Task Learning on ESP32

```cpp
// Multi-task model for ESP32
class MultiTaskTFLiteModel {
private:
    OptimizedTFLiteInterpreter interpreter;
    
public:
    struct TaskOutputs {
        float gesture_prediction[5];      // 5 gesture classes
        float anomaly_score;              // Anomaly detection score
        float keyword_prediction[3];      // 3 keyword classes
    };
    
    TfLiteStatus Initialize(const uint8_t* model_data) {
        return interpreter.Initialize(model_data);
    }
    
    TfLiteStatus RunInference(const float* input_data, TaskOutputs* outputs) {
        // Set input
        TfLiteTensor* input = interpreter.input(0);
        memcpy(input->data.f, input_data, input->bytes);
        
        // Run inference
        TfLiteStatus status = interpreter.Invoke();
        if (status != kTfLiteOk) {
            return status;
        }
        
        // Get outputs
        TfLiteTensor* gesture_output = interpreter.output(0);
        TfLiteTensor* anomaly_output = interpreter.output(1);
        TfLiteTensor* keyword_output = interpreter.output(2);
        
        // Copy results
        memcpy(outputs->gesture_prediction, gesture_output->data.f, 
               sizeof(outputs->gesture_prediction));
        outputs->anomaly_score = anomaly_output->data.f[0];
        memcpy(outputs->keyword_prediction, keyword_output->data.f,
               sizeof(outputs->keyword_prediction));
        
        return kTfLiteOk;
    }
};
```

### Real-Time Sensor Fusion

```cpp
// Sensor fusion for multiple sensors
#include "driver/i2c.h"
#include "driver/spi_master.h"

class SensorFusion {
private:
    // I2C configuration
    i2c_config_t i2c_config;
    spi_device_handle_t spi_handle;
    
    // Sensor data buffers
    float accelerometer_data[3];
    float gyroscope_data[3];
    float magnetometer_data[3];
    
public:
    esp_err_t Initialize() {
        // Configure I2C
        i2c_config.mode = I2C_MODE_MASTER;
        i2c_config.sda_io_num = GPIO_NUM_21;
        i2c_config.scl_io_num = GPIO_NUM_22;
        i2c_config.sda_pullup_en = GPIO_PULLUP_ENABLE;
        i2c_config.scl_pullup_en = GPIO_PULLUP_ENABLE;
        i2c_config.master.clk_speed = 400000;
        
        esp_err_t ret = i2c_param_config(I2C_NUM_0, &i2c_config);
        if (ret != ESP_OK) return ret;
        
        ret = i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0);
        if (ret != ESP_OK) return ret;
        
        // Initialize sensors
        InitializeAccelerometer();
        InitializeGyroscope();
        InitializeMagnetometer();
        
        return ESP_OK;
    }
    
    void ReadSensors() {
        ReadAccelerometer(accelerometer_data);
        ReadGyroscope(gyroscope_data);
        ReadMagnetometer(magnetometer_data);
    }
    
    void FuseData(float* fused_data) {
        // Simple sensor fusion (can be enhanced with Kalman filter)
        for (int i = 0; i < 3; i++) {
            // Weighted average of sensors
            fused_data[i] = 0.4f * accelerometer_data[i] + 
                           0.3f * gyroscope_data[i] + 
                           0.3f * magnetometer_data[i];
        }
    }
    
private:
    void InitializeAccelerometer() {
        // Initialize MPU6050 or similar
        uint8_t data = 0x00;
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, 0x68 << 1 | I2C_MASTER_WRITE, true);
        i2c_master_write_byte(cmd, 0x6B, true);  // PWR_MGMT_1 register
        i2c_master_write_byte(cmd, data, true);
        i2c_master_stop(cmd);
        i2c_master_cmd_begin(I2C_NUM_0, cmd, 1000 / portTICK_PERIOD_MS);
        i2c_cmd_link_delete(cmd);
    }
    
    void ReadAccelerometer(float* data) {
        // Read accelerometer data
        uint8_t buffer[6];
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, 0x68 << 1 | I2C_MASTER_WRITE, true);
        i2c_master_write_byte(cmd, 0x3B, true);  // ACCEL_XOUT_H register
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, 0x68 << 1 | I2C_MASTER_READ, true);
        i2c_master_read(cmd, buffer, 6, I2C_MASTER_LAST_NACK);
        i2c_master_stop(cmd);
        i2c_master_cmd_begin(I2C_NUM_0, cmd, 1000 / portTICK_PERIOD_MS);
        i2c_cmd_link_delete(cmd);
        
        // Convert to float
        for (int i = 0; i < 3; i++) {
            int16_t raw = (buffer[i*2] << 8) | buffer[i*2+1];
            data[i] = raw / 16384.0f;  // Convert to g
        }
    }
    
    void InitializeGyroscope() {
        // Similar to accelerometer initialization
    }
    
    void ReadGyroscope(float* data) {
        // Similar to accelerometer reading
    }
    
    void InitializeMagnetometer() {
        // Initialize magnetometer
    }
    
    void ReadMagnetometer(float* data) {
        // Read magnetometer data
    }
};
```

## 🎯 Advanced Optimization Strategies

### Model Distillation with Temperature Scaling

```python
def temperature_scaled_distillation(teacher_model, student_model, 
                                   x_train, y_train, temperature=3.0):
    """Advanced knowledge distillation with temperature scaling"""
    
    def distillation_loss(y_true, y_pred):
        # Get teacher predictions
        teacher_pred = teacher_model.predict(x_train, verbose=0)
        
        # Apply temperature scaling
        teacher_pred_scaled = teacher_pred / temperature
        student_pred_scaled = y_pred / temperature
        
        # Soft targets loss
        soft_loss = keras.losses.categorical_crossentropy(
            teacher_pred_scaled, student_pred_scaled
        )
        
        # Hard targets loss
        hard_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Combined loss with temperature scaling
        return 0.7 * (temperature ** 2) * soft_loss + 0.3 * hard_loss
    
    # Compile student model
    student_model.compile(
        optimizer='adam',
        loss=distillation_loss,
        metrics=['accuracy']
    )
    
    # Train with distillation
    return student_model.fit(x_train, y_train, epochs=20)
```

### Progressive Quantization

```python
def progressive_quantization(model, x_train, quantization_steps=[16, 8, 4]):
    """Progressive quantization for better accuracy"""
    
    current_model = model
    
    for bits in quantization_steps:
        print(f"Quantizing to {bits} bits...")
        
        # Create quantized model
        quantize_model = tfmot.quantization.keras.quantize_model
        quantized_model = quantize_model(current_model)
        
        # Fine-tune quantized model
        quantized_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        quantized_model.fit(x_train, y_train, epochs=5, verbose=0)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set quantization parameters
        if bits == 4:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16
            ]
        
        tflite_model = converter.convert()
        
        # Evaluate and keep if good
        accuracy = evaluate_tflite_model(tflite_model, x_test, y_test)
        print(f"Accuracy with {bits} bits: {accuracy:.4f}")
        
        current_model = quantized_model
    
    return current_model
```

### Neural Architecture Optimization

```python
def optimize_architecture_for_memory(model, target_memory_kb=50):
    """Optimize architecture for memory constraints"""
    
    def count_parameters(model):
        return sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers])
    
    def estimate_memory(model):
        # Rough estimation: 4 bytes per parameter for float32
        params = count_parameters(model)
        return params * 4 / 1024  # KB
    
    current_memory = estimate_memory(model)
    
    while current_memory > target_memory_kb:
        # Reduce model size
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                current_units = layer.units
                if current_units > 16:
                    layer.units = max(16, current_units // 2)
                    break
            elif isinstance(layer, keras.layers.Conv2D):
                current_filters = layer.filters
                if current_filters > 8:
                    layer.filters = max(8, current_filters // 2)
                    break
        
        # Rebuild model
        model = keras.models.clone_model(model)
        current_memory = estimate_memory(model)
    
    return model
```

## 🔧 Best Practices for Production

### Model Versioning and Deployment

```python
import json
import hashlib
from datetime import datetime

class ModelVersioning:
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model_version(self, model, metadata):
        """Save model with versioning"""
        
        # Generate version hash
        model_config = model.get_config()
        config_str = json.dumps(model_config, sort_keys=True)
        version_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create version directory
        version_dir = os.path.join(self.model_dir, f"v{version_hash}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, "model.h5")
        model.save(model_path)
        
        # Save metadata
        metadata['version'] = version_hash
        metadata['created_at'] = datetime.now().isoformat()
        metadata['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
        
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(version_dir, "model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        return version_hash
    
    def load_model_version(self, version_hash):
        """Load specific model version"""
        
        version_dir = os.path.join(self.model_dir, f"v{version_hash}")
        model_path = os.path.join(version_dir, "model.h5")
        metadata_path = os.path.join(version_dir, "metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model version {version_hash} not found")
        
        model = keras.models.load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
```

### Automated Testing Pipeline

```python
class TinyMLTestSuite:
    def __init__(self, model, test_data, test_labels):
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
    
    def run_comprehensive_tests(self):
        """Run comprehensive model tests"""
        
        results = {}
        
        # Accuracy test
        results['accuracy'] = self.test_accuracy()
        
        # Memory usage test
        results['memory_usage'] = self.test_memory_usage()
        
        # Inference speed test
        results['inference_speed'] = self.test_inference_speed()
        
        # Robustness test
        results['robustness'] = self.test_robustness()
        
        # Edge case test
        results['edge_cases'] = self.test_edge_cases()
        
        return results
    
    def test_accuracy(self):
        """Test model accuracy"""
        predictions = self.model.predict(self.test_data)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(self.test_labels, axis=1))
        return accuracy
    
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Run inference
        self.model.predict(self.test_data[:100])
        
        memory_after = process.memory_info().rss
        memory_used = (memory_after - memory_before) / 1024  # KB
        
        return memory_used
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        # Warm up
        for _ in range(10):
            self.model.predict(self.test_data[:1])
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            self.model.predict(self.test_data[:1])
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        return avg_time
    
    def test_robustness(self):
        """Test model robustness to noise"""
        # Add noise to test data
        noisy_data = self.test_data + np.random.normal(0, 0.1, self.test_data.shape)
        
        predictions = self.model.predict(noisy_data)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(self.test_labels, axis=1))
        
        return accuracy
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with zero input
        zero_input = np.zeros_like(self.test_data[:1])
        zero_pred = self.model.predict(zero_input)
        
        # Test with extreme values
        extreme_input = np.ones_like(self.test_data[:1]) * 1000
        extreme_pred = self.model.predict(extreme_input)
        
        return {
            'zero_input_prediction': zero_pred[0].tolist(),
            'extreme_input_prediction': extreme_pred[0].tolist()
        }
```

### Performance Monitoring

```cpp
// Performance monitoring for ESP32
class PerformanceMonitor {
private:
    uint64_t inference_times[100];
    int time_index = 0;
    uint64_t total_inferences = 0;
    uint64_t total_time = 0;
    
public:
    void RecordInference(uint64_t start_time, uint64_t end_time) {
        uint64_t duration = end_time - start_time;
        
        inference_times[time_index] = duration;
        time_index = (time_index + 1) % 100;
        
        total_inferences++;
        total_time += duration;
    }
    
    float GetAverageInferenceTime() {
        if (total_inferences == 0) return 0.0f;
        return (float)total_time / total_inferences / 1000.0f;  // Convert to ms
    }
    
    float GetPercentileInferenceTime(float percentile) {
        if (total_inferences == 0) return 0.0f;
        
        // Sort times for percentile calculation
        uint64_t sorted_times[100];
        memcpy(sorted_times, inference_times, sizeof(inference_times));
        
        // Simple bubble sort (for small arrays)
        for (int i = 0; i < 99; i++) {
            for (int j = 0; j < 99 - i; j++) {
                if (sorted_times[j] > sorted_times[j + 1]) {
                    uint64_t temp = sorted_times[j];
                    sorted_times[j] = sorted_times[j + 1];
                    sorted_times[j + 1] = temp;
                }
            }
        }
        
        int index = (int)(percentile * 100 / 100.0f);
        return (float)sorted_times[index] / 1000.0f;  // Convert to ms
    }
    
    void PrintStats() {
        float avg_time = GetAverageInferenceTime();
        float p95_time = GetPercentileInferenceTime(95.0f);
        float p99_time = GetPercentileInferenceTime(99.0f);
        
        ESP_LOGI("Performance", "Inference Stats:");
        ESP_LOGI("Performance", "  Average: %.2f ms", avg_time);
        ESP_LOGI("Performance", "  95th percentile: %.2f ms", p95_time);
        ESP_LOGI("Performance", "  99th percentile: %.2f ms", p99_time);
        ESP_LOGI("Performance", "  Total inferences: %llu", total_inferences);
    }
};
```

## 🎯 Advanced Applications

### 1. Multi-Modal Fusion

```python
def create_multimodal_model(audio_shape, sensor_shape, num_classes):
    """Multi-modal model combining audio and sensor data"""
    
    # Audio branch
    audio_input = keras.Input(shape=audio_shape)
    audio_x = layers.Conv1D(32, 3, activation='relu')(audio_input)
    audio_x = layers.GlobalAveragePooling1D()(audio_x)
    
    # Sensor branch
    sensor_input = keras.Input(shape=sensor_shape)
    sensor_x = layers.LSTM(16)(sensor_input)
    
    # Fusion
    fused = layers.Concatenate()([audio_x, sensor_x])
    x = layers.Dense(64, activation='relu')(fused)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model([audio_input, sensor_input], outputs)
```

### 2. Continual Learning

```python
class ContinualLearner:
    def __init__(self, base_model, memory_size=1000):
        self.base_model = base_model
        self.memory_buffer = []
        self.memory_size = memory_size
    
    def update_model(self, new_data, new_labels):
        """Update model with new data while preserving old knowledge"""
        
        # Add new data to memory
        for data, label in zip(new_data, new_labels):
            self.memory_buffer.append((data, label))
            
            # Maintain memory size
            if len(self.memory_buffer) > self.memory_size:
                self.memory_buffer.pop(0)
        
        # Create training data from memory
        memory_data = np.array([item[0] for item in self.memory_buffer])
        memory_labels = np.array([item[1] for item in self.memory_buffer])
        
        # Fine-tune model
        self.base_model.fit(
            memory_data, memory_labels,
            epochs=5,
            validation_split=0.2
        )
    
    def get_model(self):
        return self.base_model
```

### 3. Federated Learning for Edge Devices

```python
class FederatedLearner:
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_models = []
    
    def train_on_client(self, client_id, client_data, client_labels):
        """Train model on client device"""
        
        # Create client model copy
        client_model = keras.models.clone_model(self.global_model)
        client_model.set_weights(self.global_model.get_weights())
        
        # Train on client data
        client_model.fit(client_data, client_labels, epochs=10)
        
        # Store client model
        self.client_models.append({
            'id': client_id,
            'model': client_model,
            'data_size': len(client_data)
        })
    
    def aggregate_models(self):
        """Aggregate client models using FedAvg"""
        
        if not self.client_models:
            return
        
        # Calculate total data size
        total_data_size = sum(client['data_size'] for client in self.client_models)
        
        # Initialize aggregated weights
        aggregated_weights = []
        for layer in self.global_model.layers:
            if layer.weights:
                aggregated_weights.append(np.zeros_like(layer.get_weights()[0]))
        
        # Weighted average of client models
        for client in self.client_models:
            client_weights = client['model'].get_weights()
            weight = client['data_size'] / total_data_size
            
            for i, layer_weights in enumerate(client_weights):
                aggregated_weights[i] += weight * layer_weights
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Clear client models
        self.client_models = []
```

## 🎉 Conclusion

This advanced tutorial covers:

✅ **Advanced optimization techniques** - NAS, mixed-precision, progressive quantization  
✅ **Microcontroller-specific models** - Keyword spotting, gesture recognition, anomaly detection  
✅ **Advanced ESP32 techniques** - Memory optimization, multi-task learning, sensor fusion  
✅ **Production best practices** - Model versioning, testing, performance monitoring  
✅ **Advanced applications** - Multi-modal fusion, continual learning, federated learning  

### Key Takeaways:

1. **Start simple** - Begin with basic models and gradually add complexity
2. **Profile everything** - Monitor memory, speed, and accuracy continuously
3. **Test thoroughly** - Use comprehensive testing suites for production
4. **Optimize iteratively** - Apply optimizations one at a time and measure impact
5. **Consider the full pipeline** - From data collection to deployment

### Next Steps:

1. **Implement sensor fusion** - Combine multiple sensors for better accuracy
2. **Add continual learning** - Enable models to adapt to new data
3. **Explore federated learning** - Train across multiple edge devices
4. **Build production pipelines** - Automated testing and deployment

**Happy advanced TinyML development!** 🚀

---

*Build the future of edge AI!* 🎯 