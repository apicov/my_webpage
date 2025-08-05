# Information Theory and Compression for Neural Networks: A Comprehensive Keras 3 Course

## Course Overview

Welcome to this university-level course on information theory and compression applied to neural networks and transformers. This self-contained tutorial will take you from classical information theory concepts to cutting-edge compression techniques, all implemented in Keras 3. Whether you're deploying models to mobile devices or understanding the theoretical foundations of transformers, this course provides both the mathematical rigor and practical skills you need.

### Prerequisites
- Working knowledge of neural networks
- Basic Python programming
- Rusty on math? No problem - we'll refresh calculus and linear algebra as needed

### Learning Outcomes
By the end of this course, you will:
1. Master fundamental information theory concepts and their neural network applications
2. Understand attention mechanisms through an information-theoretic lens
3. Implement state-of-the-art model compression techniques
4. Analyze transformer capacity using theoretical frameworks
5. Deploy optimized models using Keras 3 across multiple backends

---

## Module 1: Classical Information Theory Foundations

### 1.1 Information and Surprise: The Building Blocks

Let's start with an intuitive question: **What is information?**

Imagine you're waiting for a friend who's always punctual. If they arrive on time, you gain little information - it's expected. But if they're an hour late, that's surprising and informative! This intuition captures the essence of information theory.

#### Mathematical Foundation: Self-Information

The information content of an event with probability $p$ is:

$$I(x) = -\log_2(p(x)) = \log_2\left(\frac{1}{p(x)}\right) \text{ bits}$$

**Math Refresher - Logarithms:**
- $\log_2(x)$ asks: "2 to what power equals x?"
- Key properties: $\log(ab) = \log(a) + \log(b)$, $\log(a^n) = n\log(a)$
- We use log base 2 to measure information in bits

Let's implement this in Keras 3:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"
import keras
from keras import ops
import numpy as np
import matplotlib.pyplot as plt

def self_information(prob):
    """Calculate self-information in bits"""
    return -ops.log2(prob)

# Visualize the relationship
probs = np.linspace(0.01, 1.0, 100)
info = [-np.log2(p) for p in probs]

plt.figure(figsize=(10, 6))
plt.plot(probs, info, linewidth=2)
plt.xlabel('Probability of Event', fontsize=12)
plt.ylabel('Information Content (bits)', fontsize=12)
plt.title('Information Content vs. Probability: Rare Events Carry More Information', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

**Key Insight**: Rare events (low probability) carry more information than common events. This principle underlies everything from data compression to neural network learning.

### 1.2 Entropy: Average Information Content

While self-information tells us about individual events, **entropy** measures the average information content of a random variable.

#### Shannon Entropy

For a discrete random variable $X$ with probability distribution $p(x)$:

$$H(X) = -\sum_{x} p(x) \log_2(p(x)) = \mathbb{E}[\log_2(1/p(x))]$$

**Intuitive Understanding**: Entropy measures uncertainty or "average surprise". A fair coin has maximum entropy (1 bit), while a biased coin has lower entropy.

#### Keras 3 Implementation: Custom Entropy Layer

```python
class EntropyLayer(keras.layers.Layer):
    """Compute entropy of probability distributions"""
    
    def __init__(self, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def call(self, inputs):
        # Ensure valid probabilities
        probs = ops.clip(inputs, self.epsilon, 1.0 - self.epsilon)
        # Normalize if needed
        probs = probs / ops.sum(probs, axis=-1, keepdims=True)
        # Compute entropy
        entropy = -ops.sum(probs * ops.log2(probs), axis=-1)
        return entropy

# Example: Analyzing entropy in neural network outputs
def demonstrate_entropy_in_training():
    # Create a simple model
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),  # Raw logits
        keras.layers.Softmax(),
        EntropyLayer(name='entropy')
    ])
    
    # Generate sample data
    x = np.random.randn(1000, 20)
    
    # Get entropy of predictions
    entropy_values = model(x).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(entropy_values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Entropy (bits)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Entropy in Untrained Network', fontsize=14)
    plt.axvline(np.log2(10), color='red', linestyle='--', 
                label=f'Maximum possible entropy: {np.log2(10):.2f} bits')
    plt.legend()
    plt.show()
    
    return model, entropy_values

model, initial_entropy = demonstrate_entropy_in_training()
```

### 1.3 Mutual Information: Shared Knowledge

**Mutual information** $I(X;Y)$ measures how much knowing one variable tells us about another.

#### Mathematical Definition

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

Alternative formulations:
- $I(X;Y) = H(X) - H(X|Y)$ (reduction in uncertainty about X given Y)
- $I(X;Y) = \sum_{x,y} p(x,y) \log_2\frac{p(x,y)}{p(x)p(y)}$

**Visual Intuition**: Think of information as overlapping circles in a Venn diagram:

```python
def visualize_mutual_information():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Venn diagram representation
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    
    # Independent variables (no mutual information)
    circle1 = Circle((0.3, 0.5), 0.25, alpha=0.5, color='blue')
    circle2 = Circle((0.7, 0.5), 0.25, alpha=0.5, color='red')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Independent Variables\nI(X;Y) = 0', fontsize=14)
    ax1.text(0.3, 0.5, 'H(X)', ha='center', va='center', fontsize=12)
    ax1.text(0.7, 0.5, 'H(Y)', ha='center', va='center', fontsize=12)
    
    # Dependent variables (high mutual information)
    circle3 = Circle((0.4, 0.5), 0.25, alpha=0.5, color='blue')
    circle4 = Circle((0.6, 0.5), 0.25, alpha=0.5, color='red')
    ax2.add_patch(circle3)
    ax2.add_patch(circle4)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Dependent Variables\nI(X;Y) > 0', fontsize=14)
    ax2.text(0.3, 0.5, 'H(X|Y)', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 0.5, 'I(X;Y)', ha='center', va='center', fontsize=12, weight='bold')
    ax2.text(0.7, 0.5, 'H(Y|X)', ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

visualize_mutual_information()
```

### 1.4 KL Divergence: Measuring Distribution Differences

**Kullback-Leibler (KL) divergence** measures how one probability distribution differs from another.

#### Mathematical Definition

$$D_{KL}(P||Q) = \sum_x p(x) \log_2\frac{p(x)}{q(x)} = \mathbb{E}_P\left[\log_2\frac{P(x)}{Q(x)}\right]$$

**Key Properties**:
- Always non-negative: $D_{KL}(P||Q) \geq 0$
- Zero if and only if $P = Q$
- **Not symmetric**: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

**Intuition**: KL divergence measures the "extra bits" needed when using distribution Q to encode data from distribution P.

#### Cross-Entropy Connection

$$H(P, Q) = H(P) + D_{KL}(P||Q)$$

This explains why we minimize cross-entropy in classification - it's equivalent to minimizing KL divergence!

```python
class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    """Regularize layer outputs to match a target distribution"""
    
    def __init__(self, target_distribution, weight=1.0):
        self.target_distribution = ops.convert_to_tensor(target_distribution)
        self.weight = weight
    
    def __call__(self, x):
        # Ensure valid probabilities
        x = ops.nn.softmax(x)
        target = self.target_distribution
        
        # Compute KL divergence
        kl_div = ops.sum(x * ops.log(x / (target + 1e-8) + 1e-8), axis=-1)
        return self.weight * ops.mean(kl_div)
```

### 1.5 The Information Bottleneck Principle

The **Information Bottleneck (IB)** principle provides a framework for understanding how neural networks learn to compress information.

#### Mathematical Framework

Given input X and target Y, find representation T that:
- Maximizes $I(T;Y)$ (preserves relevant information)
- Minimizes $I(X;T)$ (compresses input)

Objective: $\max_T [I(T;Y) - \beta I(X;T)]$

```python
class InformationBottleneckLayer(keras.layers.Layer):
    """Layer implementing the Information Bottleneck principle"""
    
    def __init__(self, encoding_dim, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.beta = beta
        
    def build(self, input_shape):
        # Encoder
        self.encoder_mean = keras.layers.Dense(self.encoding_dim)
        self.encoder_logvar = keras.layers.Dense(self.encoding_dim)
        
        # Decoder (for reconstruction)
        self.decoder = keras.layers.Dense(input_shape[-1])
        
    def call(self, inputs, training=None):
        # Encode to distribution parameters
        z_mean = self.encoder_mean(inputs)
        z_logvar = self.encoder_logvar(inputs)
        
        if training:
            # Sample using reparameterization trick
            epsilon = keras.random.normal(shape=ops.shape(z_mean))
            z = z_mean + ops.exp(0.5 * z_logvar) * epsilon
            
            # Add IB loss
            # I(X;T) ≈ KL divergence from prior
            kl_loss = -0.5 * ops.sum(
                1 + z_logvar - ops.square(z_mean) - ops.exp(z_logvar),
                axis=-1
            )
            self.add_loss(self.beta * ops.mean(kl_loss))
        else:
            z = z_mean
            
        # Decode
        reconstruction = self.decoder(z)
        
        return z, reconstruction
```

---

## Module 2: Understanding Attention Through Information Theory

### 2.1 Attention as Information Routing

Modern transformers achieve their remarkable capabilities through attention mechanisms. But what's really happening under the hood? Let's understand attention as an **information routing system**.

#### The Attention Equation Revisited

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

From an information-theoretic perspective:
- **Queries (Q)**: What information am I looking for?
- **Keys (K)**: What information is available?
- **Values (V)**: The actual information content
- **Attention weights**: Probability distribution over information sources

```python
class InformationAwareAttention(keras.layers.Layer):
    """Multi-head attention with information-theoretic analysis"""
    
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split last dimension into (num_heads, depth)"""
        x = ops.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return ops.transpose(x, axes=[0, 2, 1, 3])
    
    def compute_attention_entropy(self, attention_weights):
        """Compute entropy of attention distribution"""
        # Add small epsilon to prevent log(0)
        attention_weights = ops.clip(attention_weights, 1e-8, 1.0)
        entropy = -ops.sum(attention_weights * ops.log2(attention_weights), axis=-1)
        return ops.mean(entropy)
    
    def call(self, query, key, value, mask=None, training=None):
        batch_size = ops.shape(query)[0]
        
        # Linear transformations
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        
        # Split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = ops.matmul(Q, ops.transpose(K, axes=[0, 1, 3, 2]))
        scaled_attention_logits = matmul_qk / ops.sqrt(ops.cast(self.depth, 'float32'))
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = ops.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Compute and store attention entropy
        entropy = self.compute_attention_entropy(attention_weights)
        self.add_metric(entropy, name='attention_entropy')
        
        # Information flow analysis
        if training:
            # Detect entropy collapse
            low_entropy_threshold = 0.5
            if entropy < low_entropy_threshold:
                print(f"Warning: Low attention entropy detected: {entropy:.3f}")
        
        # Apply dropout
        if training:
            attention_weights = keras.layers.Dropout(self.dropout)(attention_weights)
        
        output = ops.matmul(attention_weights, V)
        output = ops.transpose(output, axes=[0, 2, 1, 3])
        output = ops.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(output)
        
        return output, attention_weights
```

### 2.2 Attention Entropy and Training Stability

Recent research (2023) has revealed that **attention entropy collapse** is a primary cause of training instability in transformers.

#### Understanding Entropy Collapse

When attention entropy becomes too low:
- Attention focuses on very few tokens
- Gradients become unstable
- Training may diverge

#### Solution: Entropy Regularization

```python
class EntropyRegularizedAttention(keras.layers.Layer):
    """Attention with entropy regularization to prevent collapse"""
    
    def __init__(self, d_model, num_heads, min_entropy=1.0, reg_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.min_entropy = min_entropy
        self.reg_weight = reg_weight
        
    def compute_entropy_loss(self, attention_scores):
        """Compute entropy regularization loss"""
        # Attention scores shape: (batch, heads, seq_len, seq_len)
        probs = ops.nn.softmax(attention_scores, axis=-1)
        
        # Compute entropy for each attention distribution
        entropy = -ops.sum(probs * ops.log(probs + 1e-8), axis=-1)
        mean_entropy = ops.mean(entropy)
        
        # Penalize low entropy
        entropy_loss = ops.maximum(0.0, self.min_entropy - mean_entropy)
        
        return entropy_loss
    
    def call(self, inputs, training=None):
        # Get attention output and scores
        output, scores = self.attention(
            inputs, inputs, 
            return_attention_scores=True,
            training=training
        )
        
        if training:
            # Add entropy regularization
            entropy_loss = self.compute_entropy_loss(scores)
            self.add_loss(self.reg_weight * entropy_loss)
            
        return output
```

### 2.3 Attention as Energy-Based Pattern Matching

A breakthrough insight connects transformers to **Hopfield networks** and energy-based models.

#### The Energy Perspective

Attention can be viewed as minimizing an energy function:

$$E(\xi; X) = \frac{1}{2}\xi^T\xi - \log\sum_i \exp(X^T\xi)$$

This reveals attention as:
1. **Pattern Storage**: Keys and values encode patterns in an energy landscape
2. **Pattern Retrieval**: Queries perform gradient descent to find relevant patterns
3. **Associative Memory**: Similar to how the brain might work!

```python
class EnergyBasedAttention(keras.layers.Layer):
    """Attention viewed through energy-based lens"""
    
    def __init__(self, d_model, num_heads, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.temperature = temperature
        self.depth = d_model // num_heads
        
        # Projections
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model) 
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)
        
    def compute_energy(self, queries, keys):
        """Compute energy landscape for attention"""
        
        # Scaled dot product
        qk = ops.matmul(queries, ops.transpose(keys, axes=[0, 1, 3, 2]))
        scaled_qk = qk / ops.sqrt(ops.cast(self.depth, 'float32'))
        
        # Energy interpretation: lower energy = higher attention
        energy = -scaled_qk / self.temperature
        
        return energy
    
    def call(self, query, key, value, training=None):
        batch_size = ops.shape(query)[0]
        
        # Linear projections
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        
        # Reshape for multi-head
        Q = ops.reshape(Q, (batch_size, -1, self.num_heads, self.depth))
        K = ops.reshape(K, (batch_size, -1, self.num_heads, self.depth))
        V = ops.reshape(V, (batch_size, -1, self.num_heads, self.depth))
        
        Q = ops.transpose(Q, axes=[0, 2, 1, 3])
        K = ops.transpose(K, axes=[0, 2, 1, 3])
        V = ops.transpose(V, axes=[0, 2, 1, 3])
        
        # Compute energy landscape
        energy = self.compute_energy(Q, K)
        
        # Convert energy to attention (Boltzmann distribution)
        attention_weights = ops.nn.softmax(-energy, axis=-1)
        
        # Retrieve patterns based on energy minimum
        output = ops.matmul(attention_weights, V)
        
        # Reshape back
        output = ops.transpose(output, axes=[0, 2, 1, 3])
        output = ops.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        
        # Add energy statistics as metrics
        if training:
            mean_energy = ops.mean(energy)
            energy_variance = ops.var(energy)
            self.add_metric(mean_energy, name='mean_energy')
            self.add_metric(energy_variance, name='energy_variance')
        
        return output, energy
```

---

## Module 3: Model Compression for Deployment

Now that we understand the information-theoretic foundations, let's apply these insights to compress models for real-world deployment.

### 3.1 Why Compression Works: An Information-Theoretic View

Neural networks are **massively overparameterized**. From an information theory perspective:

1. **Redundancy**: Many parameters encode similar information
2. **Noise**: Some parameters capture noise rather than signal
3. **Efficiency**: Optimal codes remove redundancy

This theoretical foundation justifies three main compression approaches: pruning, quantization, and knowledge distillation.

### 3.2 Pruning: Removing Redundant Connections

**Pruning** removes unnecessary weights based on the insight that many parameters contribute little to the model's function.

#### Magnitude-Based Pruning

```python
class MagnitudePruning(keras.layers.Layer):
    """Custom layer supporting magnitude-based pruning"""
    
    def __init__(self, units, sparsity_schedule=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sparsity_schedule = sparsity_schedule or self.default_schedule
        self.pruning_step = self.add_weight(
            name='pruning_step',
            shape=(),
            initializer='zeros',
            trainable=False
        )
        
    def default_schedule(self, step):
        """Default pruning schedule: gradual increase in sparsity"""
        # Start pruning after 1000 steps, reach 50% sparsity at 5000 steps
        if step < 1000:
            return 0.0
        elif step > 5000:
            return 0.5
        else:
            return 0.5 * (step - 1000) / 4000
    
    def build(self, input_shape):
        # Main weights
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Pruning mask
        self.pruning_mask = self.add_weight(
            name='pruning_mask',
            shape=(input_shape[-1], self.units),
            initializer='ones',
            trainable=False
        )
        
    def update_pruning_mask(self):
        """Update pruning mask based on current sparsity target"""
        current_step = self.pruning_step.numpy()
        target_sparsity = self.sparsity_schedule(current_step)
        
        if target_sparsity > 0:
            # Get weight magnitudes
            abs_weights = ops.abs(self.kernel)
            
            # Compute threshold
            k = int(ops.size(abs_weights) * (1 - target_sparsity))
            if k > 0:
                threshold = ops.top_k(ops.flatten(abs_weights), k=k)[0][-1]
                
                # Create new mask
                new_mask = ops.cast(abs_weights >= threshold, 'float32')
                self.pruning_mask.assign(new_mask)
                
                # Log sparsity
                actual_sparsity = 1 - ops.sum(new_mask) / ops.size(new_mask)
                print(f"Step {current_step}: Target sparsity {target_sparsity:.2%}, "
                      f"Actual sparsity {actual_sparsity:.2%}")
    
    def call(self, inputs, training=None):
        if training:
            # Update pruning step and mask
            self.pruning_step.assign_add(1)
            self.update_pruning_mask()
        
        # Apply pruning mask
        pruned_kernel = self.kernel * self.pruning_mask
        
        return ops.matmul(inputs, pruned_kernel) + self.bias
```

#### Structured Pruning

```python
class StructuredPruning:
    """Structured pruning for convolutional layers"""
    
    @staticmethod
    def compute_channel_importance(conv_layer):
        """Compute importance scores for each channel"""
        
        # Get convolutional filters
        filters = conv_layer.kernel  # Shape: (height, width, in_channels, out_channels)
        
        # Compute L2 norm for each output channel
        channel_norms = ops.sqrt(ops.sum(ops.square(filters), axis=[0, 1, 2]))
        
        return channel_norms
    
    @staticmethod
    def prune_channels(model, layer_name, pruning_ratio=0.5):
        """Prune least important channels from a layer"""
        
        layer = model.get_layer(layer_name)
        if not isinstance(layer, keras.layers.Conv2D):
            raise ValueError("Layer must be Conv2D")
        
        # Compute importance
        importance = StructuredPruning.compute_channel_importance(layer)
        
        # Determine channels to keep
        n_channels = len(importance)
        n_keep = int(n_channels * (1 - pruning_ratio))
        
        # Get indices of top channels
        top_indices = ops.top_k(importance, k=n_keep)[1]
        
        # Create new layer with fewer channels
        new_layer = keras.layers.Conv2D(
            filters=n_keep,
            kernel_size=layer.kernel_size,
            strides=layer.strides,
            padding=layer.padding,
            activation=layer.activation
        )
        
        print(f"Pruned {layer_name}: {n_channels} → {n_keep} channels "
              f"({pruning_ratio:.0%} reduction)")
        
        return new_layer, top_indices
```

### 3.3 Quantization: Reducing Numerical Precision

**Quantization** reduces memory and computation by using lower-precision numbers.

#### Understanding Quantization

- **FP32 → INT8**: 4× memory reduction
- **Key insight**: Neural networks are robust to small numerical errors
- **Information theory**: We're reducing the "alphabet size" of our encoding

```python
class QuantizationAwareTraining(keras.layers.Layer):
    """Layer that simulates quantization during training"""
    
    def __init__(self, bits=8, symmetric=True, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        self.symmetric = symmetric
        self.quantization_levels = 2 ** bits
        
    def quantize(self, x, training=None):
        """Simulate quantization effects"""
        
        if self.symmetric:
            # Symmetric quantization: [-max, max]
            max_val = ops.maximum(ops.max(ops.abs(x)), 1e-8)
            scale = (self.quantization_levels / 2 - 1) / max_val
            
            # Quantize
            x_scaled = x * scale
            x_quantized = ops.round(x_scaled)
            x_quantized = ops.clip(x_quantized, 
                                   -(self.quantization_levels / 2 - 1),
                                   self.quantization_levels / 2 - 1)
            
            # Dequantize
            x_dequantized = x_quantized / scale
        else:
            # Asymmetric quantization: [min, max]
            min_val = ops.min(x)
            max_val = ops.max(x)
            scale = (self.quantization_levels - 1) / (max_val - min_val + 1e-8)
            
            # Quantize
            x_shifted = x - min_val
            x_scaled = x_shifted * scale
            x_quantized = ops.round(x_scaled)
            x_quantized = ops.clip(x_quantized, 0, self.quantization_levels - 1)
            
            # Dequantize
            x_dequantized = x_quantized / scale + min_val
        
        # Straight-through estimator for gradients
        if training:
            return x + ops.stop_gradient(x_dequantized - x)
        else:
            return x_dequantized
    
    def call(self, inputs, training=None):
        return self.quantize(inputs, training)
```

### 3.4 Knowledge Distillation: Teaching Smaller Models

**Knowledge distillation** transfers knowledge from a large "teacher" model to a smaller "student" model.

#### Information-Theoretic Perspective

The teacher's soft predictions contain more information than hard labels:
- **Hard label**: "This is a cat" (1 bit of information)
- **Soft prediction**: "90% cat, 7% dog, 3% other" (more nuanced information)

```python
class DistillationTrainer:
    """Complete knowledge distillation training framework"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def distillation_loss(self, y_true, student_logits, teacher_logits):
        """Compute combined distillation and student loss"""
        
        # Standard student loss
        student_loss = keras.losses.sparse_categorical_crossentropy(
            y_true, student_logits, from_logits=True
        )
        
        # Distillation loss
        teacher_probs = ops.nn.softmax(teacher_logits / self.temperature)
        student_log_probs = ops.nn.log_softmax(student_logits / self.temperature)
        
        # KL divergence between teacher and student
        distillation_loss = ops.sum(
            teacher_probs * (ops.log(teacher_probs + 1e-8) - student_log_probs),
            axis=-1
        )
        distillation_loss *= self.temperature ** 2
        
        # Combined loss
        total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
        
        return total_loss, student_loss, distillation_loss

# Practical example: Compress a vision model
def create_teacher_student_pair():
    """Create teacher and student models for vision tasks"""
    
    # Teacher: Large CNN
    teacher = keras.Sequential([
        keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation='relu', name='teacher_feat1'),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', name='teacher_feat2'),
        keras.layers.Dense(10)
    ], name='teacher')
    
    # Student: Smaller CNN
    student = keras.Sequential([
        keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(32, 3, activation='relu', name='student_feat1'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu', name='student_feat2'),
        keras.layers.Dense(10)
    ], name='student')
    
    print(f"Teacher parameters: {teacher.count_params():,}")
    print(f"Student parameters: {student.count_params():,}")
    print(f"Compression ratio: {teacher.count_params() / student.count_params():.1f}x")
    
    return teacher, student

teacher, student = create_teacher_student_pair()
```

---

## Module 4: Theoretical Analysis of Transformer Capacity

### 4.1 Understanding Transformer Expressivity

How much can a transformer actually learn? Recent theoretical advances provide precise bounds.

#### Memory Capacity Bounds

For a transformer with $P$ parameters memorizing $n$ sequences:

**Theorem**: A one-layer transformer can memorize $n = \Theta(P)$ distinct input-output pairs.

This optimal result shows transformers use parameters efficiently!

```python
def analyze_transformer_capacity(model):
    """Analyze theoretical capacity of a transformer model"""
    
    # Count parameters excluding embeddings
    total_params = model.count_params()
    
    # Estimate embedding parameters
    embedding_params = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Embedding):
            embedding_params += layer.count_params()
    
    non_embedding_params = total_params - embedding_params
    
    # Theoretical capacity (order of magnitude)
    memorization_capacity = non_embedding_params  # O(P)
    
    print(f"Model Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-embedding parameters: {non_embedding_params:,}")
    print(f"Theoretical memorization capacity: O({non_embedding_params:,}) sequences")
    
    # Estimate information processing capacity
    # Each parameter can store ~log2(precision) bits
    bits_per_param = 5  # Empirical estimate for neural networks
    information_capacity = non_embedding_params * bits_per_param
    
    print(f"Information capacity: ~{information_capacity:,} bits")
    print(f"Equivalent to: ~{information_capacity / 8 / 1024 / 1024:.1f} MB of information")
    
    return non_embedding_params, information_capacity

# Example analysis
example_transformer = keras.Sequential([
    keras.layers.Embedding(10000, 512),
    keras.layers.MultiHeadAttention(num_heads=8, key_dim=64),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10000)
])

analyze_transformer_capacity(example_transformer)
```

### 4.2 Scaling Laws and Information Processing

The famous scaling laws reveal how transformers process information:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

where:
- $L$ = loss (bits per token)
- $N$ = number of parameters
- $N_c \approx 8.8 \times 10^{13}$ = critical parameter count
- $\alpha_N \approx 0.076$ = scaling exponent

```python
def plot_scaling_laws():
    """Visualize transformer scaling laws"""
    
    # Parameter counts from 1M to 1T
    param_counts = np.logspace(6, 12, 100)
    
    # Compute loss using scaling law
    Nc = 8.8e13
    alpha_N = 0.076
    loss = (Nc / param_counts) ** alpha_N
    
    # Also show data and compute scaling
    Dc = 5.4e13
    alpha_D = 0.095
    data_tokens = np.logspace(8, 14, 100)
    loss_data = (Dc / data_tokens) ** alpha_D
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameter scaling
    ax1.loglog(param_counts, loss, linewidth=2)
    ax1.set_xlabel('Number of Parameters', fontsize=12)
    ax1.set_ylabel('Loss (bits per token)', fontsize=12)
    ax1.set_title('Loss vs Model Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Annotate key model sizes
    model_sizes = {'GPT-2': 1.5e9, 'GPT-3': 175e9, 'GPT-4 (est)': 1e12}
    for name, size in model_sizes.items():
        if size >= param_counts[0] and size <= param_counts[-1]:
            loss_val = (Nc / size) ** alpha_N
            ax1.plot(size, loss_val, 'ro')
            ax1.annotate(name, (size, loss_val), xytext=(10, 10), 
                        textcoords='offset points')
    
    # Data scaling
    ax2.loglog(data_tokens, loss_data, linewidth=2, color='green')
    ax2.set_xlabel('Number of Training Tokens', fontsize=12)
    ax2.set_ylabel('Loss (bits per token)', fontsize=12)
    ax2.set_title('Loss vs Training Data', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_scaling_laws()
```

### 4.3 Information Bottleneck in Transformers

Each transformer layer acts as an information bottleneck, compressing while preserving task-relevant information.

```python
class InformationBottleneckAnalysis:
    """Analyze information flow through transformer layers"""
    
    def __init__(self, model):
        self.model = model
        self.layer_outputs = []
        
    def compute_layer_information(self, inputs, labels):
        """Estimate information at each layer"""
        
        # Get intermediate outputs
        layer_models = []
        for i, layer in enumerate(self.model.layers):
            if 'attention' in str(type(layer)).lower():
                intermediate_model = keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[i].output
                )
                layer_models.append((i, layer.name, intermediate_model))
        
        # Compute information metrics
        information_metrics = []
        
        for idx, name, intermediate_model in layer_models:
            output = intermediate_model(inputs)
            
            # Estimate entropy (compression)
            # Using discrete binning for approximation
            output_flat = ops.reshape(output, (-1, output.shape[-1]))
            
            # Simple entropy estimation
            output_std = ops.std(output_flat, axis=0)
            avg_std = ops.mean(output_std)
            
            # Higher std ≈ higher entropy ≈ less compression
            compression_estimate = 1.0 / (1.0 + avg_std)
            
            information_metrics.append({
                'layer_idx': idx,
                'layer_name': name,
                'compression': float(compression_estimate),
                'dimensionality': output.shape[-1]
            })
        
        return information_metrics
    
    def visualize_information_flow(self, metrics):
        """Visualize how information changes through layers"""
        
        layers = [m['layer_idx'] for m in metrics]
        compression = [m['compression'] for m in metrics]
        dims = [m['dimensionality'] for m in metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Compression through layers
        ax1.plot(layers, compression, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Compression Estimate', fontsize=12)
        ax1.set_title('Information Compression Through Transformer Layers', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Dimensionality
        ax2.plot(layers, dims, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Hidden Dimension', fontsize=12)
        ax2.set_title('Representation Dimensionality', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

---

## Module 5: Advanced Compression Techniques

### 5.1 Combining Multiple Compression Methods

The most effective compression combines multiple techniques synergistically.

```python
class ComprehensiveCompression:
    """Combine pruning, quantization, and distillation"""
    
    def __init__(self, original_model, target_compression=10):
        self.original_model = original_model
        self.target_compression = target_compression
        
    def compress_model(self):
        """Apply comprehensive compression pipeline"""
        
        print("=== Comprehensive Model Compression Pipeline ===\n")
        
        # Step 1: Knowledge Distillation
        print("Step 1: Knowledge Distillation")
        student_model = self.create_student_architecture()
        
        # Step 2: Quantization-Aware Training
        print("\nStep 2: Quantization-Aware Training")
        quantized_student = self.apply_quantization_aware_training(student_model)
        
        # Step 3: Pruning
        print("\nStep 3: Magnitude-Based Pruning")
        pruned_model = self.apply_pruning(quantized_student, sparsity=0.5)
        
        # Step 4: Final optimization
        print("\nStep 4: Final Optimization")
        final_model = self.optimize_for_deployment(pruned_model)
        
        # Analyze compression
        self.analyze_compression_results(final_model)
        
        return final_model
    
    def create_student_architecture(self):
        """Create smaller student architecture"""
        
        # Analyze original model
        original_params = self.original_model.count_params()
        target_params = original_params // (self.target_compression // 2)
        
        print(f"Original model: {original_params:,} parameters")
        print(f"Target student: ~{target_params:,} parameters")
        
        # Create student (simplified for demo)
        student = keras.Sequential([
            keras.layers.Input(shape=self.original_model.input_shape[1:]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10)
        ])
        
        return student
    
    def apply_quantization_aware_training(self, model):
        """Add quantization to model"""
        
        # Add quantization layers
        quantized_model = keras.Sequential()
        
        for layer in model.layers:
            quantized_model.add(layer)
            if isinstance(layer, keras.layers.Dense):
                quantized_model.add(QuantizationAwareTraining(bits=8))
        
        print("Added quantization layers for 8-bit inference")
        return quantized_model
    
    def apply_pruning(self, model, sparsity):
        """Apply magnitude-based pruning"""
        
        # Note: Simplified implementation
        print(f"Applying {sparsity:.0%} sparsity to dense layers")
        
        # In practice, would use pruning callbacks during training
        return model
    
    def optimize_for_deployment(self, model):
        """Final optimizations for deployment"""
        
        # Remove training-only layers
        # Fuse batch normalization
        # Optimize graph
        
        print("Applied deployment optimizations")
        return model
    
    def analyze_compression_results(self, compressed_model):
        """Analyze final compression results"""
        
        original_params = self.original_model.count_params()
        compressed_params = compressed_model.count_params()
        
        # Estimate size reduction
        # Original: 32-bit floats
        # Compressed: 8-bit integers + 50% sparse
        original_size = original_params * 4  # bytes
        compressed_size = compressed_params * 1 * 0.5  # 8-bit, 50% sparse
        
        print("\n=== Compression Results ===")
        print(f"Parameter reduction: {original_params:,} → {compressed_params:,}")
        print(f"Size reduction: {original_size/1024/1024:.1f} MB → "
              f"{compressed_size/1024/1024:.1f} MB")
        print(f"Total compression: {original_size/compressed_size:.1f}x")

# Demonstrate comprehensive compression
demo_model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10)
])

compressor = ComprehensiveCompression(demo_model, target_compression=10)
# compressed_model = compressor.compress_model()
```

### 5.2 Hardware-Aware Optimization

Different hardware benefits from different compression techniques.

```python
class HardwareAwareCompression:
    """Optimize compression for specific hardware targets"""
    
    @staticmethod
    def optimize_for_mobile(model):
        """Optimize for mobile devices (ARM CPUs)"""
        
        optimizations = {
            'quantization': 'int8',  # ARM NEON supports int8
            'pruning': 'structured',  # Better for CPU caches
            'batch_size': 1,  # Single inference
            'fusion': True  # Fuse operations
        }
        
        print("Mobile optimization profile:")
        for key, value in optimizations.items():
            print(f"  {key}: {value}")
        
        return optimizations
    
    @staticmethod
    def optimize_for_edge_tpu(model):
        """Optimize for Google Edge TPU"""
        
        optimizations = {
            'quantization': 'uint8',  # Edge TPU requirement
            'operations': 'supported_only',  # Limited op support
            'model_size': '<10MB',  # Memory constraint
            'batch_size': 1
        }
        
        print("Edge TPU optimization profile:")
        for key, value in optimizations.items():
            print(f"  {key}: {value}")
        
        return optimizations
    
    @staticmethod
    def optimize_for_gpu(model):
        """Optimize for GPU inference"""
        
        optimizations = {
            'quantization': 'fp16',  # GPUs handle fp16 well
            'pruning': 'unstructured',  # GPUs can handle sparse ops
            'batch_size': 'dynamic',  # Batched inference
            'memory': 'optimize_layout'  # Memory coalescing
        }
        
        print("GPU optimization profile:")
        for key, value in optimizations.items():
            print(f"  {key}: {value}")
        
        return optimizations
```

### 5.3 Emerging Techniques

#### Dynamic Sparsity

```python
class DynamicSparsityLayer(keras.layers.Layer):
    """Layer with input-dependent sparsity"""
    
    def __init__(self, units, base_sparsity=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.base_sparsity = base_sparsity
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Gating network for dynamic sparsity
        self.gate_network = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
    def call(self, inputs):
        # Compute input-dependent sparsity
        sparsity_factor = self.gate_network(inputs)
        
        # Apply dynamic pruning
        weight_magnitude = ops.abs(self.kernel)
        threshold = ops.quantile(
            ops.flatten(weight_magnitude),
            self.base_sparsity * sparsity_factor
        )
        
        dynamic_mask = ops.cast(weight_magnitude > threshold, 'float32')
        sparse_kernel = self.kernel * dynamic_mask
        
        return ops.matmul(inputs, sparse_kernel) + self.bias
```

---

## Practical Exercises

### Exercise Set 1: Information Theory Foundations

1. **Entropy Calculator**: Build a tool that computes entropy for any probability distribution and visualizes how entropy changes as you modify the distribution.

2. **Mutual Information Estimator**: Implement MINE (Mutual Information Neural Estimation) to estimate MI between high-dimensional variables.

3. **Information Bottleneck Autoencoder**: Create an autoencoder that explicitly optimizes the IB objective and visualize the compression-relevance tradeoff.

### Exercise Set 2: Attention Analysis

1. **Attention Entropy Monitor**: Build a callback that tracks attention entropy during training and automatically adjusts hyperparameters to prevent collapse.

2. **Energy Landscape Visualizer**: Create an interactive tool to visualize how attention energy landscapes evolve during training.

3. **Information Flow Tracer**: Implement attention rollout and flow analysis to understand how information propagates through transformer layers.

### Exercise Set 3: Model Compression

1. **Compression Pipeline**: Build an end-to-end compression pipeline that combines pruning, quantization, and distillation to achieve 10x compression with minimal accuracy loss.

2. **Hardware-Specific Optimizer**: Create a tool that automatically selects compression techniques based on target hardware specifications.

3. **Dynamic Compression**: Implement a model that adjusts its compression level based on available computational resources at inference time.

### Exercise Set 4: Advanced Projects

1. **Information-Theoretic Architecture Search**: Design a neural architecture search method that uses information theory metrics to guide the search process.

2. **Adaptive Precision Training**: Implement a training method that automatically determines optimal bit-widths for different layers based on their information content.

3. **Theoretical Analysis Tool**: Build a tool that analyzes any transformer model and provides theoretical bounds on its capacity and scaling behavior.

---

## Conclusion: Bringing It All Together

### Key Takeaways

1. **Information Theory Provides Deep Insights**: Understanding entropy, mutual information, and KL divergence illuminates how neural networks process information.

2. **Attention Is Information Routing**: Transformers can be understood as sophisticated information routing systems with energy-based pattern matching.

3. **Compression Works Due to Redundancy**: Neural networks are overparameterized, containing redundant information that can be removed without hurting performance.

4. **Multiple Techniques Synergize**: The best compression results come from combining pruning, quantization, and distillation.

5. **Theory Guides Practice**: Scaling laws and capacity bounds help us design better models and training procedures.

### Future Directions

The intersection of information theory and deep learning continues to yield insights:

- **Emergent Abilities**: Understanding how scaling leads to qualitatively new capabilities
- **Efficient Architectures**: Designing models that achieve optimal information processing with minimal parameters
- **Adaptive Systems**: Models that dynamically adjust their information processing based on input complexity
- **Theoretical Foundations**: Deeper understanding of why deep learning works so well

### Final Project: Build Your Own Compressed Transformer

As a capstone project, implement a complete system that:

1. Trains a transformer model on a text or vision task
2. Analyzes its information-theoretic properties
3. Applies comprehensive compression techniques
4. Deploys the compressed model with Keras 3
5. Measures and reports the compression-performance tradeoff

```python
class CompressedTransformerProject:
    """Template for final project"""
    
    def __init__(self, task='classification', dataset='cifar10'):
        self.task = task
        self.dataset = dataset
        
    def build_teacher_transformer(self):
        """Build and train the teacher transformer"""
        pass
        
    def analyze_information_flow(self):
        """Analyze information-theoretic properties"""
        pass
        
    def compress_model(self):
        """Apply comprehensive compression"""
        pass
        
    def evaluate_compression(self):
        """Evaluate compression-performance tradeoff"""
        pass
        
    def deploy_model(self):
        """Deploy compressed model with Keras 3"""
        pass

# Start your project!
project = CompressedTransformerProject()
```

### Resources for Continued Learning

1. **Papers**:
   - "Attention Is All You Need" (Vaswani et al.)
   - "Deep Learning and the Information Bottleneck Principle" (Tishby & Zaslavsky)
   - "The Lottery Ticket Hypothesis" (Frankle & Carbin)

2. **Books**:
   - "Elements of Information Theory" (Cover & Thomas)
   - "Information Theory, Inference, and Learning Algorithms" (MacKay)

3. **Tools**:
   - Keras 3 Documentation: keras.io
   - TensorFlow Model Optimization Toolkit
   - Neural Network Intelligence (NNI) for compression

This course has equipped you with both theoretical understanding and practical skills to work at the cutting edge of neural network compression and optimization. The journey from Shannon's information theory to modern transformer compression shows how fundamental mathematical insights can revolutionize practical applications.

Remember: The best solutions often come from deeply understanding the theoretical foundations while maintaining a pragmatic focus on real-world deployment. Happy compressing!