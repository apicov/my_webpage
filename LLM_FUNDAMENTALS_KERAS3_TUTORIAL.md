# LLM Fundamentals Tutorial: Keras 3.0 Edition

## 🧠 Introduction to Large Language Models with Keras 3.0

Large Language Models (LLMs) have revolutionized AI by demonstrating remarkable capabilities in understanding and generating human language. This tutorial covers everything from basic concepts to advanced implementations using Keras 3.0's multi-backend capabilities.

### What are Large Language Models?

**Large Language Models (LLMs)** are artificial intelligence systems trained on vast amounts of text data to understand and generate human language. Think of them as incredibly sophisticated autocomplete systems that can:

1. **Read and Understand**: Process text to understand context, meaning, and relationships between words
2. **Learn Patterns**: Identify patterns in language, grammar, knowledge, and reasoning
3. **Generate Responses**: Create coherent, contextually appropriate text that sounds human-like

### How Do LLMs Work?

At their core, LLMs work through these key steps:

1. **Tokenization**: Converting words and phrases into numbers (tokens) that computers can process
2. **Embedding**: Converting tokens into high-dimensional vectors that capture meaning
3. **Processing**: Using neural networks (specifically Transformers) to understand relationships
4. **Generation**: Producing new tokens based on learned patterns and context

### The Transformer Revolution

The **Transformer architecture** (introduced in 2017) revolutionized LLMs by introducing:
- **Self-Attention**: The ability to "pay attention" to different parts of input text
- **Parallel Processing**: Processing entire sequences at once (unlike RNNs)
- **Scalability**: Ability to handle much larger models and datasets

### Why Keras 3.0 for LLMs?

Keras 3.0 brings several advantages for LLM development:

1. **Multi-backend Support**: Choose the best backend for your needs:
   - **TensorFlow**: Most mature, great for production
   - **PyTorch**: Popular in research, dynamic computation
   - **JAX**: Fastest for training, functional programming style

2. **Unified API**: Write code once, run on any backend
3. **Better Performance**: Optimized for modern hardware (GPUs, TPUs)
4. **Easier Deployment**: Simplified model serving and production deployment
5. **Integration**: Works seamlessly with your existing TinyML and IoT projects

### Prerequisites

Before starting this tutorial, you should be familiar with:
- Basic Python programming
- Fundamental machine learning concepts
- Basic understanding of neural networks
- Your existing Flask and React setup (from previous tutorials)

**What you'll learn:**
- LLM architecture and training principles with Keras 3.0
- Transformer models and attention mechanisms
- Fine-tuning and prompt engineering
- Practical implementation with Keras 3.0
- Integration with your existing projects

---

## 🏗️ Chapter 1: Understanding LLM Architecture with Keras 3.0

### The Transformer Architecture

The Transformer architecture, introduced in "Attention Is All You Need" (2017), is the foundation of modern LLMs. Let's implement it using Keras 3.0.

#### What is the Transformer Architecture?

The **Transformer** is a neural network architecture that revolutionized natural language processing. Unlike previous models (RNNs, LSTMs) that process text sequentially, Transformers can:

1. **Process entire sequences at once** (parallel processing)
2. **Capture long-range dependencies** through attention mechanisms
3. **Scale to much larger models** and datasets
4. **Handle variable-length inputs** efficiently

#### Key Components of Transformers:

1. **Multi-Head Attention**: The core mechanism that allows the model to focus on different parts of the input
2. **Positional Encoding**: Adds information about word positions (since Transformers don't process sequentially)
3. **Feed-Forward Networks**: Simple neural networks that process each position independently
4. **Layer Normalization**: Stabilizes training by normalizing activations
5. **Residual Connections**: Help with gradient flow during training

#### Understanding Attention Mechanisms:

**Attention** is like having a spotlight that can focus on different parts of a sentence. For example, when processing "The cat sat on the mat", the model might pay more attention to "cat" when trying to understand what "sat" refers to.

The attention mechanism works by:
1. **Query (Q)**: What we're looking for
2. **Key (K)**: What each word offers
3. **Value (V)**: The actual content of each word
4. **Attention Score**: How much to focus on each word

```python
import keras
from keras import layers
import math
import numpy as np

class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention Layer - The core mechanism of Transformers
    
    This layer implements the attention mechanism that allows the model to focus on
    different parts of the input sequence. Think of it as having multiple "spotlights"
    that can each focus on different aspects of the text.
    
    Parameters:
    - d_model: The dimension of the model (e.g., 512, 768, 1024)
    - num_heads: Number of attention heads (e.g., 8, 12, 16)
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure d_model is divisible by num_heads for clean splitting
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Dimension of each attention head
        self.d_k = d_model // num_heads
        
        # Linear transformation layers for Query, Key, Value, and Output
        # These convert the input into the Q, K, V representations
        self.w_q = layers.Dense(d_model)  # Query transformation
        self.w_k = layers.Dense(d_model)  # Key transformation  
        self.w_v = layers.Dense(d_model)  # Value transformation
        self.w_o = layers.Dense(d_model)  # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention - The heart of the attention mechanism
        
        This function computes how much attention to pay to each part of the input.
        It's like having a spotlight that can focus on different words in a sentence.
        
        Parameters:
        - Q: Query matrix - what we're looking for
        - K: Key matrix - what each word offers
        - V: Value matrix - the actual content of each word
        - mask: Optional mask to hide certain positions (e.g., future tokens in decoder)
        
        Returns:
        - output: The weighted combination of values
        - attention_weights: How much attention was paid to each position
        """
        # Step 1: Calculate attention scores using dot product
        # This measures how much each query should pay attention to each key
        # We transpose K to align dimensions for matrix multiplication
        scores = keras.ops.matmul(Q, keras.ops.transpose(K, axes=[0, 1, 3, 2])) / math.sqrt(self.d_k)
        
        # Step 2: Apply masking if provided (used in decoder to prevent looking at future tokens)
        if mask is not None:
            # Set masked positions to very negative values so they get ~0 attention after softmax
            scores = keras.ops.where(mask == 0, -1e9, scores)
        
        # Step 3: Apply softmax to convert scores to probabilities (attention weights)
        # This ensures all attention weights sum to 1
        attention_weights = keras.ops.softmax(scores, axis=-1)
        
        # Step 4: Apply attention weights to values to get the final output
        # This is like taking a weighted average of the values
        output = keras.ops.matmul(attention_weights, V)
        return output, attention_weights
    
    def call(self, query, key, value, mask=None, training=None):
        """
        Forward pass of the Multi-Head Attention layer
        
        This is the main function that gets called during model inference and training.
        It orchestrates the entire attention mechanism.
        
        Parameters:
        - query: Input sequence (what we're looking for)
        - key: Keys for attention (what each position offers)
        - value: Values to attend to (actual content)
        - mask: Optional mask to hide certain positions
        - training: Whether we're in training mode
        
        Returns:
        - output: The attended output sequence
        """
        # Get batch size for reshaping operations
        batch_size = keras.ops.shape(query)[0]
        
        # Step 1: Linear transformations to create Q, K, V
        # Each input is transformed into Query, Key, and Value representations
        # These transformations allow the model to learn different aspects of the input
        
        # Transform query into Query matrix
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        Q = keras.ops.reshape(Q, [batch_size, -1, self.num_heads, self.d_k])  # Split into heads
        Q = keras.ops.transpose(Q, axes=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, d_k]
        
        # Transform key into Key matrix
        K = self.w_k(key)  # [batch_size, seq_len, d_model]
        K = keras.ops.reshape(K, [batch_size, -1, self.num_heads, self.d_k])  # Split into heads
        K = keras.ops.transpose(K, axes=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, d_k]
        
        # Transform value into Value matrix
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        V = keras.ops.reshape(V, [batch_size, -1, self.num_heads, self.d_k])  # Split into heads
        V = keras.ops.transpose(V, axes=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, d_k]
        
        # Step 2: Apply scaled dot-product attention
        # This computes how much attention each position should pay to every other position
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 3: Concatenate attention heads
        # We have multiple attention heads, each focusing on different aspects
        # Now we combine their outputs back into a single representation
        
        # Transpose back to [batch_size, seq_len, num_heads, d_k]
        attention_output = keras.ops.transpose(attention_output, axes=[0, 2, 1, 3])
        # Reshape to [batch_size, seq_len, d_model] (concatenate all heads)
        attention_output = keras.ops.reshape(attention_output, [batch_size, -1, self.d_model])
        
        # Step 4: Final linear transformation
        # This projects the concatenated attention output back to the desired dimension
        output = self.w_o(attention_output)
        return output

class TransformerBlock(layers.Layer):
    """
    Transformer Block - A complete Transformer layer
    
    This is a single layer of the Transformer architecture. Each block contains:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network
    4. Add & Norm (residual connection + layer normalization)
    
    Parameters:
    - d_model: Dimension of the model (e.g., 512, 768, 1024)
    - num_heads: Number of attention heads
    - d_ff: Dimension of the feed-forward network (usually 4x d_model)
    - dropout: Dropout rate for regularization
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Multi-Head Self-Attention layer
        # This allows the model to focus on different parts of the input sequence
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer Normalization layers
        # These stabilize training by normalizing activations
        self.norm1 = layers.LayerNormalization()  # After attention
        self.norm2 = layers.LayerNormalization()  # After feed-forward
        
        # Feed-Forward Network
        # This is a simple 2-layer neural network that processes each position independently
        # It typically expands the dimension (d_model -> d_ff) then contracts it back (d_ff -> d_model)
        self.feed_forward = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),  # Expand dimension
            layers.Dropout(dropout),                # Regularization
            layers.Dense(d_model)                   # Contract back to original dimension
        ])
        
        # Dropout for regularization
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, mask=None, training=None):
        """
        Forward pass of the Transformer Block
        
        This implements the complete Transformer block with residual connections
        and layer normalization. The flow is:
        1. Self-Attention → Add & Norm
        2. Feed-Forward → Add & Norm
        
        Parameters:
        - x: Input sequence [batch_size, seq_len, d_model]
        - mask: Optional attention mask
        - training: Whether in training mode
        
        Returns:
        - x: Transformed sequence [batch_size, seq_len, d_model]
        """
        # Step 1: Multi-Head Self-Attention
        # The attention mechanism allows each position to attend to all positions
        # We use the same input for query, key, and value (self-attention)
        attn_output = self.attention(x, x, x, mask, training)
        
        # Step 2: Add & Norm (Residual Connection + Layer Normalization)
        # Residual connection: x + dropout(attention_output)
        # This helps with gradient flow during training
        # Layer normalization stabilizes the activations
        x = self.norm1(x + self.dropout(attn_output, training=training))
        
        # Step 3: Feed-Forward Network
        # This processes each position independently
        # It's like having a small neural network at each position
        ff_output = self.feed_forward(x, training=training)
        
        # Step 4: Add & Norm (Residual Connection + Layer Normalization)
        # Another residual connection and normalization
        x = self.norm2(x + self.dropout(ff_output, training=training))
        
        return x

### 🔍 **Understanding What We Just Built**

Now that we've implemented the core Transformer components, let's understand what each part does and why it's important:

#### **1. Multi-Head Attention Explained**

Think of **Multi-Head Attention** as having multiple "spotlights" that can each focus on different aspects of a sentence:

**Example**: For the sentence "The cat sat on the mat because it was comfortable"

- **Head 1** might focus on **subject-verb relationships** (cat → sat)
- **Head 2** might focus on **spatial relationships** (on → mat)  
- **Head 3** might focus on **causal relationships** (because → comfortable)
- **Head 4** might focus on **pronoun references** (it → mat)

**Why Multiple Heads?**
- Each head can specialize in different types of relationships
- More heads = more capacity to understand complex patterns
- Parallel processing makes it computationally efficient

#### **2. The Attention Mechanism Step-by-Step**

Let's trace through what happens when processing "The cat sat":

1. **Input**: "The cat sat" → [token1, token2, token3]

2. **Linear Transformations**:
   - Query: "What am I looking for?"
   - Key: "What does each word offer?"
   - Value: "What's the actual content?"

3. **Attention Scores**: 
   - How much should "sat" pay attention to "The"? (probably low)
   - How much should "sat" pay attention to "cat"? (probably high)
   - How much should "sat" pay attention to "sat"? (medium)

4. **Softmax**: Convert scores to probabilities (sum to 1)

5. **Weighted Sum**: Combine values based on attention weights

#### **3. Transformer Block Architecture**

Each Transformer block follows this pattern:

```
Input → Self-Attention → Add & Norm → Feed-Forward → Add & Norm → Output
```

**Why This Pattern?**
- **Self-Attention**: Captures relationships between all positions
- **Add & Norm**: Residual connections help gradients flow, normalization stabilizes training
- **Feed-Forward**: Processes each position independently, adds non-linearity
- **Multiple Blocks**: Each block refines the understanding further

#### **4. Key Innovations of Transformers**

**Before Transformers (RNNs/LSTMs)**:
- Process text sequentially (word by word)
- Struggle with long-range dependencies
- Hard to parallelize

**With Transformers**:
- Process entire sequence at once
- Can capture relationships between any two positions
- Highly parallelizable
- Scale to much larger models

#### **5. Why This Matters for LLMs**

Modern LLMs like GPT, BERT, and T5 are built on this architecture:

- **GPT**: Uses decoder-only Transformers (generates text)
- **BERT**: Uses encoder-only Transformers (understands text)
- **T5**: Uses encoder-decoder Transformers (translates/transforms text)

The attention mechanism allows these models to:
- Understand context across long documents
- Generate coherent, contextually appropriate text
- Learn complex language patterns
- Handle multiple languages and tasks

class SimpleTransformer(keras.Model):
    """
    Simple Transformer Model - A complete Transformer implementation
    
    This is a simplified version of a Transformer model that can be used for
    language modeling tasks. It includes all the essential components:
    
    1. Token Embeddings: Convert tokens to vectors
    2. Positional Encoding: Add position information
    3. Multiple Transformer Blocks: Process the sequence
    4. Output Layer: Generate predictions
    
    Parameters:
    - vocab_size: Size of the vocabulary (number of unique tokens)
    - d_model: Dimension of the model (embedding dimension)
    - num_heads: Number of attention heads
    - num_layers: Number of Transformer blocks
    - d_ff: Dimension of feed-forward networks
    - max_seq_len: Maximum sequence length
    - dropout: Dropout rate for regularization
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        # Token Embedding Layer
        # Converts token IDs (integers) to dense vectors
        # This is where the model learns word representations
        self.embedding = layers.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        # Since Transformers process all positions at once, they need explicit
        # information about token positions in the sequence
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Stack of Transformer Blocks
        # Each block refines the understanding of the sequence
        # More layers = more capacity to learn complex patterns
        self.transformer_blocks = [TransformerBlock(d_model, num_heads, d_ff, dropout) 
                                  for _ in range(num_layers)]
        
        # Dropout for regularization
        self.dropout = layers.Dropout(dropout)
        
        # Final Output Layer
        # Projects the final representations back to vocabulary size
        # This allows the model to predict the next token
        self.final_layer = layers.Dense(vocab_size)
        
    def create_positional_encoding(self, max_seq_len, d_model):
        """
        Create Positional Encoding for Transformer
        
        Since Transformers process all positions simultaneously, they need explicit
        information about token positions. This function creates sinusoidal positional
        encodings that the model can learn to use.
        
        The encoding uses sine and cosine functions of different frequencies:
        - Even dimensions: sin(pos / 10000^(2i/d_model))
        - Odd dimensions: cos(pos / 10000^(2i/d_model))
        
        This allows the model to:
        1. Learn relative positions (position 5 is close to position 6)
        2. Generalize to sequences longer than those seen during training
        3. Maintain the same encoding regardless of sequence length
        
        Parameters:
        - max_seq_len: Maximum sequence length to support
        - d_model: Dimension of the model (must match embedding dimension)
        
        Returns:
        - pe: Positional encoding tensor [1, max_seq_len, d_model]
        """
        # Initialize positional encoding matrix
        pe = keras.ops.zeros([max_seq_len, d_model])
        
        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        position = keras.ops.arange(0, max_seq_len, dtype='float32')
        position = keras.ops.expand_dims(position, axis=1)  # [max_seq_len, 1]
        
        # Calculate division terms for different frequencies
        # This creates different frequencies for different dimensions
        div_term = keras.ops.exp(keras.ops.arange(0, d_model, 2, dtype='float32') * 
                                -(math.log(10000.0) / d_model))
        
        # Apply sine function to even dimensions
        # This creates a unique pattern for each position
        pe = keras.ops.tensor_scatter_nd_update(
            pe,
            keras.ops.stack([keras.ops.arange(0, max_seq_len), 
                           keras.ops.arange(0, d_model, 2)], axis=1),
            keras.ops.sin(position * div_term)
        )
        
        # Apply cosine function to odd dimensions
        # This creates a complementary pattern
        pe = keras.ops.tensor_scatter_nd_update(
            pe,
            keras.ops.stack([keras.ops.arange(0, max_seq_len), 
                           keras.ops.arange(1, d_model, 2)], axis=1),
            keras.ops.cos(position * div_term)
        )
        
        # Add batch dimension for broadcasting
        return keras.ops.expand_dims(pe, axis=0)  # [1, max_seq_len, d_model]
    
    def call(self, x, mask=None, training=None):
        """
        Forward pass of the Transformer model
        
        This is the main function that processes input sequences through the entire
        Transformer architecture. The flow is:
        
        1. Token Embedding: Convert token IDs to vectors
        2. Scale Embeddings: Multiply by sqrt(d_model) for stability
        3. Add Positional Encoding: Add position information
        4. Apply Dropout: Regularization during training
        5. Process through Transformer Blocks: Multiple layers of attention
        6. Final Output: Project to vocabulary size for predictions
        
        Parameters:
        - x: Input token IDs [batch_size, seq_len]
        - mask: Optional attention mask
        - training: Whether in training mode
        
        Returns:
        - output: Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        # Get sequence length for positional encoding
        seq_len = keras.ops.shape(x)[1]
        
        # Step 1: Token Embedding
        # Convert token IDs (integers) to dense vectors
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # Step 2: Scale Embeddings
        # Multiply by sqrt(d_model) to prevent embeddings from being too large
        # This helps with training stability
        x = x * math.sqrt(self.embedding.embedding_dim)
        
        # Step 3: Add Positional Encoding
        # Add position information to each token
        # This tells the model where each token is in the sequence
        x = x + self.pos_encoding[:, :seq_len]  # [batch_size, seq_len, d_model]
        
        # Step 4: Apply Dropout
        # Regularization to prevent overfitting
        x = self.dropout(x, training=training)
        
        # Step 5: Process through Transformer Blocks
        # Each block refines the understanding of the sequence
        # The output of each block becomes the input to the next
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, training)
        
        # Step 6: Final Output Layer
        # Project the final representations to vocabulary size
        # This allows the model to predict the next token
        output = self.final_layer(x)  # [batch_size, seq_len, vocab_size]
        return output

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_len = 512

model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)
print("Model created successfully!")

### 🎯 **Understanding the Model Parameters**

Let's break down what each parameter means and why it's important:

#### **Model Architecture Parameters:**

1. **`vocab_size = 10000`**
   - Number of unique tokens in your vocabulary
   - Larger vocabulary = more words, but more parameters
   - Typical values: 10K-50K for English models

2. **`d_model = 512`**
   - Dimension of the model (embedding dimension)
   - Larger = more capacity, but more computation
   - Typical values: 512 (small), 768 (medium), 1024 (large)

3. **`num_heads = 8`**
   - Number of attention heads
   - Each head focuses on different aspects
   - Usually d_model must be divisible by num_heads
   - Typical values: 8, 12, 16

4. **`num_layers = 6`**
   - Number of Transformer blocks
   - More layers = deeper understanding, but more computation
   - Typical values: 6-24 for smaller models, 96+ for large models

5. **`d_ff = 2048`**
   - Dimension of feed-forward networks
   - Usually 4x d_model for good performance
   - Larger = more capacity in feed-forward layers

6. **`max_seq_len = 512`**
   - Maximum sequence length the model can handle
   - Longer sequences = more context, but more memory
   - Typical values: 512, 1024, 2048, 4096

#### **Model Capacity Comparison:**

| Model Size | d_model | num_layers | num_heads | Parameters | Use Case |
|------------|---------|------------|-----------|------------|----------|
| Small | 512 | 6 | 8 | ~15M | Learning, prototyping |
| Medium | 768 | 12 | 12 | ~85M | Production, fine-tuning |
| Large | 1024 | 24 | 16 | ~350M | Research, high performance |
| XL | 2048 | 48 | 32 | ~1.5B | State-of-the-art |

#### **Memory and Computation Considerations:**

- **Memory**: Scales with `vocab_size × d_model + seq_len × d_model × num_layers`
- **Computation**: Scales with `seq_len² × d_model × num_layers` (attention is quadratic)
- **Training Time**: Scales with both memory and computation requirements

### 🔧 **How to Use This Model**

This model can be used for various language modeling tasks:

1. **Text Generation**: Predict the next token in a sequence
2. **Language Modeling**: Learn the probability distribution of text
3. **Feature Extraction**: Use intermediate representations for other tasks
4. **Fine-tuning**: Adapt to specific domains or tasks
```

### Understanding Attention Mechanisms

```python
def visualize_attention(attention_weights, tokens):
    """Visualize attention weights using Keras 3.0"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to numpy for visualization
    attention_np = keras.ops.convert_to_numpy(attention_weights[0])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_np, 
                xticklabels=tokens, yticklabels=tokens, 
                cmap='Blues', annot=True, fmt='.2f')
    plt.title('Attention Weights')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

# Example usage
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
attention_weights = keras.ops.random.normal([1, 6, 6])  # Simulated attention weights
visualize_attention(attention_weights, tokens)
```

---

## 🎯 Chapter 2: Working with Pre-trained LLMs in Keras 3.0

### Using Keras 3.0 with Hugging Face Models

```python
import keras
from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFAutoModelForSequenceClassification
import numpy as np

# Load pre-trained model and tokenizer
model_name = "gpt2"  # or "bert-base-uncased", "t5-base", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

# Text generation
def generate_text(prompt, max_length=100, temperature=0.7):
    """Generate text using a pre-trained LLM with Keras 3.0"""
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    
    # Generate
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "The future of artificial intelligence is"
generated = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

### Text Classification with LLMs

```python
def classify_text_with_llm(text, labels):
    """Use LLM for text classification with Keras 3.0"""
    
    # Load classification model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
    )
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    
    # Predict
    outputs = model(inputs)
    probabilities = keras.ops.softmax(outputs.logits, axis=1)
    predicted_class = keras.ops.argmax(probabilities, axis=1).numpy()[0]
    
    return labels[predicted_class], probabilities[0].numpy().tolist()

# Example usage
text = "I love this product! It's amazing!"
labels = ["positive", "negative", "neutral"]
prediction, probs = classify_text_with_llm(text, labels)
print(f"Text: {text}")
print(f"Prediction: {prediction}")
print(f"Probabilities: {dict(zip(labels, probs))}")
```

---

## 🔧 Chapter 3: Fine-tuning LLMs with Keras 3.0

### Fine-tuning for Specific Tasks

```python
import keras
from keras import layers
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np

class LLMFineTuner:
    def __init__(self, model_name, task_type="text-generation"):
        self.model_name = model_name
        self.task_type = task_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if task_type == "text-generation":
            self.model = TFAutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    
    def prepare_dataset(self, texts, labels=None):
        """Prepare dataset for fine-tuning with Keras 3.0"""
        
        def tokenize_function(examples):
            if self.task_type == "text-generation":
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    return_tensors="tf"
                )
            else:
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    return_tensors="tf"
                )
        
        # Create dataset
        if labels:
            dataset_dict = {"text": texts, "label": labels}
        else:
            dataset_dict = {"text": texts}
        
        # Convert to Keras dataset
        dataset = keras.utils.data.Dataset.from_tensor_slices(dataset_dict)
        
        # Tokenize
        tokenized_dataset = dataset.map(tokenize_function)
        
        return tokenized_dataset
    
    def fine_tune(self, train_dataset, eval_dataset=None, epochs=3):
        """Fine-tune the model using Keras 3.0"""
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train
        history = self.model.fit(
            train_dataset,
            validation_data=eval_dataset,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        
        return history

# Example usage
finetuner = LLMFineTuner("gpt2", "text-generation")

# Prepare training data
training_texts = [
    "The weather is sunny today.",
    "I love programming in Python.",
    "Machine learning is fascinating.",
    # Add more training examples
]

train_dataset = finetuner.prepare_dataset(training_texts)
history = finetuner.fine_tune(train_dataset, epochs=2)
```

### LoRA (Low-Rank Adaptation) for Efficient Fine-tuning

```python
import keras
from keras import layers

class LoRALayer(layers.Layer):
    def __init__(self, original_layer, r=16, alpha=32, **kwargs):
        super().__init__(**kwargs)
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        
        # Create LoRA adapters
        input_dim = original_layer.input_spec.axes[-1]
        output_dim = original_layer.output_spec.axes[-1]
        
        self.lora_A = layers.Dense(r, use_bias=False, name=f"{self.name}_lora_A")
        self.lora_B = layers.Dense(output_dim, use_bias=False, name=f"{self.name}_lora_B")
        
        # Scaling factor
        self.scaling = alpha / r
        
    def call(self, inputs, training=None):
        # Original layer output
        original_output = self.original_layer(inputs, training=training)
        
        # LoRA adaptation
        lora_output = self.lora_B(self.lora_A(inputs))
        
        # Combine
        return original_output + self.scaling * lora_output

def apply_lora_to_model(model, r=16, alpha=32):
    """Apply LoRA to a pre-trained model"""
    
    # Create a new model with LoRA layers
    inputs = keras.Input(shape=model.input_spec.shape[1:])
    
    # Apply LoRA to attention layers
    x = inputs
    for layer in model.layers:
        if isinstance(layer, layers.MultiHeadAttention):
            # Replace with LoRA version
            lora_layer = LoRALayer(layer, r=r, alpha=alpha)
            x = lora_layer(x)
        else:
            x = layer(x)
    
    return keras.Model(inputs=inputs, outputs=x)

# Example usage
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
lora_model = apply_lora_to_model(model)

# Now fine-tune with LoRA
finetuner = LLMFineTuner("gpt2", "text-generation")
finetuner.model = lora_model  # Use LoRA model
```

---

## 🎨 Chapter 4: Prompt Engineering with Keras 3.0

### Basic Prompt Engineering Techniques

```python
class PromptEngineer:
    def __init__(self):
        self.templates = {
            "classification": "Classify the following text as {labels}: {text}",
            "summarization": "Summarize the following text in {max_words} words: {text}",
            "translation": "Translate the following text from {source_lang} to {target_lang}: {text}",
            "question_answering": "Answer the following question based on the context: Context: {context} Question: {question}",
            "code_generation": "Write Python code to {task}. Requirements: {requirements}",
        }
    
    def create_prompt(self, template_name, **kwargs):
        """Create a prompt using a template"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        return self.templates[template_name].format(**kwargs)
    
    def create_few_shot_prompt(self, examples, query):
        """Create a few-shot prompt"""
        prompt = ""
        
        # Add examples
        for example in examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        
        # Add query
        prompt += f"Input: {query}\nOutput:"
        
        return prompt
    
    def create_chain_of_thought_prompt(self, question):
        """Create a chain-of-thought prompt"""
        return f"""Let's approach this step by step:

Question: {question}

Let me think about this step by step:
1) First, I need to understand what's being asked
2) Then, I'll break it down into smaller parts
3) Finally, I'll solve each part and combine the results

Let me start:"""

# Example usage
engineer = PromptEngineer()

# Classification prompt
classification_prompt = engineer.create_prompt(
    "classification",
    labels="positive, negative, neutral",
    text="I love this product!"
)
print("Classification prompt:", classification_prompt)

# Few-shot prompt
examples = [
    {"input": "2 + 3", "output": "5"},
    {"input": "7 - 4", "output": "3"},
    {"input": "5 * 6", "output": "30"}
]
few_shot_prompt = engineer.create_few_shot_prompt(examples, "8 + 9")
print("Few-shot prompt:", few_shot_prompt)

# Chain-of-thought prompt
cot_prompt = engineer.create_chain_of_thought_prompt(
    "If a train travels 120 km in 2 hours, what is its speed in km/h?"
)
print("Chain-of-thought prompt:", cot_prompt)
```

### Advanced Prompt Engineering

```python
class AdvancedPromptEngineer:
    def __init__(self):
        self.system_prompts = {
            "assistant": "You are a helpful AI assistant. Provide accurate and helpful responses.",
            "expert": "You are an expert in your field. Provide detailed, technical explanations.",
            "creative": "You are a creative writer. Generate imaginative and engaging content.",
            "analytical": "You are an analytical thinker. Break down complex problems systematically."
        }
    
    def create_role_based_prompt(self, role, task, context=""):
        """Create a role-based prompt"""
        system_prompt = self.system_prompts.get(role, self.system_prompts["assistant"])
        
        prompt = f"""System: {system_prompt}

Context: {context}

Task: {task}

Response:"""
        
        return prompt
    
    def create_structured_prompt(self, task, constraints, examples=None):
        """Create a structured prompt with constraints"""
        prompt = f"""Task: {task}

Constraints:
"""
        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint}\n"
        
        if examples:
            prompt += "\nExamples:\n"
            for example in examples:
                prompt += f"- {example}\n"
        
        prompt += "\nPlease provide your response following the constraints above:"
        
        return prompt
    
    def create_iterative_prompt(self, initial_prompt, feedback):
        """Create an iterative prompt based on feedback"""
        return f"""Previous response: {initial_prompt}

Feedback: {feedback}

Please improve your response based on the feedback above:"""

# Example usage
advanced_engineer = AdvancedPromptEngineer()

# Role-based prompt
expert_prompt = advanced_engineer.create_role_based_prompt(
    "expert",
    "Explain quantum computing principles",
    "For a technical audience with basic physics knowledge"
)
print("Expert prompt:", expert_prompt)

# Structured prompt
structured_prompt = advanced_engineer.create_structured_prompt(
    "Write a Python function to sort a list",
    [
        "Use only built-in Python functions",
        "Handle edge cases (empty list, single element)",
        "Include type hints",
        "Add docstring"
    ],
    examples=[
        "def sort_list(lst: List[int]) -> List[int]:",
        "def bubble_sort(arr: List[Any]) -> List[Any]:"
    ]
)
print("Structured prompt:", structured_prompt)
```

---

## 🔄 Chapter 5: LLM Integration with Your Projects

### Integrating LLMs with Flask Backend

```python
# Add to your existing app.py
import keras
from transformers import TFAutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

class LLMService:
    def __init__(self):
        # Initialize models
        self.sentiment_analyzer = pipeline("sentiment-analysis", framework="tf")
        self.text_generator = pipeline("text-generation", model="gpt2", framework="tf")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="tf")
        
        # Load custom model for chat
        self.chat_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chat_model = TFAutoModelForCausalLM.from_pretrained("gpt2")
        
        if self.chat_tokenizer.pad_token is None:
            self.chat_tokenizer.pad_token = self.chat_tokenizer.eos_token
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        result = self.sentiment_analyzer(text)
        return {
            "sentiment": result[0]["label"],
            "confidence": result[0]["score"],
            "text": text
        }
    
    def generate_text(self, prompt, max_length=100):
        """Generate text from prompt"""
        result = self.text_generator(prompt, max_length=max_length, num_return_sequences=1)
        return {
            "generated_text": result[0]["generated_text"],
            "prompt": prompt
        }
    
    def summarize_text(self, text, max_length=130, min_length=30):
        """Summarize text"""
        result = self.summarizer(text, max_length=max_length, min_length=min_length)
        return {
            "summary": result[0]["summary_text"],
            "original_length": len(text.split()),
            "summary_length": len(result[0]["summary_text"].split())
        }
    
    def chat_response(self, messages, max_length=100):
        """Generate chat response"""
        # Combine messages into context
        context = " ".join([msg["content"] for msg in messages[-5:]])  # Last 5 messages
        
        inputs = self.chat_tokenizer.encode(context, return_tensors="tf", truncation=True, max_length=512)
        
        outputs = self.chat_model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.chat_tokenizer.eos_token_id
        )
        
        response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(context):].strip()  # Remove context from response

# Initialize LLM service
llm_service = LLMService()

# Add new routes to your Flask app
@app.route('/api/llm/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    result = llm_service.analyze_sentiment(text)
    return jsonify(result)

@app.route('/api/llm/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    result = llm_service.generate_text(prompt, max_length)
    return jsonify(result)

@app.route('/api/llm/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get('text', '')
    max_length = data.get('max_length', 130)
    min_length = data.get('min_length', 30)
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    result = llm_service.summarize_text(text, max_length, min_length)
    return jsonify(result)

@app.route('/api/llm/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages', [])
    max_length = data.get('max_length', 100)
    
    if not messages:
        return jsonify({'error': 'Messages are required'}), 400
    
    response = llm_service.chat_response(messages, max_length)
    return jsonify({'response': response})
```

### React Frontend Integration

```javascript
// Add to your React frontend
class LLMService {
    constructor() {
        this.baseURL = '/api/llm';
    }
    
    async analyzeSentiment(text) {
        const response = await fetch(`${this.baseURL}/sentiment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        return response.json();
    }
    
    async generateText(prompt, maxLength = 100) {
        const response = await fetch(`${this.baseURL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, max_length: maxLength })
        });
        return response.json();
    }
    
    async summarizeText(text, maxLength = 130, minLength = 30) {
        const response = await fetch(`${this.baseURL}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text, 
                max_length: maxLength, 
                min_length: minLength 
            })
        });
        return response.json();
    }
    
    async chat(messages, maxLength = 100) {
        const response = await fetch(`${this.baseURL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages, max_length: maxLength })
        });
        return response.json();
    }
}

// React component for LLM features
import React, { useState } from 'react';

const LLMFeatures = () => {
    const [text, setText] = useState('');
    const [prompt, setPrompt] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const llmService = new LLMService();
    
    const handleSentimentAnalysis = async () => {
        setLoading(true);
        try {
            const result = await llmService.analyzeSentiment(text);
            setResult(result);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const handleTextGeneration = async () => {
        setLoading(true);
        try {
            const result = await llmService.generateText(prompt);
            setResult(result);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const handleSummarization = async () => {
        setLoading(true);
        try {
            const result = await llmService.summarizeText(text);
            setResult(result);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="max-w-4xl mx-auto p-6">
            <h2 className="text-3xl font-bold mb-6">LLM Features (Keras 3.0)</h2>
            
            {/* Sentiment Analysis */}
            <div className="mb-8 p-4 border rounded-lg">
                <h3 className="text-xl font-semibold mb-4">Sentiment Analysis</h3>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter text to analyze..."
                    className="w-full p-2 border rounded"
                    rows="3"
                />
                <button
                    onClick={handleSentimentAnalysis}
                    disabled={loading}
                    className="mt-2 bg-blue-500 text-white px-4 py-2 rounded"
                >
                    {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                </button>
            </div>
            
            {/* Text Generation */}
            <div className="mb-8 p-4 border rounded-lg">
                <h3 className="text-xl font-semibold mb-4">Text Generation</h3>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Enter a prompt..."
                    className="w-full p-2 border rounded"
                    rows="3"
                />
                <button
                    onClick={handleTextGeneration}
                    disabled={loading}
                    className="mt-2 bg-green-500 text-white px-4 py-2 rounded"
                >
                    {loading ? 'Generating...' : 'Generate Text'}
                </button>
            </div>
            
            {/* Summarization */}
            <div className="mb-8 p-4 border rounded-lg">
                <h3 className="text-xl font-semibold mb-4">Text Summarization</h3>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter text to summarize..."
                    className="w-full p-2 border rounded"
                    rows="5"
                />
                <button
                    onClick={handleSummarization}
                    disabled={loading}
                    className="mt-2 bg-purple-500 text-white px-4 py-2 rounded"
                >
                    {loading ? 'Summarizing...' : 'Summarize Text'}
                </button>
            </div>
            
            {/* Results */}
            {result && (
                <div className="p-4 bg-gray-100 rounded-lg">
                    <h3 className="text-lg font-semibold mb-2">Results:</h3>
                    <pre className="whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default LLMFeatures;
```

---

## 🎯 Chapter 6: Advanced LLM Techniques with Keras 3.0

### Custom Training Pipeline

```python
import keras
from keras import layers
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np

class CustomLLMTrainer:
    def __init__(self, model_name, tokenizer_name=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = TFAutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, texts, max_length=512):
        """Prepare training data with proper tokenization"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="tf"
            )
        
        # Create Keras dataset
        dataset = keras.utils.data.Dataset.from_tensor_slices({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function)
        
        return tokenized_dataset
    
    def train_with_custom_loss(self, train_dataset, custom_loss_fn=None, epochs=3):
        """Train with custom loss function"""
        
        # Define custom loss if provided
        if custom_loss_fn:
            loss = custom_loss_fn
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=loss,
            metrics=['accuracy']
        )
        
        # Train
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        
        return history

# Example custom loss function
def focal_loss(y_true, y_pred, alpha=1, gamma=2):
    """Focal loss for handling class imbalance"""
    ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
    pt = keras.ops.exp(-ce_loss)
    focal_loss = alpha * keras.ops.power(1 - pt, gamma) * ce_loss
    return keras.ops.mean(focal_loss)

# Usage
trainer = CustomLLMTrainer("gpt2")
training_texts = [
    "The future of AI is bright.",
    "Machine learning transforms industries.",
    # Add more training data
]
train_dataset = trainer.prepare_dataset(training_texts)
history = trainer.train_with_custom_loss(train_dataset, focal_loss, epochs=2)
```

### Model Evaluation and Metrics

```python
import keras
import numpy as np

class LLMEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def calculate_perplexity(self, test_texts):
        """Calculate perplexity on test data"""
        total_loss = 0
        total_tokens = 0
        
        for text in test_texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="tf", 
                truncation=True, 
                max_length=512
            )
            
            outputs = self.model(inputs, training=False)
            loss = outputs.loss
            
            total_loss += loss.numpy() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
        
        avg_loss = total_loss / total_tokens
        perplexity = keras.ops.exp(avg_loss)
        
        return perplexity.numpy()
    
    def calculate_bleu_score(self, generated_texts, reference_texts):
        """Calculate BLEU score for text generation"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothie = SmoothingFunction().method1
        total_bleu = 0
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # Tokenize
            gen_tokens = gen_text.split()
            ref_tokens = ref_text.split()
            
            # Calculate BLEU
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
            total_bleu += bleu
        
        return total_bleu / len(generated_texts)
    
    def evaluate_generation_quality(self, prompts, reference_responses):
        """Evaluate generation quality with multiple metrics"""
        generated_responses = []
        
        # Generate responses
        for prompt in prompts:
            inputs = self.tokenizer.encode(prompt, return_tensors="tf")
            
            outputs = self.model.generate(
                inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_responses.append(response[len(prompt):].strip())
        
        # Calculate metrics
        metrics = {
            "bleu_score": self.calculate_bleu_score(generated_responses, reference_responses),
            "avg_length": np.mean([len(resp.split()) for resp in generated_responses]),
            "diversity": self.calculate_diversity(generated_responses)
        }
        
        return metrics, generated_responses
    
    def calculate_diversity(self, texts, n_grams=2):
        """Calculate diversity using n-gram overlap"""
        from collections import Counter
        
        all_ngrams = []
        for text in texts:
            words = text.split()
            ngrams = [' '.join(words[i:i+n_grams]) for i in range(len(words)-n_grams+1)]
            all_ngrams.extend(ngrams)
        
        ngram_counts = Counter(all_ngrams)
        unique_ngrams = len(ngram_counts)
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0

# Usage
evaluator = LLMEvaluator(model, tokenizer)

# Test data
test_texts = [
    "The weather is beautiful today.",
    "I love programming in Python.",
    "Machine learning is fascinating."
]

perplexity = evaluator.calculate_perplexity(test_texts)
print(f"Perplexity: {perplexity:.2f}")

# Generation evaluation
prompts = ["The future of", "I believe that", "Technology will"]
references = ["AI is bright", "innovation matters", "transform society"]

metrics, responses = evaluator.evaluate_generation_quality(prompts, references)
print("Generation Metrics:", metrics)
```

---

## 🎉 Conclusion

You now have a comprehensive understanding of LLMs with Keras 3.0:

✅ **LLM Architecture** - Transformers, attention mechanisms, model building  
✅ **Pre-trained Models** - Using Hugging Face with Keras 3.0  
✅ **Fine-tuning** - Custom training, LoRA, efficient adaptation  
✅ **Prompt Engineering** - Templates, few-shot, chain-of-thought  
✅ **Integration** - Flask backend, React frontend, API development  
✅ **Advanced Techniques** - Custom training, evaluation, metrics  

### Key Advantages of Keras 3.0 for LLMs:

1. **Multi-backend support** - TensorFlow, PyTorch, JAX
2. **Unified API** - Consistent interface across backends
3. **Better performance** - Optimized for modern hardware
4. **Easier deployment** - Simplified model serving
5. **Integration** - Works seamlessly with your TinyML projects

### Next Steps:

1. **Explore different backends** - Try PyTorch and JAX backends
2. **Implement advanced techniques** - More sophisticated fine-tuning
3. **Build production systems** - Scalable LLM services
4. **Integrate with TinyML** - Edge-cloud AI systems

**Happy LLM development with Keras 3.0!** 🚀

---

*Build intelligent applications with the power of Keras 3.0 and language models!* 🎯 