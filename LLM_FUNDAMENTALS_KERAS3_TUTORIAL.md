# LLM Fundamentals Tutorial: Keras 3.0 Edition

## 🧠 Introduction to Large Language Models with Keras 3.0

Large Language Models (LLMs) have revolutionized AI by demonstrating remarkable capabilities in understanding and generating human language. This tutorial covers everything from basic concepts to advanced implementations using Keras 3.0's multi-backend capabilities.

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

```python
import keras
from keras import layers
import math
import numpy as np

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = layers.Dense(d_model)
        self.w_k = layers.Dense(d_model)
        self.w_v = layers.Dense(d_model)
        self.w_o = layers.Dense(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        scores = keras.ops.matmul(Q, keras.ops.transpose(K, axes=[0, 1, 3, 2])) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = keras.ops.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = keras.ops.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = keras.ops.matmul(attention_weights, V)
        return output, attention_weights
    
    def call(self, query, key, value, mask=None, training=None):
        batch_size = keras.ops.shape(query)[0]
        
        # Linear transformations
        Q = keras.ops.reshape(self.w_q(query), [batch_size, -1, self.num_heads, self.d_k])
        Q = keras.ops.transpose(Q, axes=[0, 2, 1, 3])
        
        K = keras.ops.reshape(self.w_k(key), [batch_size, -1, self.num_heads, self.d_k])
        K = keras.ops.transpose(K, axes=[0, 2, 1, 3])
        
        V = keras.ops.reshape(self.w_v(value), [batch_size, -1, self.num_heads, self.d_k])
        V = keras.ops.transpose(V, axes=[0, 2, 1, 3])
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = keras.ops.transpose(attention_output, axes=[0, 2, 1, 3])
        attention_output = keras.ops.reshape(attention_output, [batch_size, -1, self.d_model])
        
        # Final linear layer
        output = self.w_o(attention_output)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.feed_forward = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x, mask=None, training=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask, training)
        x = self.norm1(x + self.dropout(attn_output, training=training))
        
        # Feed-forward
        ff_output = self.feed_forward(x, training=training)
        x = self.norm2(x + self.dropout(ff_output, training=training))
        
        return x

class SimpleTransformer(keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads, d_ff, dropout) 
                                  for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)
        self.final_layer = layers.Dense(vocab_size)
        
    def create_positional_encoding(self, max_seq_len, d_model):
        pe = keras.ops.zeros([max_seq_len, d_model])
        position = keras.ops.arange(0, max_seq_len, dtype='float32')
        position = keras.ops.expand_dims(position, axis=1)
        
        div_term = keras.ops.exp(keras.ops.arange(0, d_model, 2, dtype='float32') * 
                                -(math.log(10000.0) / d_model))
        
        pe = keras.ops.tensor_scatter_nd_update(
            pe,
            keras.ops.stack([keras.ops.arange(0, max_seq_len), 
                           keras.ops.arange(0, d_model, 2)], axis=1),
            keras.ops.sin(position * div_term)
        )
        
        pe = keras.ops.tensor_scatter_nd_update(
            pe,
            keras.ops.stack([keras.ops.arange(0, max_seq_len), 
                           keras.ops.arange(1, d_model, 2)], axis=1),
            keras.ops.cos(position * div_term)
        )
        
        return keras.ops.expand_dims(pe, axis=0)
    
    def call(self, x, mask=None, training=None):
        seq_len = keras.ops.shape(x)[1]
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoding[:, :seq_len]
        x = self.dropout(x, training=training)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask, training)
        
        output = self.final_layer(x)
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