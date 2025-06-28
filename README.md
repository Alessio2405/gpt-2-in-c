# GPT-2 Implementation in C

A simplified yet functional implementation of the GPT-2 language model written entirely in C.

This project demonstrates the core concepts of transformer-based language models with a focus on educational clarity and minimal dependencies.

## Features

- **Pure C Implementation**: No external dependencies beyond standard C libraries
- **Complete Transformer Architecture**: Includes multi-head attention, feed-forward networks, and layer normalization
- **Training Capabilities**: Basic training loop with cross-entropy loss and gradient updates
- **Text Generation**: Interactive text completion with temperature-controlled sampling
- **Memory Management**: Proper allocation and deallocation of matrix structures
- **Educational Focus**: Well-commented code with debug output for learning purposes

## Architecture Overview

The implementation includes:

- **Token and Positional Embeddings**: Convert input tokens to dense vector representations
- **Multi-Head Self-Attention**: Core attention mechanism for modeling token relationships
- **Feed-Forward Networks**: Position-wise fully connected layers with GELU activation
- **Layer Normalization**: Stabilizes training and improves convergence
- **Language Modeling Head**: Projects hidden states to vocabulary logits

### Model Configuration

Default hyperparameters (defined in `gpt2.h`):
- **Vocabulary Size**: 50,257 (standard GPT-2 vocabulary)
- **Embedding Dimension**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12
- **Feed-Forward Dimension**: 3072
- **Maximum Sequence Length**: 1024
- **Learning Rate**: 0.0001

## File Structure

```
├── gpt2.h          # Header file with structure definitions and constants
├── gpt2.c          # Core implementation (matrix operations, model architecture)
├── main.c          # Training loop and text generation interface
├── train.txt       # Training data file (user-provided)
└── README.md       # This file
```

## Prerequisites

- C compiler (GCC, Clang, or MSVC)
- Standard C libraries (`stdio.h`, `stdlib.h`, `string.h`, `math.h`, `time.h`)

## Compilation

```bash
gcc -o gpt2 main.c gpt2.c -lm
```

Or using Makefile:
```makefile
CC=gcc
CFLAGS=-Wall -O2
LIBS=-lm

gpt2: main.c gpt2.c gpt2.h
	$(CC) $(CFLAGS) -o gpt2 main.c gpt2.c $(LIBS)

clean:
	rm -f gpt2
```

## Usage

### 1. Prepare Training Data

Create a `train.txt` (or simply use this one) file with your training text. Each line will be treated as a separate training example:

```
Hello world, this is a sample sentence.
Machine learning is fascinating.
Transformers have revolutionized NLP.
```

### 2. Run Training and Generation

```bash
./gpt2
```

The program will:
1. Load training data from `train.txt`
2. Initialize the GPT-2 model with random weights
3. Train for a specified number of epochs
4. Generate text completions for test prompts
5. Enter interactive mode for custom text generation

### 3. Execution

```
Enter prompt: Altherya

(…………………)

[forward_pass] ÔûÂ exiting
[generate_text] Forward pass completed
[generate_text] Sampled token: 32 (' ')
[generate_text] Generation complete, final seq_len=104
[generate_text] Final text: 'Altherya nel                                                                                                    '
Generated: "Altherya nel                                                                                                    "

```
Clearly it still have many issues, but this is for exercise purpose only.


## Code Structure

## Limitations and Simplifications

This implementation makes several simplifications for educational clarity:

1. **Character-level Tokenization**: Uses simple character mapping instead of BPE
2. **Simplified Attention**: Multi-head attention is partially implemented
3. **Basic Optimizer**: Uses simple gradient updates instead of Adam
4. **Limited Gradient Computation**: Backward pass is simplified
5. **No Regularization**: Missing dropout and weight decay
6. **Fixed Hyperparameters**: No dynamic learning rate scheduling

## Educational Value

This implementation is designed to help understand:

- **Transformer Architecture**: How attention, feed-forward layers, and normalization work together
- **Matrix Operations**: Low-level implementation of neural network computations
- **Memory Management**: Proper handling of dynamic memory in C
- **Training Loop**: How language models learn from sequential data
- **Text Generation**: Autoregressive sampling strategies

## Performance Considerations

- **Low Memory Usage**
- **Training Speed**: Significantly slower than optimized implementations
- **Numerical Stability**: Uses basic floating-point arithmetic
- **Scalability**: Not optimized for large datasets or models

## Extensions and Improvements

Potential enhancements:

1. **Better Tokenization**: Implement BPE or WordPiece tokenization
2. **Optimized Attention**: Full multi-head attention with proper reshaping
3. **Advanced Optimizers**: Adam, AdamW, or other modern optimizers
4. **Regularization**: Dropout, layer dropout, attention dropout
5. **Mixed Precision**: Half-precision training for memory efficiency
6. **Parallel Processing**: Multi-threading or GPU acceleration
7. **Model Checkpointing**: Save/load trained models
8. **Evaluation Metrics**: Perplexity calculation and validation

## Learning Resources

To better understand the concepts implemented here:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [GPT-2 Architecture](https://jalammar.github.io/illustrated-gpt2/) - GPT-2 specific details

## Contributing

This is an educational implementation. Contributions that improve clarity, add comments, or fix bugs are welcome. Please maintain the focus on readability and educational value.

## Disclaimer

This is a simplified implementation for educational purposes and should not be used for production applications. The model architecture and training procedures are significantly simplified compared to the original GPT-2.
