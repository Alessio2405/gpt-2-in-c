#ifndef GPT2_H
#define GPT2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Model configuration
#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024
#define EMBED_DIM 768
#define NUM_HEADS 12
#define NUM_LAYERS 12
#define FF_DIM 3072
#define BATCH_SIZE 4
#define LEARNING_RATE 0.0001
#define MAX_EPOCHS 100

// Data structures
typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    Matrix* weights;
    Matrix* bias;
    Matrix* weight_grad;
    Matrix* bias_grad;
} LinearLayer;

typedef struct {
    Matrix* query;
    Matrix* key;
    Matrix* value;
    Matrix* output;
    Matrix* query_grad;
    Matrix* key_grad;
    Matrix* value_grad;
    Matrix* output_grad;
} AttentionLayer;

typedef struct {
    AttentionLayer attention;
    LinearLayer attn_proj;
    LinearLayer ff1;
    LinearLayer ff2;
    Matrix* ln1_weight;
    Matrix* ln1_bias;
    Matrix* ln2_weight;
    Matrix* ln2_bias;
    Matrix* ln1_grad_weight;
    Matrix* ln1_grad_bias;
    Matrix* ln2_grad_weight;
    Matrix* ln2_grad_bias;
} TransformerBlock;

typedef struct {
    Matrix* token_embed;
    Matrix* pos_embed;
    TransformerBlock* blocks;
    LinearLayer lm_head;
    Matrix* ln_f_weight;
    Matrix* ln_f_bias;
    Matrix* ln_f_grad_weight;
    Matrix* ln_f_grad_bias;
} GPT2Model;

// Matrix operations
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix* mat);
void random_init_matrix(Matrix* mat);
void zero_matrix(Matrix* mat);
Matrix* matrix_multiply(Matrix* a, Matrix* b);
Matrix* matrix_add(Matrix* a, Matrix* b);
Matrix* matrix_subtract(Matrix* a, Matrix* b);
Matrix* matrix_transpose(Matrix* mat);
void matrix_copy(Matrix* dest, Matrix* src);
Matrix* softmax(Matrix* mat);
Matrix* layer_norm(Matrix* input, Matrix* weight, Matrix* bias);
Matrix* gelu(Matrix* input);

// Attention mechanism
Matrix* scaled_dot_product_attention(Matrix* query, Matrix* key, Matrix* value, Matrix* mask);
Matrix* multi_head_attention(AttentionLayer* attn, Matrix* input, Matrix* mask);

// Model functions
GPT2Model* create_gpt2_model();
void free_gpt2_model(GPT2Model* model);
Matrix* forward_pass(GPT2Model* model, int* input_ids, int seq_len);
void backward_pass(GPT2Model* model, Matrix* loss_grad);
void update_weights(GPT2Model* model);

// Training functions
float cross_entropy_loss(Matrix* logits, int* targets, int seq_len);
void train_step(GPT2Model* model, int* input_ids, int* targets, int seq_len);
void train_epoch(GPT2Model* model, int** batches, int** targets, int num_batches, int seq_len);

// Generation functions
int sample_from_logits(Matrix* logits, int position, float temperature);
char* generate_text(GPT2Model* model, const char* prompt, int max_length, float temperature);
void complete_text(GPT2Model* model, const char* prompt);

// Utility functions
int* tokenize_text(const char* text, int* seq_len);

#endif 