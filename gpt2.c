#include "gpt2.h"
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


Matrix* create_matrix(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)calloc(rows * cols, sizeof(float));
    return mat;
}

void free_matrix(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

void random_init_matrix(Matrix* mat) {
    srand(time(NULL));
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
}

void zero_matrix(Matrix* mat) {
    memset(mat->data, 0, mat->rows * mat->cols * sizeof(float));
}

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        printf("Matrix dimensions don't match for multiplication\n");
        return NULL;
    }
    
    Matrix* result = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * b->cols + j] = sum;
        }
    }
    return result;
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Matrix dimensions don't match for addition\n");
        return NULL;
    }
    
    Matrix* result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Matrix* matrix_subtract(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Matrix dimensions don't match for subtraction\n");
        return NULL;
    }
    
    Matrix* result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

// Not the optimal method 
Matrix* matrix_transpose(Matrix* mat) {
    Matrix* result = create_matrix(mat->cols, mat->rows);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result->data[j * mat->rows + i] = mat->data[i * mat->cols + j];
        }
    }
    return result;
}

void matrix_copy(Matrix* dest, Matrix* src) {
    if (dest->rows != src->rows || dest->cols != src->cols) {
        printf("Matrix dimensions don't match for copy\n");
        return;
    }
    memcpy(dest->data, src->data, src->rows * src->cols * sizeof(float));
}

Matrix* softmax(Matrix* mat) {
    Matrix* result = create_matrix(mat->rows, mat->cols);
    
    for (int i = 0; i < mat->rows; i++) {
        float max_val = mat->data[i * mat->cols];
        for (int j = 1; j < mat->cols; j++) {
            if (mat->data[i * mat->cols + j] > max_val) {
                max_val = mat->data[i * mat->cols + j];
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < mat->cols; j++) {
            result->data[i * mat->cols + j] = exp(mat->data[i * mat->cols + j] - max_val);
            sum += result->data[i * mat->cols + j];
        }
        
        for (int j = 0; j < mat->cols; j++) {
            result->data[i * mat->cols + j] /= sum;
        }
    }
    return result;
}

Matrix* layer_norm(Matrix* input, Matrix* weight, Matrix* bias) {
    Matrix* result = create_matrix(input->rows, input->cols);
    
    for (int i = 0; i < input->rows; i++) {
        float mean = 0.0f;
        for (int j = 0; j < input->cols; j++) {
            mean += input->data[i * input->cols + j];
        }
        mean /= input->cols;
        
        float var = 0.0f;
        for (int j = 0; j < input->cols; j++) {
            float diff = input->data[i * input->cols + j] - mean;
            var += diff * diff;
        }
        var /= input->cols;
        
        float std_dev = sqrt(var + 1e-5f);
        
        for (int j = 0; j < input->cols; j++) {
            float normalized = (input->data[i * input->cols + j] - mean) / std_dev;
            result->data[i * input->cols + j] = normalized * weight->data[j] + bias->data[j];
        }
    }
    return result;
}

Matrix* gelu(Matrix* input) {
    Matrix* result = create_matrix(input->rows, input->cols);
    for (int i = 0; i < input->rows * input->cols; i++) {
        float x = input->data[i];
        result->data[i] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
    return result;
}

// Attention mechanism
Matrix* scaled_dot_product_attention(Matrix* query, Matrix* key, Matrix* value, Matrix* mask) {
    Matrix* key_t = matrix_transpose(key);
    Matrix* scores = matrix_multiply(query, key_t);
    
    // Scale scores
    float scale = sqrt((float)key->cols);
    for (int i = 0; i < scores->rows * scores->cols; i++) {
        scores->data[i] /= scale;
    }
    
    // Apply mask
    if (mask) {
        for (int i = 0; i < scores->rows * scores->cols; i++) {
            scores->data[i] += mask->data[i];
        }
    }
    
    Matrix* attention_weights = softmax(scores);
    Matrix* output = matrix_multiply(attention_weights, value);
    
    free_matrix(key_t);
    free_matrix(scores);
    free_matrix(attention_weights);
    
    return output;
}

Matrix* multi_head_attention(AttentionLayer* attn, Matrix* input, Matrix* mask) {
    int head_dim = EMBED_DIM / NUM_HEADS;
    int seq_len = input->rows;
    
    // Project to query, key, value
    Matrix* q_proj = matrix_multiply(input, attn->query);
    Matrix* k_proj = matrix_multiply(input, attn->key);
    Matrix* v_proj = matrix_multiply(input, attn->value);
    
    // Reshape for multi-head attention (simplified)
    Matrix* output = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask);
    
    Matrix* final_output = matrix_multiply(output, attn->output);
    
    free_matrix(q_proj);
    free_matrix(k_proj);
    free_matrix(v_proj);
    free_matrix(output);
    
    return final_output;
}

// Model creation and initialization
GPT2Model* create_gpt2_model() {
    GPT2Model* model = (GPT2Model*)malloc(sizeof(GPT2Model));
    
    // Token and positional embeddings
    model->token_embed = create_matrix(VOCAB_SIZE, EMBED_DIM);
    model->pos_embed = create_matrix(MAX_SEQ_LEN, EMBED_DIM);
    random_init_matrix(model->token_embed);
    random_init_matrix(model->pos_embed);
    
    // Transformer blocks
    model->blocks = (TransformerBlock*)malloc(NUM_LAYERS * sizeof(TransformerBlock));
    for (int i = 0; i < NUM_LAYERS; i++) {
        TransformerBlock* block = &model->blocks[i];
        
        // Attention layer
        block->attention.query = create_matrix(EMBED_DIM, EMBED_DIM);
        block->attention.key = create_matrix(EMBED_DIM, EMBED_DIM);
        block->attention.value = create_matrix(EMBED_DIM, EMBED_DIM);
        block->attention.output = create_matrix(EMBED_DIM, EMBED_DIM);
        random_init_matrix(block->attention.query);
        random_init_matrix(block->attention.key);
        random_init_matrix(block->attention.value);
        random_init_matrix(block->attention.output);
        
        // Attention projection
        block->attn_proj.weights = create_matrix(EMBED_DIM, EMBED_DIM);
        block->attn_proj.bias = create_matrix(1, EMBED_DIM);
        block->attn_proj.weight_grad = create_matrix(EMBED_DIM, EMBED_DIM);
        block->attn_proj.bias_grad = create_matrix(1, EMBED_DIM);
        random_init_matrix(block->attn_proj.weights);
        zero_matrix(block->attn_proj.bias);
        zero_matrix(block->attn_proj.weight_grad);
        zero_matrix(block->attn_proj.bias_grad);
        
        // Feed-forward layers
        block->ff1.weights = create_matrix(EMBED_DIM, FF_DIM);
        block->ff1.bias = create_matrix(1, FF_DIM);
        block->ff1.weight_grad = create_matrix(EMBED_DIM, FF_DIM);
        block->ff1.bias_grad = create_matrix(1, FF_DIM);
        random_init_matrix(block->ff1.weights);
        zero_matrix(block->ff1.bias);
        zero_matrix(block->ff1.weight_grad);
        zero_matrix(block->ff1.bias_grad);
        
        block->ff2.weights = create_matrix(FF_DIM, EMBED_DIM);
        block->ff2.bias = create_matrix(1, EMBED_DIM);
        block->ff2.weight_grad = create_matrix(FF_DIM, EMBED_DIM);
        block->ff2.bias_grad = create_matrix(1, EMBED_DIM);
        random_init_matrix(block->ff2.weights);
        zero_matrix(block->ff2.bias);
        zero_matrix(block->ff2.weight_grad);
        zero_matrix(block->ff2.bias_grad);
        
        // Layer normalization
        block->ln1_weight = create_matrix(1, EMBED_DIM);
        block->ln1_bias = create_matrix(1, EMBED_DIM);
        block->ln1_grad_weight = create_matrix(1, EMBED_DIM);
        block->ln1_grad_bias = create_matrix(1, EMBED_DIM);
        for (int j = 0; j < EMBED_DIM; j++) {
            block->ln1_weight->data[j] = 1.0f;
            block->ln1_bias->data[j] = 0.0f;
        }
        zero_matrix(block->ln1_grad_weight);
        zero_matrix(block->ln1_grad_bias);
        
        block->ln2_weight = create_matrix(1, EMBED_DIM);
        block->ln2_bias = create_matrix(1, EMBED_DIM);
        block->ln2_grad_weight = create_matrix(1, EMBED_DIM);
        block->ln2_grad_bias = create_matrix(1, EMBED_DIM);
        for (int j = 0; j < EMBED_DIM; j++) {
            block->ln2_weight->data[j] = 1.0f;
            block->ln2_bias->data[j] = 0.0f;
        }
        zero_matrix(block->ln2_grad_weight);
        zero_matrix(block->ln2_grad_bias);
    }
    
    model->lm_head.weights = create_matrix(EMBED_DIM, VOCAB_SIZE);
    model->lm_head.bias = create_matrix(1, VOCAB_SIZE);
    model->lm_head.weight_grad = create_matrix(EMBED_DIM, VOCAB_SIZE);
    model->lm_head.bias_grad = create_matrix(1, VOCAB_SIZE);
    random_init_matrix(model->lm_head.weights);
    zero_matrix(model->lm_head.bias);
    zero_matrix(model->lm_head.weight_grad);
    zero_matrix(model->lm_head.bias_grad);
    
    model->ln_f_weight = create_matrix(1, EMBED_DIM);
    model->ln_f_bias = create_matrix(1, EMBED_DIM);
    model->ln_f_grad_weight = create_matrix(1, EMBED_DIM);
    model->ln_f_grad_bias = create_matrix(1, EMBED_DIM);
    for (int j = 0; j < EMBED_DIM; j++) {
        model->ln_f_weight->data[j] = 1.0f;
        model->ln_f_bias->data[j] = 0.0f;
    }
    zero_matrix(model->ln_f_grad_weight);
    zero_matrix(model->ln_f_grad_bias);
    
    return model;
}

void free_gpt2_model(GPT2Model* model) {
    if (!model) return;
    
    free_matrix(model->token_embed);
    free_matrix(model->pos_embed);
    
    for (int i = 0; i < NUM_LAYERS; i++) {
        TransformerBlock* block = &model->blocks[i];
        
        free_matrix(block->attention.query);
        free_matrix(block->attention.key);
        free_matrix(block->attention.value);
        free_matrix(block->attention.output);
        
        free_matrix(block->attn_proj.weights);
        free_matrix(block->attn_proj.bias);
        free_matrix(block->attn_proj.weight_grad);
        free_matrix(block->attn_proj.bias_grad);
        
        free_matrix(block->ff1.weights);
        free_matrix(block->ff1.bias);
        free_matrix(block->ff1.weight_grad);
        free_matrix(block->ff1.bias_grad);
        
        free_matrix(block->ff2.weights);
        free_matrix(block->ff2.bias);
        free_matrix(block->ff2.weight_grad);
        free_matrix(block->ff2.bias_grad);
        
        free_matrix(block->ln1_weight);
        free_matrix(block->ln1_bias);
        free_matrix(block->ln1_grad_weight);
        free_matrix(block->ln1_grad_bias);
        
        free_matrix(block->ln2_weight);
        free_matrix(block->ln2_bias);
        free_matrix(block->ln2_grad_weight);
        free_matrix(block->ln2_grad_bias);
    }
    
    free(model->blocks);
    
    free_matrix(model->lm_head.weights);
    free_matrix(model->lm_head.bias);
    free_matrix(model->lm_head.weight_grad);
    free_matrix(model->lm_head.bias_grad);
    
    free_matrix(model->ln_f_weight);
    free_matrix(model->ln_f_bias);
    free_matrix(model->ln_f_grad_weight);
    free_matrix(model->ln_f_grad_bias);
    
    free(model);
}

Matrix* safe_matrix_multiply(Matrix* a, Matrix* b) {
    if (!a || !b || !a->data || !b->data) {
        printf("ERROR: NULL matrix in multiplication\n");
        return NULL;
    }
    if (a->cols != b->rows) {
        printf("ERROR: Matrix dimensions don't match for multiplication (%dx%d) x (%dx%d)\n", 
               a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    return matrix_multiply(a, b);
}

Matrix* safe_layer_norm(Matrix* input, Matrix* weight, Matrix* bias) {
    if (!input || !weight || !bias || !input->data || !weight->data || !bias->data) {
        printf("ERROR: NULL matrix in layer norm\n");
        return NULL;
    }
    if (input->cols != weight->cols || input->cols != bias->cols) {
        printf("ERROR: Layer norm dimension mismatch\n");
        return NULL;
    }
    return layer_norm(input, weight, bias);
}

Matrix* forward_pass(GPT2Model* model, int* input_ids, int seq_len) {
    printf("[forward_pass] ▶ entered (seq_len=%d)\n", seq_len);
    fflush(stdout);

    if (!model || !input_ids || seq_len <= 0 || seq_len > MAX_SEQ_LEN) {
        printf("ERROR: Invalid input to forward_pass\n");
        return NULL;
    }

    printf("[forward_pass] ▶ embedding lookup start\n");
    fflush(stdout);
    
    Matrix* embeddings = create_matrix(seq_len, EMBED_DIM);
    if (!embeddings) {
        printf("ERROR: Failed to create embeddings matrix\n");
        return NULL;
    }
    
    for (int i = 0; i < seq_len; i++) {
        int tid = input_ids[i];
        if (tid < 0 || tid >= VOCAB_SIZE) {
            printf("WARNING: Invalid token %d at position %d, using 0\n", tid, i);
            tid = 0;
        }
        if (i >= MAX_SEQ_LEN) {
            printf("WARNING: Position %d exceeds MAX_SEQ_LEN, using 0\n", i);
            i = 0;
        }
        
        for (int j = 0; j < EMBED_DIM; j++) {
            embeddings->data[i*EMBED_DIM + j] =
                model->token_embed->data[tid*EMBED_DIM + j] +
                model->pos_embed->data[i*EMBED_DIM + j];
        }
    }
    printf("[forward_pass] ▶ embedding lookup done\n");
    fflush(stdout);

    Matrix* hidden = embeddings;

    // Simplified transformer blocks (reduce to 1 layer)
    int debug_layers = NUM_LAYERS > 1 ? 1 : NUM_LAYERS; 
    
    for (int layer = 0; layer < debug_layers; layer++) {
        TransformerBlock* blk = &model->blocks[layer];
        printf("[forward_pass] ▶ layer %d start\n", layer);
        fflush(stdout);

        // LayerNorm 1 - with safety checks
        printf("[forward_pass]   – ln1 start\n");
        fflush(stdout);
        Matrix* ln1 = safe_layer_norm(hidden, blk->ln1_weight, blk->ln1_bias);
        if (!ln1) {
            printf("ERROR: ln1 failed\n");
            free_matrix(hidden);
            return NULL;
        }
        printf("[forward_pass]   – ln1 done\n");
        fflush(stdout);

        // Simplified attention (skip multi-head for now)
        printf("[forward_pass]   – simplified attn start\n");
        fflush(stdout);
        
        // Just use a simple linear transformation instead of full attention
        Matrix* attn = safe_matrix_multiply(ln1, blk->attention.query);
        if (!attn) {
            printf("ERROR: attention failed\n");
            free_matrix(ln1);
            free_matrix(hidden);
            return NULL;
        }
        printf("[forward_pass]   – simplified attn done\n");
        fflush(stdout);

        // Residual connection
        Matrix* res1 = matrix_add(hidden, attn);
        if (!res1) {
            printf("ERROR: residual 1 failed\n");
            free_matrix(ln1);
            free_matrix(attn);
            free_matrix(hidden);
            return NULL;
        }

        // Skip the FFN for now to isolate the issue
        printf("[forward_pass]   – skipping FFN for debugging\n");
        fflush(stdout);

        // Cleanup & prepare next layer
        if (hidden != embeddings) free_matrix(hidden);
        hidden = res1;
        free_matrix(ln1);
        free_matrix(attn);

        printf("[forward_pass] ▶ layer %d done\n", layer);
        fflush(stdout);
    }

    printf("[forward_pass] ▶ all layers done\n");
    fflush(stdout);

    // Skip final layer norm and LM head for debugging
    printf("[forward_pass] ▶ creating simple output\n");
    fflush(stdout);
    
    // Create a simple output matrix instead of full LM head
    Matrix* logits = create_matrix(seq_len, VOCAB_SIZE);
    if (!logits) {
        printf("ERROR: Failed to create logits matrix\n");
        free_matrix(hidden);
        return NULL;
    }
    
    // Fill with simple values for debugging
    for (int i = 0; i < seq_len * VOCAB_SIZE; i++) {
        logits->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    printf("[forward_pass] ▶ exiting\n");
    fflush(stdout);
    
    if (hidden != embeddings) free_matrix(hidden);
    if (embeddings) free_matrix(embeddings);

    return logits;
}

float cross_entropy_loss(Matrix* logits, int* targets, int seq_len) {
    float total_loss = 0.0f;
    
    for (int i = 0; i < seq_len; i++) {
        float max_logit = logits->data[i * VOCAB_SIZE];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (logits->data[i * VOCAB_SIZE + j] > max_logit) {
                max_logit = logits->data[i * VOCAB_SIZE + j];
            }
        }
        
        float log_sum_exp = 0.0f;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            log_sum_exp += exp(logits->data[i * VOCAB_SIZE + j] - max_logit);
        }
        log_sum_exp = log(log_sum_exp) + max_logit;
        
        int target = targets[i];
        if (target >= VOCAB_SIZE) target = 0;
        
        total_loss += log_sum_exp - logits->data[i * VOCAB_SIZE + target];
    }
    
    return total_loss / seq_len;
}

void train_step(GPT2Model* model, int* input_ids, int* targets, int seq_len) {
    printf("[train_step] entered, seq_len=%d\n", seq_len);
    fflush(stdout);

    // Forward pass
    printf("[train_step] calling forward_pass...\n");
    fflush(stdout);
    Matrix* logits = forward_pass(model, input_ids, seq_len);
    printf("[train_step] forward_pass returned\n");
    fflush(stdout);

    // Calculate loss
    printf("[train_step] calling cross_entropy_loss...\n");
    fflush(stdout);
    float loss = cross_entropy_loss(logits, targets, seq_len);
    printf("[train_step] loss = %f\n", loss);
    fflush(stdout);

    // Backward pass...
    printf("[train_step] doing weight updates\n");
    fflush(stdout);
    for (size_t i = 0, n = model->lm_head.weights->rows * model->lm_head.weights->cols; i < n; i++) {
        model->lm_head.weights->data[i] -= LEARNING_RATE * ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    printf("[train_step] free logits and return\n");
    fflush(stdout);

    free_matrix(logits);
}

void train_epoch(GPT2Model* model, int** batches, int** targets, int num_batches, int seq_len) {
    for (int batch = 0; batch < num_batches; batch++) {
        train_step(model, batches[batch], targets[batch], seq_len);
    }
} 

int sample_from_logits(Matrix* logits, int position, float temperature) {
    // Safety check
    if (!logits || position < 0 || position >= logits->rows) {
        return 0; 
    }
    
    if (temperature <= 0.0f) {
        float max_val = logits->data[position * VOCAB_SIZE];
        int max_idx = 0;
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (logits->data[position * VOCAB_SIZE + i] > max_val) {
                max_val = logits->data[position * VOCAB_SIZE + i];
                max_idx = i;
            }
        }
        return max_idx;
    }
    
    // Apply temperature scaling
    float max_logit = logits->data[position * VOCAB_SIZE];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits->data[position * VOCAB_SIZE + i] > max_logit) {
            max_logit = logits->data[position * VOCAB_SIZE + i];
        }
    }
    
    // Compute softmax with temperature => NB. normally you'll need to use real softmax func here
    float sum = 0.0f;
    float probs[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = exp((logits->data[position * VOCAB_SIZE + i] - max_logit) / temperature);
        sum += probs[i];
    }
    
    // Safety check for sum
    if (sum <= 0.0f) {
        return 0; 
    }
    
    // Normalize probabilities
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] /= sum;
    }

    float random_val = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cumsum += probs[i];
        if (random_val <= cumsum) {
            if (i >= 32 && i <= 126) {
                return i;
            }
        }
    }
    
    return ' '; 
}

char* generate_text(GPT2Model* model, const char* prompt, int max_length, float temperature) {
    printf("[generate_text] Starting with prompt: '%s'\n", prompt);
    fflush(stdout);
        
    int prompt_len;
    int* prompt_tokens = tokenize_text(prompt, &prompt_len);
    printf("[generate_text] Prompt tokenized to %d tokens\n", prompt_len);
    fflush(stdout);
    
    int* sequence = (int*)malloc(max_length * sizeof(int));
    int seq_len = 0;
    
    for (int i = 0; i < prompt_len && i < max_length - 10; i++) {
        if (prompt_tokens[i] >= 0 && prompt_tokens[i] < VOCAB_SIZE) {
            sequence[seq_len++] = prompt_tokens[i];
        }
    }
    
    // If no valid prompt tokens, start with a simple character
    if (seq_len == 0) {
        sequence[seq_len++] = 'A'; // Start with 'A'
    }
    
    printf("[generate_text] Starting generation with seq_len=%d\n", seq_len);
    fflush(stdout);
    
    int max_iterations = 100; 
    for (int iter = 0; iter < max_iterations && seq_len < max_length - 1; iter++) {
        printf("[generate_text] Iteration %d, seq_len=%d\n", iter, seq_len);
        fflush(stdout);
        
        Matrix* logits = forward_pass(model, sequence, seq_len);
        if (!logits) {
            printf("ERROR: forward_pass failed\n");
            break;
        }
        printf("[generate_text] Forward pass completed\n");
        fflush(stdout);
        
        int next_token = sample_from_logits(logits, seq_len - 1, temperature);
        printf("[generate_text] Sampled token: %d ('%c')\n", next_token, 
               (next_token >= 32 && next_token <= 126) ? (char)next_token : '?');
        fflush(stdout);
        
        sequence[seq_len] = next_token;
        seq_len++;
        
        free_matrix(logits);
        
        if (next_token == 0 || next_token == '\n' || next_token == '.' || 
            next_token < 0 || next_token >= VOCAB_SIZE) {
            printf("[generate_text] Stopping on token %d\n", next_token);
            break;
        }
    }
    
    printf("[generate_text] Generation complete, final seq_len=%d\n", seq_len);
    fflush(stdout);
    
    // Convert tokens back to text
    char* generated_text = (char*)malloc((seq_len + 1) * sizeof(char));
    for (int i = 0; i < seq_len; i++) {
        int token = sequence[i];
        if (token >= 32 && token <= 126) { 
            generated_text[i] = (char)token;
        } else {
            // Replace non-printable with ?
            generated_text[i] = '?'; 
        }
    }
    generated_text[seq_len] = '\0';
    
    printf("[generate_text] Final text: '%s'\n", generated_text);
    fflush(stdout);
    
    free(prompt_tokens);
    free(sequence);
    
    return generated_text;
}

void complete_text(GPT2Model* model, const char* prompt) {
    printf("\nPrompt: \"%s\"\n", prompt);
    
    // Sample to generate text with different temperatures
    float temperatures[] = {0.1f, 0.7f, 1.0f};
    int num_temps = sizeof(temperatures) / sizeof(temperatures[0]);
    
    for (int t = 0; t < num_temps; t++) {
        printf("\nTemperature %.1f:\n", temperatures[t]);
        char* generated = generate_text(model, prompt, 100, temperatures[t]);
        printf("Generated: \"%s\"\n", generated);
        free(generated);
    }
}