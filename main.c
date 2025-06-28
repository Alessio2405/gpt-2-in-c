#include "gpt2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LEN 256
#define MAX_TOKENS 128

// Dummy tokenizer: one token per character (for demo)
int* tokenize_text(const char* text, int* seq_len) {
    int len = strlen(text);
    int* tokens = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    *seq_len = len;
    return tokens;
}

int main() {
    srand(time(NULL)); 
    
    FILE* f = fopen("train.txt", "r");
    if (!f) {
        printf("Could not open train.txt\n");
        return 1;
    }
    
    char line[MAX_LINE_LEN];
    int* batches[16];
    int* targets[16];
    int num_batches = 0;
    int seq_len = 0;
    
    printf("Loading training data...\n");
    while (fgets(line, sizeof(line), f) && num_batches < 16) {
        int len = 0;
        int* tokens = tokenize_text(line, &len);
        if (len < 2) {
            free(tokens);
            continue;
        }
        
        batches[num_batches] = (int*)malloc((len-1) * sizeof(int));
        targets[num_batches] = (int*)malloc((len-1) * sizeof(int));
        
        for (int i = 0; i < len-1; i++) {
            batches[num_batches][i] = tokens[i];
            targets[num_batches][i] = tokens[i+1];
        }
        seq_len = len-1;
        free(tokens);
        num_batches++;
    }
    fclose(f);
    
    printf("Loaded %d training batches\n", num_batches);

    printf("\nCreating GPT-2 model...\n");
    GPT2Model* model = create_gpt2_model();

    printf("\nStarting training...\n");
    fflush(stdout);
    // Obviously this is only for demonstration purpose, you should have a correct number of epochs to train the model
    for (int epoch = 0; epoch < 3; epoch++) {
        printf("\n=== Epoch %d ===\n", epoch+1);
        train_epoch(model, batches, targets, num_batches, seq_len);
    }

    printf("\nTRAINING COMPLETE - GENERATING TEXT\n");

    const char* test_prompts[] = {
        "Altherya",
        "Il",
        "Cuore",
        ""  
    };
    
    int num_prompts = sizeof(test_prompts) / sizeof(test_prompts[0]);
    
    for (int i = 0; i < num_prompts; i++) {
        complete_text(model, test_prompts[i]);
        printf("\n\n\n\n\n\n\n\n");
    }
    
    printf("\nInteractive mode (type 'quit' to exit):\n");
    char user_input[256];
    while (1) {
        printf("\nEnter prompt: ");
        if (!fgets(user_input, sizeof(user_input), stdin)) break;
        
        user_input[strcspn(user_input, "\n")] = 0;
        
        if (strcmp(user_input, "quit") == 0) break;
        
        char* generated = generate_text(model, user_input, 150, 1.0f);
        printf("Generated: \"%s\"\n", generated);
        free(generated);        
    }

    printf("\nCleaning up...\n");
    for (int i = 0; i < num_batches; i++) {
        free(batches[i]);
        free(targets[i]);
    }
    free_gpt2_model(model);
    
    printf("Done!\n");
    return 0;
}