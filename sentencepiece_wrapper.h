#ifndef SENTENCEPIECE_WRAPPER_H
#define SENTENCEPIECE_WRAPPER_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "MoreBin/logger.h"

// Constants
#define LOGIT_SIZE 32128    // Logit Tensor Shape = BatchSize x 1 x 32128
#define GIBB_CLASSES 4      // Clean, Mild Gibberish, Word Salad, Noise
#define MAX_TOKENS 100      // Maximum sequence length allowed (NOTE: No safety bounds are in place to enforce or set this limit) (Maybe prepare_texts function should be updated to handle this)
#define MAX_BATCH_SIZE 500  // Maximum batch size allowed (NOTE: No safety bounds are in place to enforce or set this limit)
// Types
typedef struct {
    int64_t ids[MAX_BATCH_SIZE*MAX_TOKENS];
    int64_t attention_mask[MAX_BATCH_SIZE*MAX_TOKENS];
    int64_t shape[2];
    int length;
} Tokenized_WP_Output;

typedef struct {
    int64_t ids[MAX_BATCH_SIZE*MAX_TOKENS];             // Array of token IDs
    int64_t attention_mask[MAX_BATCH_SIZE*MAX_TOKENS];  // Attention mask marking non-padding tokens
    int64_t shape[2];
    size_t data_len;
    int newline_size;                   // Number of newline strings
    int newline_inds[MAX_BATCH_SIZE];   // Indicies of the newline strings in the full array of texts
    void* newline_strs;
} TokenizedTexts;

#ifdef __cplusplus
extern "C" {
#endif
void* initialize_processor(const char* model_path);

// Groups the texts into strings less than total max_tokens. Then tokenize those strings and make them padded to the same length
TokenizedTexts* prepare_texts(void* processor_ptr, char** texts, int num_texts);//, int max_tokens);

// Decode token IDs into texts and combines them with newline string into a single string
char* decode_texts(void* processor_ptr, int decoded_ids[MAX_BATCH_SIZE][MAX_TOKENS], TokenizedTexts* tokensObj);     //char* decode_texts(void* processor_ptr, int** decoded_ids, int num_texts, int num_ids, int newLn_size, char **newLn_strs, int *newLn_inds);

//TODO: Can still be optimized like the prepare_texts()
Tokenized_WP_Output batch_gibb_texts(const char* gibb_tok_path, char** texts, int batchSize);

void free_processor(void* processor_ptr);
void free_tokenized_texts(TokenizedTexts* obj);
void free_tokenized_wp_output(Tokenized_WP_Output output);

// Debugging
void encodeText(void* processor_ptr, char* text);
//void* prepare_texts(void* processor_ptr, char** texts, int num_texts, int max_tokens);


#ifdef __cplusplus
}
#endif

#endif // SENTENCEPIECE_WRAPPER_H
