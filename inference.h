#ifndef INFERENCE_H
#define INFERENCE_H
#include <assert.h>
#include <json-c/json.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <malloc.h>
#include "sentencepiece_wrapper.h"
#include "onnxruntime_c_api.h"
#include "MoreBin/logger.h"
#include "membox.h"

// Constants
extern bool USE_GPU;

// Decoder Input/Output Names
extern char* decoder_output_names[51];
extern char* decPast_input_names[51];
extern char* decPast_output_names[25]; 

// G.E.C.O. => Grammar Error Corrector Onnx
typedef struct {
    char device_id[8];
    const OrtApi* g_ort;
    OrtEnv *env;
    OrtRunOptions* run_options;
    OrtSessionOptions *session_options;
    OrtCUDAProviderOptionsV2* cuda_options;
    OrtMemoryInfo* memory_info;
    OrtArenaCfg* arena_cfg;

    // Model Sessions
    OrtSession *encoder_session;
    OrtSession *decoder_session;
    OrtSession *decPast_session;
    OrtSession *gibb_session;

    // Allocators
    OrtAllocator *enc_allocator;
    OrtAllocator *dec_allocator;
    OrtAllocator *decPast_allocator;
    OrtAllocator *gibb_allocator;

    // IO Bindings
    OrtIoBinding *enc_io_binding;
    OrtIoBinding *dec_io_binding;
    OrtIoBinding *decPast_io_binding;
    OrtIoBinding *gibb_io_binding;

    // Generated Tokens
    int generated_tokens[MAX_BATCH_SIZE][MAX_TOKENS];  // Array of generated tokens for each sequence in the batch

    // Number of Runs Counter
    int runCounter;

    // SentencePiece Utilities
    void* processor;
} Geco;

typedef struct {
    bool useGpu;
    int OrtLogLevel;    // (0: VERBOSE, 1: INFO, 2: WARNING, 3: ERROR, 4: FATAL)
    char* gpuId;
    bool doProfile;
} GecoConfig;

// Abort the program if an error occurs
#define ORT_ABORT_ON_ERROR(geco, expr)                                   \
    do {                                                                 \
        OrtStatus *onnx_status = (expr);                                 \
        if (onnx_status != NULL) {                                       \
            const char *msg = geco->g_ort->GetErrorMessage(onnx_status); \
            fprintf(stderr, "ERROR: %s\n", msg);                         \
            geco->g_ort->ReleaseStatus(onnx_status);                     \
            abort();                                                     \
        }                                                                \
    } while (0);

// Goto the cleanup label if an error occurs
#define ORT_CLEAN_ON_ERROR(cleanup_label, geco, expr)                     \
    do {                                                                    \
        OrtStatus *onnx_status = (expr);                                    \
        if (onnx_status != NULL) {                                          \
            const char *msg = geco->g_ort->GetErrorMessage(onnx_status);    \
            fprintf(stderr, "ERROR in '%s()': %s\n", __func__, msg);        \
            geco->g_ort->ReleaseStatus(onnx_status);                        \
            goto cleanup_label;                                             \
        }                                                                   \
    } while (0);


void UseGeco();
void free_tensor(Geco* geco, OrtValue** tensor);

// Function to load configuration from a JSON file
int load_config(const char *filename);

// Initialize a new GECO instance
void* NewGeco(GecoConfig config);

// Frees all of the allocated memory used in this GECO object
void FreeGeco(void* geco);

// Free the memory allocated by the array of ORT output values
void cleanBoundOutputs(Geco* geco, OrtValue*** output_tensors, size_t output_len);

// Checks if last 6 tokens in an array are the same or alternating between two distinct values
bool checkRepeating(int* arr, int lastInd);

// Find the most probable tokens for each sequence
int getMaxTokens(Geco* geco, OrtValue* logits, int64_t* newTokens, int batchSize, int* completed_sequences, int runNum);

// Run the Encoder Model 
OrtValue* runEncoder(Geco* geco, OrtValue* Ort_AttnMask, int64_t* tokenIds, int64_t dataShape[2]);  //OrtValue* runEncoder(Geco* geco, OrtValue* Ort_AttnMask, TokenizedTexts tokenized_obj);
//void runEncoder(Geco* geco, OrtValue** lastHiddenState, OrtValue* Ort_AttnMask, int64_t* tokenIds, int64_t dataShape[2]);

// Recursively run the Decoder with past model
void runPast(Geco* geco, int runNum, int64_t* nextToks, int batchSize, int* completed_sequences);

// Run the both decoder models
void runDecoders(Geco* geco, OrtValue* lastHiddenState, OrtValue* Ort_AttnMask, int batchSize);


// Execute inference using the Geco context
char* GecoRun(void* context, char** texts, int num_texts);
char* InferModel(Geco* geco, char** texts, int num_texts);

// Execute gibberish detection using the Geco context
void GecoGibb(void* context, double probs[MAX_BATCH_SIZE][GIBB_CLASSES], char** texts, int num_batches);
void InferGibb(Geco* geco, double probs[MAX_BATCH_SIZE][GIBB_CLASSES], char** texts, int num_batches);

void EncText(void* ctx, char* text);

#endif // INFERENCE_H