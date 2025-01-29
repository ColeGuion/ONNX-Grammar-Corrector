#ifndef INFER_H
#define INFER_H
#include <assert.h>
#include <json-c/json.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../MoreBin/logger.h"
#include "../sentencepiece_wrapper.h"
#include "onnxruntime_c_api.h"

// G.E.C.O. => Grammar Error Corrector Onnx
typedef struct {
    const OrtApi* g_ort;
    OrtEnv *env;
    OrtMemoryInfo* memory_info;
    OrtRunOptions* run_options;
    OrtSessionOptions *session_options;
    OrtCUDAProviderOptionsV2* cuda_options;
    OrtArenaCfg* arena_cfg;
    char device_id[8];

    OrtSession *session;
    OrtAllocator *allocator;
    OrtIoBinding *io_binding;

    // Generated Tokens
    int generated_tokens[MAX_BATCH_SIZE][MAX_TOKENS];  // Array of generated tokens for each sequence in the batch

    // SentencePiece Utilities
    void* processor;
} Llm;

// Constants
//static char PATH_MODEL[1000];
//static char PATH_SP_MODEL[1000];

// Decoder Input/Output Names
extern char* model_inpNames[66];
extern char* model_outNames[65]; 

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


int load_conf(const char *filename);
void* NewLlm(int useGpu, int gpuId);
void FreeLlm(void* objPtr);
void cleanBoundOutputs2(Llm* llm, OrtValue** output_tensors, size_t output_len);

TokenizedTexts* doTokenize(Llm* llm, char* text);
void runModel(Llm* llm, TokenizedTexts *myToks, int batchSize);
//void runModel(Llm* llm, TokenizedTexts *tokensObj, OrtValue* Ort_Mask, OrtValue* Ort_IDs);
void LlmRun(void* ctx, char* text);
void InferLlm(Llm* llm, char* text);

#endif // INFER_H