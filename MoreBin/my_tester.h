#ifndef MY_TESTER_H
#define MY_TESTER_H
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <json-c/json.h>
#include "onnxruntime_c_api.h"
#include "../inference.h"


typedef struct {
    int64_t shape[2];
    char** texts;
    int** ids;
    int** attn_mask;
    double** exp_scores;
} TestGibb;

typedef struct {
    int batchSize;
    int maxTokens;
    char** origTexts;
    char** exp_texts;
    int exp_num_texts;
    int exp_newLn_size;
    char** exp_newLn_strs;
    int* exp_newLn_inds;
} PreProcessingTests;

typedef struct {
    int64_t shape[2];
    char** texts;
    int64_t* ids;
    int64_t* attn_mask;
} TestWP;


typedef struct {
    int batchSize; 
    double avgTime;
    double avgTimeGpu;
    char** texts;
    char** expect;
    char* exp_text;
} TestCase;
typedef struct {
    int batchSize;
    char** texts;
    char* exp;
} TestGec;


void speedTest(void* gecoObj, int testNum, int enableGpu);
void runSpeedTests(void* geco, int numTests, int enableGpu);
void customRun(void* gecoObj, char** allTexts, int batchSize);  // Run your own GEC test and time the output

void prtResult(char* result, char* expect, double timeTaken);
void test_gec(void* geco, bool runAll);
void test_gec2(void* geco, int n);
void test_gec3(void* geco, int testNum);

//* Preprocessing Tests
void prtNewlineLits(char* str);

//* Gibberish Tests
void compare_gibb_results(double res[MAX_BATCH_SIZE][GIBB_CLASSES], double** exp, int batchSize);
void test_gibbs(void* geco, int testNum);
void test_allGibbs(void* geco, int numTests);

//* WordPiece Tokenizer Tests
int compTokenOuts(Tokenized_WP_Output res, TestWP exp);


#endif // MY_TESTER_H
