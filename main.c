#include "membox.h"
#include "inference.h"
#include "Llm/infer.h"
#include "MoreBin/utils.h"
#include "MoreBin/my_tester.h"
#include "MoreBin/logger.h"
#include "onnxruntime_c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <malloc.h>

int numTests = -1;   // Default # of tests to run
int enableGpu = 0;   // CPU=0, GPU=1
int myLogLev = DEBUG;   // (0)Debug, (1)Info, (2)Warning, (3)Error, (4)Critical


void tmp() {
    UseGeco();
    malloc_trim(0);
    getMem("After Malloc");
}

void mainGeco() {
    getMem("Start of Program");
    clock_t t1 = clock();
    GecoConfig config = {false, 2, "0", false}; // {UseGpu, LogLevel, GpuId, doProfile}
    void* geco = NewGeco(config);
    if (geco == NULL) {
        fprintf(stderr, "Failed to create geco object!\n");
        return;
    }
    getMem("Geco Created");

    int selection = 10;
    switch (selection) {
        case 0:     // 0) TestCase2 tests
            test_gec(geco, 0);
            break;
        case 1:     // 1) Speed Tests
            runSpeedTests(geco, numTests, enableGpu);
            break;
        case 2:     // 2) Gibberish tests
            test_gibbs(geco, 1);
            break;
        case 3:     // 3) All Gibberish tests
            test_allGibbs(geco, numTests);
            break;
        case 4:     // 4) Encoding tests
            EncText(geco, "\\x1b[32mAdditional Sentence.");
            break;
        case 5:     // 5) Full Run
            speedTest(geco, 1, enableGpu);
            test_gibbs(geco, 1);
            break;
        case 6:
            test_gec2(geco, 4);
            break;
        case 7:
            test_gec3(geco, 6);
            test_gec3(geco, 7);
            test_gec3(geco, 8);
            test_gec3(geco, 9);
            test_gec3(geco, 10);
            test_gec3(geco, 11);
            test_gec3(geco, 12);
            break;
        case 8:
            test_gec3(geco, 6);     // Batch: 58
            test_gec3(geco, 6);
            test_gec3(geco, 12);    // Batch: 10
            test_gec3(geco, 7);     // Batch: 37
            test_gec3(geco, 8);     // Batch: 25
            test_gec3(geco, 6);
            test_gec3(geco, 10);    // Batch: 5
            PrintMemBox();
            malloc_trim(0);
            getMem("After Malloc");
            break;
        case 9:
            // 10 runs
            test_gec3(geco, 6);     // Batch: 58
            test_gec3(geco, 6);
            /* test_gec3(geco, 6);
            test_gec3(geco, 6);
            test_gec3(geco, 6);
            test_gec3(geco, 6);
            test_gec3(geco, 6);
            test_gec3(geco, 6);
            test_gec3(geco, 6);
            test_gec3(geco, 6); */
            PrintMemBox();
            malloc_trim(0);
            getMem("After Malloc");
            break;
        case 10:
            //tmp();
            test_gec3(geco, 6);
            PrintCheckpoints();
            break;
        default:
            break;
    }


    FreeGeco(geco); // Free the geco object
    geco = NULL;
    getMem("Geco Freed");
    malloc_trim(0);
    getMem("After Malloc");

    clock_t t2 = clock();
    double total_time = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("\nExec Time: %.2f\n", total_time);

}

void mainMistral() {
    void* llm = NewLlm(enableGpu, 0);
    if (llm == NULL) {
        fprintf(stderr, "Failed to create LLM object!\n");
        return;
    }

    LlmRun(llm, "Hello world!");

    FreeLlm(llm);
}

int main(int argc, char *argv[]) {
    char* runType = "geco";
    if (argc >= 2) {
        numTests = atoi(argv[1]);
        printf("Number of Tests: %d\n\n", numTests);
        
        // Set run type
        if (argc >= 3) runType = argv[2];

        // Set log level
        if (argc >= 4) myLogLev = atoi(argv[3]);

        // Use CPU or GPU
        if (argc >= 5) {
            if (atoi(argv[4]) == 0 || atoi(argv[4]) == 1) {
                enableGpu = atoi(argv[4]);  // Use the CPU/GPU
            }
        }
    }
    LOG_LEVEL = myLogLev;

    (runType[0] == 'm') ? mainMistral() : mainGeco();

    return 0;
}

