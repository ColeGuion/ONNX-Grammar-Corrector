#ifndef UTILS_H
#define UTILS_H

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <json-c/json.h>
#include "onnxruntime_c_api.h"
#include "../inference.h"
#include "../Llm/infer.h"
#include "logger.h"

extern const OrtApi* g_ort;


// Print out an array
void printArr(char* label, int64_t arr[], int size);

// Write timing results to 'times.txt'
void writeTimedResults(int num_timers, double allTimes[], const char** time_names);
// Appends text to a .txt file
void writeToFile(char* filename, size_t testNum, char* text);

char* getEnumType(char* ortType, int type); // Get name of different ORT enum types
void GetOrtType(Geco *geco, OrtValue* tensor);
void getProviderOpts(Geco* geco);   // Print CUDA Provider Options
void ListProviders(Geco *geco);     // List all available providers
void GetCudaOptions(Geco *geco, OrtCUDAProviderOptionsV2* cuda_options);
// Get the Input / Output data type for the models session. Returns a number to the data type value
void GetSessionDataType(Geco *geco, OrtSession *session);
void GetSessionDataType2(Llm *llm, OrtSession *session);
// Get information from allocator
void GetMemoryInfo(Geco* geco, OrtAllocator* myAlloc);  
// Get the details from a OrtValue tensor
void GetTensorDetails(Geco *geco, char* tensor_name, OrtValue* tensor);


// Memory usage
int get_memstat(char* stat);
void get_memstat2();
void print_memstat(char* stat);
void getMem(char* label);

// Memory usage in MB
void get_memory_usage();
void print_memory_usage();
void print_malloc_stats();


#endif // UTILS_H