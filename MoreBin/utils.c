#include "utils.h"  // Include the header file

// Data Types for enumeration
const char *tensorDataTypes[] = {"UNDEFINED", "FLOAT", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "BOOL", "FLOAT16", "DOUBLE", "UINT32", "UINT64", "COMPLEX64", "COMPLEX128", "BFLOAT16", "FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ", "UINT4", "INT"};
const char *onnxTypes[] = {"ONNX_TYPE_UNKNOWN", "ONNX_TYPE_TENSOR", "ONNX_TYPE_SEQUENCE", "ONNX_TYPE_MAP", "ONNX_TYPE_OPAQUE", "ONNX_TYPE_SPARSETENSOR", "ONNX_TYPE_OPTIONAL"};


// Print out an array
void printArr(char* label, int64_t arr[], int size) {
    printf("%s: [", label);
    for (int i = 0; i < size; i++) {
        printf("%ld, ", arr[i]);
    }
    printf("\b\b]\n");
}

// Write timing results to 'times.txt'
void writeTimedResults(int num_timers, double allTimes[], const char** time_names) {
    // Print and clear the time values
    FILE *tmFile = fopen("times.txt", "a");
    if (tmFile == NULL) {
        Log(ERROR, "Failed to open times.txt for writing\n");
        return;
    } 
    fprintf(tmFile, "Timed Results: {");
    for (int i=0; i<num_timers; i++) {
        if (allTimes[i] != 0) {
            fprintf(tmFile, "'%s': %.5f, ", time_names[i], allTimes[i]);
        }
    }
    if (tmFile != NULL) {
        fprintf(tmFile, "}\n");
        fclose(tmFile);
    }
}

// Appends text to a .txt file
void writeToFile(char* filename, size_t testNum, char* text) {
    FILE *resultsFile = fopen(filename, "a");
    if (resultsFile != NULL) {
        fprintf(resultsFile, "Test #%zu Result: \"%s\"\n\n", testNum, text);
        fclose(resultsFile);
    } else {
        Log(ERROR, "Failed to open %s for writing", filename);
    }
}


// Get name of different ORT enum types
char* getEnumType(char* ortType, int type) {
    if (strcmp(ortType, "OrtMemType") == 0) {
        switch (type) {
        case -2:
            return "CPU Input";
        case -1:
            return "CPU Output";
        case 0:
            return "Default";
            //break;
        default:
            return "Unknown";
        }
    } else if (strcmp(ortType, "OrtAllocatorType") == 0) {
        switch (type) {
        case -1:
            return "Invalid Allocator";
        case 0:
            return "Device Allocator";
        case 1:
            return "Arena Allocator";
        default:
            return "Unknown";
        }
    } else if (strcmp(ortType, "OrtMemoryInfoDeviceType") == 0) {
        switch (type) {
        case 0:
            return "CPU";
        case 1:
            return "GPU";
        case 2:
            return "FPGA";
        default:
            return "Unknown";
        }
    }
    return "Unknown";
}

void GetOrtType(Geco *geco, OrtValue* tensor) {
    enum ONNXType outTypeIdx;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetValueType(tensor, &outTypeIdx));
    printf("Output Type [%d]: \"%s\"\n", outTypeIdx, onnxTypes[outTypeIdx]);
}

// Print CUDA Provider Options
void getProviderOpts(Geco* geco) {
    char* cuda_options_str = NULL;
    ORT_CLEAN_ON_ERROR(leave_func, geco, geco->g_ort->GetCUDAProviderOptionsAsString(geco->cuda_options, geco->enc_allocator, &cuda_options_str));
    Log(DEBUG, "  - CUDA Provider Options:\n");
    
    // Print the options delimited by ';'
    char* token = strtok(cuda_options_str, ";");
    while (token != NULL) {
        Log(DEBUG, "    > \"%s\"\n", token);
        token = strtok(NULL, ";");
    }

    if (token) free(token);
    if (cuda_options_str) {
        ORT_CLEAN_ON_ERROR(leave_func, geco, geco->g_ort->AllocatorFree(geco->enc_allocator, cuda_options_str));
        cuda_options_str = NULL;
    }

    leave_func:
    return;
}

// List all available providers
void ListProviders(Geco *geco) {
    int provider_length;
    char** providers;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetAvailableProviders(&providers, &provider_length));

    /* # of Providers: 3
        - TensorrtExecutionProvider
        - CUDAExecutionProvider
        - CPUExecutionProvider
    */
    printf("# of Providers: %d\n", provider_length);
    for (int i=0; i<provider_length; i++) {
        printf("  - Provider #%d: %s\n", i, providers[i]);
    }
    printf("\n");
}

void GetCudaOptions(Geco *geco, OrtCUDAProviderOptionsV2* cuda_options) {
    OrtAllocator *cuda_allocator = NULL;
    char *cuda_options_str = NULL;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetAllocatorWithDefaultOptions(&cuda_allocator));
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetCUDAProviderOptionsAsString(cuda_options, cuda_allocator, &cuda_options_str));
    printf("Cuda Options: \"%s\"\n\n", cuda_options_str);
    geco->g_ort->ReleaseAllocator(cuda_allocator);
}

void GetSessionDataType(Geco *geco, OrtSession *session) {
    size_t input_count, output_count;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->SessionGetInputCount(session, &input_count));
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->SessionGetOutputCount(session, &output_count));

    // Allocate
    OrtAllocator *allocator;
    OrtMemoryInfo *memory_info;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->CreateAllocator(session, memory_info, &allocator)); 

    // Loop through Inputs
    printf("Input Data Types (%zu):\n", input_count);
    for (size_t i = 0; i < input_count; i++) {
        char *input_name;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->SessionGetInputName(session, i, allocator, &input_name));

        OrtTypeInfo* input_type_info;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->SessionGetInputTypeInfo(session, i, &input_type_info));

        // Extract TensorTypeAndShapeInfo
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->CastTypeInfoToTensorInfo(input_type_info, &tensor_info));

        // Get data type
        ONNXTensorElementDataType input_type;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetTensorElementType(tensor_info, &input_type));

        int input_type_index = (int)input_type;
        printf("  - \x1b[1;36m%s\x1b[0m = '\x1b[1;33m%s\x1b[0m'\n", tensorDataTypes[input_type_index], input_name);

        /* // Get output shape
        size_t num_dims;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetDimensionsCount(tensor_info, &num_dims));
        int64_t* input_shape = (int64_t*)malloc(num_dims * sizeof(int64_t));
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetDimensions(tensor_info, input_shape, num_dims));
        printArr("  > Shape", input_shape, num_dims); */

        // Clean up
        geco->g_ort->ReleaseTypeInfo(input_type_info);
    }
    printf("\n");

    // Loop through Outputs
    printf("Output Data Types (%zu):\n", output_count);
    for (size_t i = 0; i < output_count; i++) {
        if (i >= 5 && i < output_count-2) {
            if (i == 5) {
                printf("  ...\n");
            }
            continue;
        }

        char *output_name;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->SessionGetOutputName(session, i, allocator, &output_name));

        OrtTypeInfo* output_type_info;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->SessionGetOutputTypeInfo(session, i, &output_type_info));

        // Extract TensorTypeAndShapeInfo
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->CastTypeInfoToTensorInfo(output_type_info, &tensor_info));

        // Get data type
        ONNXTensorElementDataType output_type;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetTensorElementType(tensor_info, &output_type));
        printf("  - \x1b[1;36m%s\x1b[0m = '\x1b[1;33m%s\x1b[0m'\n", tensorDataTypes[(int)output_type], output_name);

        /* // Get output shape
        size_t num_dims;
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetDimensionsCount(tensor_info, &num_dims));
        int64_t* output_shape = (int64_t*)malloc(num_dims * sizeof(int64_t));
        ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetDimensions(tensor_info, output_shape, num_dims));
        printArr("  > Shape", output_shape, num_dims);
        free(output_shape); */

        // Clean up
        geco->g_ort->ReleaseTypeInfo(output_type_info);
    }
    printf("\n");
}

void GetSessionDataType2(Llm *llm, OrtSession *session) {
    size_t input_count, output_count;
    ORT_ABORT_ON_ERROR(llm, llm->g_ort->SessionGetInputCount(session, &input_count));
    ORT_ABORT_ON_ERROR(llm, llm->g_ort->SessionGetOutputCount(session, &output_count));

    // Allocate
    OrtAllocator *allocator;
    OrtMemoryInfo *memory_info;
    ORT_ABORT_ON_ERROR(llm, llm->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    ORT_ABORT_ON_ERROR(llm, llm->g_ort->CreateAllocator(session, memory_info, &allocator)); 

    // Loop through Inputs
    printf("Input Data Types (%zu):\n", input_count);
    for (size_t i = 0; i < input_count; i++) {
        if (i >= 5 && i < input_count-2) {
            if (i == 5) {
                printf("  ...\n");
            }
            continue;
        }
        char *inp_name;
        OrtTypeInfo* inp_type_info;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType inp_type;

        ORT_ABORT_ON_ERROR(llm, llm->g_ort->SessionGetInputName(session, i, allocator, &inp_name));
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->SessionGetInputTypeInfo(session, i, &inp_type_info));
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->CastTypeInfoToTensorInfo(inp_type_info, &tensor_info));   // TensorTypeAndShapeInfo
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->GetTensorElementType(tensor_info, &inp_type));    // Get data type

        int inp_type_index = (int)inp_type;
        printf("  - \x1b[1;36m%s\x1b[0m = '\x1b[1;33m%s\x1b[0m'\n", tensorDataTypes[inp_type_index], inp_name);

        /* // Get output shape
        size_t num_dims;
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->GetDimensionsCount(tensor_info, &num_dims));
        int64_t* input_shape = (int64_t*)malloc(num_dims * sizeof(int64_t));
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->GetDimensions(tensor_info, input_shape, num_dims));
        printArr("  > Shape", input_shape, num_dims); */

        // Clean up
        llm->g_ort->ReleaseTypeInfo(inp_type_info);
    }
    printf("\n");

    // Loop through Outputs
    printf("Output Data Types (%zu):\n", output_count);
    for (size_t i = 0; i < output_count; i++) {
        if (i >= 5 && i < output_count-2) {
            if (i == 5) {
                printf("  ...\n");
            }
            continue;
        }

        char *output_name;
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->SessionGetOutputName(session, i, allocator, &output_name));

        OrtTypeInfo* output_type_info;
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->SessionGetOutputTypeInfo(session, i, &output_type_info));

        // Extract TensorTypeAndShapeInfo
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->CastTypeInfoToTensorInfo(output_type_info, &tensor_info));

        // Get data type
        ONNXTensorElementDataType output_type;
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->GetTensorElementType(tensor_info, &output_type));
        printf("  - \x1b[1;36m%s\x1b[0m = '\x1b[1;33m%s\x1b[0m'\n", tensorDataTypes[(int)output_type], output_name);

        /* // Get output shape
        size_t num_dims;
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->GetDimensionsCount(tensor_info, &num_dims));
        int64_t* output_shape = (int64_t*)malloc(num_dims * sizeof(int64_t));
        ORT_ABORT_ON_ERROR(llm, llm->g_ort->GetDimensions(tensor_info, output_shape, num_dims));
        printArr("  > Shape", output_shape, num_dims);
        free(output_shape); */

        // Clean up
        llm->g_ort->ReleaseTypeInfo(output_type_info);
    }
    printf("\n");
}

// Get information from allocator
void GetMemoryInfo(Geco* geco, OrtAllocator* myAlloc) {
    // Get memory information
    const OrtMemoryInfo* memInfo = NULL;
    //memInfo = geco->memory_info;
    //OrtMemoryInfo* memInfo = geco->enc_allocator->Info(myAlloc);
    ORT_CLEAN_ON_ERROR(mem_cleanup, geco, geco->g_ort->AllocatorGetInfo(myAlloc, &memInfo));

    const char* name;
    OrtMemType mem_type;
    int id;
    OrtAllocatorType alloc_type;              // (-1)OrtInvalidAllocator, (0)OrtDeviceAllocator, (1)OrtArenaAllocator
    OrtMemoryInfoDeviceType dev_type;    // (0)CPU, (1)GPU, (2)FPGA
    char* deviceTypes[] = {"CPU", "GPU", "FPGA"};

    geco->g_ort->MemoryInfoGetDeviceType(memInfo, &dev_type);
    
    ORT_CLEAN_ON_ERROR(mem_cleanup, geco, geco->g_ort->MemoryInfoGetName(memInfo, &name));
    ORT_CLEAN_ON_ERROR(mem_cleanup, geco, geco->g_ort->MemoryInfoGetMemType(memInfo, &mem_type));
    ORT_CLEAN_ON_ERROR(mem_cleanup, geco, geco->g_ort->MemoryInfoGetId(memInfo, &id));
    ORT_CLEAN_ON_ERROR(mem_cleanup, geco, geco->g_ort->MemoryInfoGetType(memInfo, &alloc_type));

    Log(DEBUG, "Memory Info:");
    Log(DEBUG, "%12s %s", "Name:", name);
    Log(DEBUG, "%12s %d", "ID:", id);
    Log(DEBUG, "%12s %s", "Memory:", getEnumType("OrtMemType", mem_type));
    Log(DEBUG, "%12s %s", "Alloc:", getEnumType("OrtAllocatorType", alloc_type));
    Log(DEBUG, "%12s %s\n", "Device:", getEnumType("OrtMemoryInfoDeviceType", dev_type));
    /* Log(DEBUG, "  > ID: %d", id);
    Log(DEBUG, "  > Memory Type: %s", getEnumType("OrtMemType", mem_type));

    if (allocType == -1) printf("  > Alloc Type: Invalid Allocator\n");
    else if (allocType == 0) printf("  > Alloc Type: Device Allocator\n");
    else if (allocType == 1) printf("  > Alloc Type: Arena Allocator\n");
    Log(DEBUG, "  > Device Type: %s\n", deviceTypes[dev_type]); */

    mem_cleanup:
    //geco->g_ort->ReleaseMemoryInfo(memInfo);
    return;
    /* geco->g_ort->ReleaseMemoryInfo(memory_info);
    ORT_CLEAN_ON_ERROR(model_run_cleanup, geco, geco->g_ort->AllocatorGetInfo(geco->enc_allocator, &geco->memory_info));
    ORT_CLEAN_ON_ERROR(model_run_cleanup, geco, geco->g_ort->GetAllocatorMemoryInfo(geco->memory_info, &memory_info));

    size_t allocated_memory = 0;
    size_t used_memory = 0;
    ORT_CLEAN_ON_ERROR(model_run_cleanup, geco, geco->g_ort->GetAllocatorMemoryInfo(memory_info, &allocated_memory, &used_memory));

    Log(DEBUG, "Allocated Memory: %zu bytes", allocated_memory);
    Log(DEBUG, "Used Memory: %zu bytes", used_memory);

    geco->g_ort->ReleaseMemoryInfo(memory_info); */
}

// Get the details from a OrtValue tensor
void GetTensorDetails(Geco *geco, char* tensor_name, OrtValue* tensor) {
    printf("\x1b[1;36m\"%s\" Tensor Info:\x1b[0m\n", tensor_name);

    // Get tensor type and shape information
    OrtTensorTypeAndShapeInfo* type_info;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetTensorTypeAndShape(tensor, &type_info));

    size_t num_dims;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetDimensionsCount(type_info, &num_dims));

    int64_t* tensor_shape = (int64_t*)malloc(num_dims * sizeof(int64_t));
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetDimensions(type_info, tensor_shape, num_dims));

    // Extract data type
    ONNXTensorElementDataType tensor_data_type;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetTensorElementType(type_info, &tensor_data_type));
    
    printf("  > Data Type: %s\n", tensorDataTypes[tensor_data_type]);
    printArr("  > Shape", tensor_shape, num_dims);   // Tensor Shape: 5 x 11 x 768

    // Print the first 6 values in the tensor
    float* tensor_data;
    ORT_ABORT_ON_ERROR(geco, geco->g_ort->GetTensorMutableData(tensor, (void**)&tensor_data));

    printf("  > Values: [");
    for (int i = 0; i < 6; i++) {
        printf("%f, ", tensor_data[i]);
    }
    printf("...]\n\n");

    //free(tensor_data);
    free(tensor_shape);
    geco->g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
}


// Disable warnings
#ifdef __linux__
// Get Memory Statistics from '/proc/self/status' (VmRSS: Amount of RAM the process is currently using)
int get_memstat(char* stat) {
    int mem_kb = 0;
    if (strcmp(stat, "RSS") == 0) {
        // RUsage => Maximum resident set size (RSS) used by the process during its lifetime
        // RSS => Amout of physical memory(RAM) currently used by the process
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        mem_kb = usage.ru_maxrss; 
    } else {
        FILE* file = fopen("/proc/self/status", "r");
        if (!file) {
            perror("Error opening /proc/self/status");
            return mem_kb;
        }

        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, stat, strlen(stat)) == 0) {
                char format[64];
                snprintf(format, sizeof(format), "%s: %%d kB", stat);
                sscanf(line, format, &mem_kb);
                break;
            }
        }
        fclose(file);
    }
    return mem_kb;
}

// Get stats from "/proc/self/statm"
void get_memstat2() {
    FILE *file = fopen("/proc/self/statm", "r");
    if (file == NULL) {
        perror("Error opening /proc/self/statm");
        return;
    }

    // Variables to store memory statistics
    unsigned long total_program_size; // in pages
    unsigned long resident_set_size;  // in pages
    unsigned long shared_pages;       // in pages
    unsigned long text_pages;         // in pages
    unsigned long library_pages;      // in pages
    unsigned long data_pages;         // in pages
    unsigned long dirty_pages;        // in pages

    // Read the memory statistics from /proc/self/statm
    if (fscanf(file, "%lu %lu %lu %lu %lu %lu %lu",
               &total_program_size,
               &resident_set_size,
               &shared_pages,
               &text_pages,
               &library_pages,
               &data_pages,
               &dirty_pages) != 7) {
        perror("Error reading memory stats");
        fclose(file);
        return;
    }

    fclose(file);

    // Page size (usually 4 KB on Linux)
    long page_size = sysconf(_SC_PAGESIZE) / 1024; // in KB

    void TO_MB(const char* label, unsigned long pages) {
        float f = (float)((pages * page_size) / 1000.0);
        printf("%20s:  %.1f mb\n", label, f);
    }
    void TO_KB(const char* label, unsigned long pages) {
        float f = pages * page_size;
        printf("%20s:  %.1f kb\n", label, f);
    }


    printf("\x1b[1;32mMemory Usage Statistics:\x1b[0m\n");
    TO_KB("Total Size", total_program_size);
    TO_KB("RSS", resident_set_size);
    TO_KB("Shared Pages", shared_pages);
    TO_KB("Text Pages", text_pages);
    TO_KB("Library Pages", library_pages);
    TO_KB("Data Pages", data_pages);
    TO_KB("Dirty Pages", dirty_pages);
    printf("\n");
}

#else
int get_memstat(char* stat) {
    return 0;
}
void get_memstat2() {
    return;
}
#endif

void print_memstat(char* stat) {
    printf("%8s: %.2f mb\n", stat, (get_memstat(stat) / 1024.0));
    //printf("%8s: %d kb\n", stat, get_memstat(stat));
}

void getMem(char* label) {
    if (strcmp(label, "") != 0) {
        printf("\n\x1b[1;31m======== %s ========\x1b[0m\n", label);
    }
    print_memstat("VmRSS");
    print_memstat("VmData");
    //print_memstat("VmSize");
    //printf("\n");
}

// Function to print memory usage
void print_memory_usage() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) {
        perror("Error opening /proc/self/status");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            //printf("Memory usage (VmRSS): %s\n", line);
            int memory_kb;
            sscanf(line, "VmRSS: %d kB", &memory_kb);
            float memory_mb = memory_kb / 1024.0;
            printf("Memory Usage: %.2f MB (VmRSS)\n", memory_kb / 1024.0);
        }
    }

    fclose(file);
}

// Memory usage in MB
void get_memory_usage() {
    printf("\x1b[4;1;32mMemory Usage\x1b[0m:\n");
    print_memstat("VmRSS");
    print_memstat("VmSize");
    print_memstat("RSS");
    printf("\n");
    print_malloc_stats();
    printf("\n");
}


void print_malloc_stats() {
    // Missing 'max mmap regions' and 'max mmap bytes' from malloc_stats()
    struct mallinfo2 mi = mallinfo2();  // Get memory allocation info

    printf("\x1b[4mMalloc Stats\x1b[0m:\n");
    printf("  (Arena) System:  %.1f mb\n", mi.arena / 1024.0 / 1024.0); // NOTE: Arena + Mmap = Total System Memory
    printf("  (Total) System:  %.1f mb\n", (mi.arena + mi.hblkhd) / 1024.0 / 1024.0);
    printf("          In-use:  %.1f mb\n", (mi.uordblks + mi.hblkhd) / 1024.0 / 1024.0);
    printf("  Allocated Mmap:  %.1f mb\n", mi.hblkhd / 1024.0 / 1024.0);
    printf("     Free Memory:  %.1f mb\n", mi.fordblks / 1024.0 / 1024.0); // NOTE: Free Memory + In-Use = Total System Memory
    printf("     Free Chunks:  %zu\n", mi.ordblks);
    printf("\n");
}


float memHolder[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
char* memLabels[9] = {"VmRSS", "VmSize", "RSS", "(Arena) System", "(Total) System", "In-use", "Allocated Mmap", "Free Memory", "Free Chunks"};
//TODO: Print difference between past value and current vlaue