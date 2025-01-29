// infer.c - For Mistral-7b inference
#include "infer.h"
#include "../MoreBin/utils.h"

// Model paths
static char PATH_MODEL[1000];
static char PATH_SP_MODEL[1000];

// Decoder Input/Output Names   // All Float16 values (except ids and mask)
char* model_inpNames[66] = {"input_ids", "attention_mask", "past_key_values.0.key", "past_key_values.0.value", "past_key_values.1.key", "past_key_values.1.value", "past_key_values.2.key", "past_key_values.2.value", "past_key_values.3.key", "past_key_values.3.value", "past_key_values.4.key", "past_key_values.4.value", "past_key_values.5.key", "past_key_values.5.value", "past_key_values.6.key", "past_key_values.6.value", "past_key_values.7.key", "past_key_values.7.value", "past_key_values.8.key", "past_key_values.8.value", "past_key_values.9.key", "past_key_values.9.value", "past_key_values.10.key", "past_key_values.10.value", "past_key_values.11.key", "past_key_values.11.value", "past_key_values.12.key", "past_key_values.12.value", "past_key_values.13.key", "past_key_values.13.value", "past_key_values.14.key", "past_key_values.14.value", "past_key_values.15.key", "past_key_values.15.value", "past_key_values.16.key", "past_key_values.16.value", "past_key_values.17.key", "past_key_values.17.value", "past_key_values.18.key", "past_key_values.18.value", "past_key_values.19.key", "past_key_values.19.value", "past_key_values.20.key", "past_key_values.20.value", "past_key_values.21.key", "past_key_values.21.value", "past_key_values.22.key", "past_key_values.22.value", "past_key_values.23.key", "past_key_values.23.value", "past_key_values.24.key", "past_key_values.24.value", "past_key_values.25.key", "past_key_values.25.value", "past_key_values.26.key", "past_key_values.26.value", "past_key_values.27.key", "past_key_values.27.value", "past_key_values.28.key", "past_key_values.28.value", "past_key_values.29.key", "past_key_values.29.value", "past_key_values.30.key", "past_key_values.30.value", "past_key_values.31.key", "past_key_values.31.value"};
char* model_outNames[65] = {"logits", "present.0.key", "present.0.value", "present.1.key", "present.1.value", "present.2.key", "present.2.value", "present.3.key", "present.3.value", "present.4.key", "present.4.value", "present.5.key", "present.5.value", "present.6.key", "present.6.value", "present.7.key", "present.7.value", "present.8.key", "present.8.value", "present.9.key", "present.9.value", "present.10.key", "present.10.value", "present.11.key", "present.11.value", "present.12.key", "present.12.value", "present.13.key", "present.13.value", "present.14.key", "present.14.value", "present.15.key", "present.15.value", "present.16.key", "present.16.value", "present.17.key", "present.17.value", "present.18.key", "present.18.value", "present.19.key", "present.19.value", "present.20.key", "present.20.value", "present.21.key", "present.21.value", "present.22.key", "present.22.value", "present.23.key", "present.23.value", "present.24.key", "present.24.value", "present.25.key", "present.25.value", "present.26.key", "present.26.value", "present.27.key", "present.27.value", "present.28.key", "present.28.value", "present.29.key", "present.29.value", "present.30.key", "present.30.value", "present.31.key", "present.31.value"};
#define MAX_BTC_SIZE 10
#define MAX_SEQ_LEN 100
#define SEQ_LEN 1   // Single token inference


void cleanBoundOutputs2(Llm* llm, OrtValue** output_tensors, size_t output_len) {
    if (llm == NULL || output_tensors == NULL || output_len == 0) {
        return;
    }
    for (size_t i = 0; i < output_len; i++) {
        if (output_tensors[i] != NULL) {
            llm->g_ort->ReleaseValue(output_tensors[i]);
        }
    }
    output_tensors = NULL;
}

int load_conf(const char *filename) {
    // Open the JSON file
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open configuration file");
        return -1;
    }

    // Read and parse the JSON file
    struct json_object *parsed_json;
    parsed_json = json_object_from_file(filename);
    if (!parsed_json) {
        fprintf(stderr, "Error parsing JSON file\n");
        fclose(file);
        return -1;
    }

    // Assign JSON values to variables   
    struct json_object *path_onnx;
    char onnxPath[200];

    if (json_object_object_get_ex(parsed_json, "llm_model_path", &path_onnx)) 
        strcpy(onnxPath, json_object_get_string(path_onnx));    //onnxPath = strdup(json_object_get_string(path_onnx));

    snprintf(PATH_MODEL, sizeof(PATH_MODEL), "%s%s", onnxPath, "mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32.onnx");
    snprintf(PATH_SP_MODEL, sizeof(PATH_SP_MODEL), "%s%s", onnxPath, "tokenizer.model");

    json_object_put(parsed_json);
    fclose(file);
    return 0;
}

void* NewLlm(int useGpu, int gpuId) {
    Log(DEBUG, "Initializing a new LLM object...");
    if (load_conf("MoreBin/config.json") != 0) {
        fprintf(stderr, "Failed to load config!\n");
        return NULL;
    }

    // Allocate memory for LLM object
    Llm* llm = (Llm*)malloc(sizeof(Llm));
    if (llm == NULL) {
        Log(ERROR, "Failed to allocate memory for LLM object!");
        return NULL;
    }

    // Set all fields to 0 or NULL
    memset(llm, 0, sizeof(Llm));

    // Set device_id
    if (useGpu) {
        snprintf(llm->device_id, sizeof(llm->device_id), "gpu:%d", gpuId);
    } else {
        snprintf(llm->device_id, sizeof(llm->device_id), "cpu");
    }
    Log(DEBUG, "  - Device: '%s'", llm->device_id);

    // Get the ONNX Runtime API handle
    llm->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (llm->g_ort == NULL) {
        Log(ERROR, "Failed to get ONNX Runtime API handle!");
        free(llm);
        llm = NULL;
        return NULL;
    }

    // Create an ONNX Runtime environment (ORT_LOGGING_LEVEL_WARNING)
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &llm->env));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &llm->memory_info)); // Memory Info (For CPU in this case)
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateSessionOptions(&llm->session_options)); // Session Options
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateRunOptions(&llm->run_options));         // Run Options 

    // Create Arena Config for GPU / CPU
    if (useGpu) { 
        Log(DEBUG, "  - Setting up the GPU Arena Configurations... ");
        // Enable CUDA for GPU acceleration
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateCUDAProviderOptions(&llm->cuda_options));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->SetCurrentGpuDeviceId(gpuId)); // Set GPU device ID
        
        char idStr[5];
        snprintf(idStr, sizeof(idStr), "%d", gpuId);
        const char* provider_keys[] = {"device_id"};
        const char* provider_values[] = {idStr};
        Log(DEBUG, "ID String: '%s'\n", idStr);
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->UpdateCUDAProviderOptions(llm->cuda_options, provider_keys, provider_values, 1));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(llm->session_options, llm->cuda_options));

        // Create arena based allocator for GPU     (From 'ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage' in test_inference.cc example)
        const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes"};
        const size_t values[] = {0, 0, 1024, 0, 256};   // let ort pick default max memory
        size_t num_keys = sizeof(keys) / sizeof(keys[0]);
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateArenaCfgV2(keys, values, num_keys, &llm->arena_cfg));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->AddRunConfigEntry(llm->run_options, "memory.enable_memory_arena_shrinkage", llm->device_id));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateAndRegisterAllocatorV2(llm->env, "CUDAExecutionProvider", llm->memory_info, llm->arena_cfg, NULL, NULL, 0));
        
        // Performs asynchronous copies while running inference
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->AddRunConfigEntry(llm->run_options, "disable_synchronize_execution_providers", "1"));  //* Improves gpu performance
    } else {
        Log(DEBUG, "  - Setting up the CPU Arena Configurations... ");

        const char* keys[] = {"initial_chunk_size_bytes", "arena_extend_strategy", "max_dead_bytes_per_chunk"};
        const size_t values[] = {(1024*1024*10), 0, 256};   // 10MB initial chunk size, kNextPowerOfTwo, & Maximum 256 dead bytes allowed per chunk
        size_t num_keys = sizeof(keys) / sizeof(keys[0]);
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateArenaCfgV2(keys, values, num_keys, &llm->arena_cfg));

        // Shared Allocator
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateAndRegisterAllocatorV2(llm->env, "CPUExecutionProvider", llm->memory_info, llm->arena_cfg, NULL, NULL, 0));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->SetIntraOpNumThreads(llm->session_options, 1));   // Huge speed improvement for CPU execution
    }
    
    // Session/Run Options
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->AddSessionConfigEntry(llm->session_options, "session.use_env_allocators", "1"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->AddRunConfigEntry(llm->run_options, "memory.enable_memory_arena_shrinkage", "cpu:0"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->SetSessionGraphOptimizationLevel(llm->session_options, ORT_ENABLE_ALL));   // Enable all graph optimizations
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->AddSessionConfigEntry(llm->session_options, "session.dynamic_block_base", "4"));       //* Improves gpu performance
    

    // Initialize Allocators & Sessions
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateSession(llm->env, PATH_MODEL, llm->session_options, &llm->session));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateAllocator(llm->session, llm->memory_info, &llm->allocator));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, llm, llm->g_ort->CreateIoBinding(llm->session, &llm->io_binding));

    // Load the SentencePiece model
    llm->processor = initialize_processor(PATH_SP_MODEL);
    Log(DEBUG, "LLM Created!\n");
    return (void*)llm;

    // If the setup fails, free the allocated memory and return NULL
    clean_setup_fail:
    Log(ERROR, "Failed to create LLM object!");
    FreeLlm((void*)llm);
    return NULL;
}

void FreeLlm(void* objPtr) {
    clock_t startTime = clock();
    Log(DEBUG, "\nCleaning up the LLM... ");
    Llm* llm = (Llm*)objPtr;   // Cast the void* to Llm*
    if (llm == NULL) {
        return;
    }

    // Check if each component is non-NULL before freeing/releasing
    if (llm->g_ort) {
        if (llm->io_binding) llm->g_ort->ReleaseIoBinding(llm->io_binding);
        if (llm->allocator) llm->g_ort->ReleaseAllocator(llm->allocator);
        if (llm->session) llm->g_ort->ReleaseSession(llm->session);
        
        if (llm->cuda_options) llm->g_ort->ReleaseCUDAProviderOptions(llm->cuda_options);
        if (llm->memory_info) llm->g_ort->ReleaseMemoryInfo(llm->memory_info);
        if (llm->run_options) llm->g_ort->ReleaseRunOptions(llm->run_options);
        if (llm->session_options) llm->g_ort->ReleaseSessionOptions(llm->session_options);
        if (llm->arena_cfg) llm->g_ort->ReleaseArenaCfg(llm->arena_cfg);
        if (llm->env) llm->g_ort->ReleaseEnv(llm->env);
    }

    if (llm->processor != NULL) {
        free_processor(llm->processor);
    }

    // Free the LLM struct itself
    free(llm);
    llm = NULL;
    Log(DEBUG, "Done!");
}


TokenizedTexts* doTokenize(Llm* llm, char* text) {
    char* texts[] = {text};
    TokenizedTexts *tokensObj = prepare_texts(llm->processor, texts, 1);
    if (tokensObj == NULL) {
        Log(ERROR, "Failed to create the TokenizedTexts object");
        return NULL;
    }
    return tokensObj;
}

void runModel(Llm* llm, TokenizedTexts *myToks, int batchSize) {
    OrtValue *input_tensors[66];
    //int64_t newTokens[MAX_BATCH_SIZE] = {0};              // Array to hold the newly generated tokens

    // Array to hold the newly generated tokens
    float past_kv_data[MAX_BTC_SIZE * 8 * SEQ_LEN * 128] = {0};  // Initialize to zeros
    //int64_t past_seqLen = 0;


    int64_t* tokenIds = myToks->ids; // [22557, 1526, 28808]
    for (int64_t i = 0; i < myToks->shape[1]; i++) {
        Log(DEBUG, "Token[%ld]: %ld", i, tokenIds[i]); // tokensObj->ids[i]
    }

    //size_t data_len = batchSize * 8 * past_seqLen * 128 * sizeof(int64_t);
    int64_t tensor_shape[4] = {batchSize, 8, SEQ_LEN, 128};
    for (int i = 2; i < 66; i++) {
        ORT_CLEAN_ON_ERROR(cleanup, llm, llm->g_ort->CreateTensorWithDataAsOrtValue(llm->memory_info, past_kv_data, sizeof(past_kv_data), tensor_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &input_tensors[i]));
        ORT_CLEAN_ON_ERROR(cleanup, llm, llm->g_ort->BindInput(llm->io_binding, model_inpNames[i], input_tensors[i]));
    }



    /* 
    size_t data_len = dataShape[0] * dataShape[1] * sizeof(int64_t);
    int64_t output_shape[3] = {dataShape[0], dataShape[1], 768};

    // Input Tensor: "input_ids"
    ORT_CLEAN_ON_ERROR(encoder_cleanup, llm, llm->g_ort->CreateTensorWithDataAsOrtValue(llm->memory_info, tokenIds, data_len, dataShape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &Ort_InputIDs));

    // Bind the input tensors to IO bindings
    ORT_CLEAN_ON_ERROR(encoder_cleanup, llm, llm->g_ort->BindInput(llm->enc_io_binding, "input_ids", Ort_InputIDs));
    ORT_CLEAN_ON_ERROR(encoder_cleanup, llm, llm->g_ort->BindInput(llm->enc_io_binding, "attention_mask", Ort_AttnMask));
    
    // Create & Bind the output tensor
    ORT_CLEAN_ON_ERROR(encoder_cleanup, llm, llm->g_ort->CreateTensorAsOrtValue(llm->enc_allocator, output_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &lastHiddenState));
    ORT_CLEAN_ON_ERROR(encoder_cleanup, llm, llm->g_ort->BindOutput(llm->enc_io_binding, "last_hidden_state", lastHiddenState));

    // Run the Encoder
    ORT_CLEAN_ON_ERROR(encoder_cleanup, llm, llm->g_ort->RunWithBinding(llm->encoder_session, llm->run_options, llm->enc_io_binding)); */
    
    cleanup:
    //llm->g_ort->ReleaseValue(Ort_IDs);    // Clean out tensor
    return;
}



void LlmRun(void* ctx, char* text) {
    // Check if the context is valid
    if (ctx == NULL) {
        fprintf(stderr, "Invalid llm context!\n");
        return;
    }

    // Cast the context to LLM pointer
    Llm* llm = (Llm*)ctx;
    InferLlm(llm, text);
}

void InferLlm(Llm* llm, char* text) {
    OrtValue *Ort_IDs = NULL;
    OrtValue *Ort_Mask = NULL;
    //char* final_result = NULL;
    
    // Retrieve input/output info
    size_t num_inputs, num_outputs;
    ORT_CLEAN_ON_ERROR(llm_clean, llm, llm->g_ort->SessionGetInputCount(llm->session, &num_inputs));
    ORT_CLEAN_ON_ERROR(llm_clean, llm, llm->g_ort->SessionGetOutputCount(llm->session, &num_outputs));
    Log(DEBUG, "Total Inputs: %ld", num_inputs);
    Log(DEBUG, "Total Outputs: %ld\n", num_outputs);
    GetSessionDataType2(llm, llm->session);

    TokenizedTexts *myToks = doTokenize(llm, text);
    
    // If their are no tokens to process, return NULL
    if (myToks->shape[0]*myToks->shape[1] == 0) {
        Log(ERROR, "No tokens to process");
        goto llm_clean;
    }

    int batchSize = (int)myToks->shape[0];
    if (batchSize > MAX_BATCH_SIZE) {
        Log(ERROR, "batch size is too large to process: %d > %d", batchSize, MAX_BATCH_SIZE);
        goto llm_clean;
    }
    printf("Batch Size: %d\n\n", batchSize);
    
    
    // Create Tensors: "attention_mask" & "input_ids"
    size_t data_len = myToks->shape[0] * myToks->shape[1] * sizeof(int64_t);
    ORT_CLEAN_ON_ERROR(llm_clean, llm, llm->g_ort->CreateTensorWithDataAsOrtValue(llm->memory_info, myToks->attention_mask, myToks->data_len, myToks->shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &Ort_Mask));
    ORT_CLEAN_ON_ERROR(llm_clean, llm, llm->g_ort->CreateTensorWithDataAsOrtValue(llm->memory_info, myToks->ids, data_len, myToks->shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &Ort_IDs));

    // Bind the input tensors to IO bindings
    ORT_CLEAN_ON_ERROR(llm_clean, llm, llm->g_ort->BindInput(llm->io_binding, "input_ids", Ort_IDs));
    ORT_CLEAN_ON_ERROR(llm_clean, llm, llm->g_ort->BindInput(llm->io_binding, "attention_mask", Ort_Mask));



    //encoder_output = runEncoder(llm, Ort_Mask, myToks->ids, myToks->shape);
    runModel(llm, myToks, batchSize);

    
    
    
    // CLEAN UP
    llm_clean:
    llm->g_ort->ReleaseValue(Ort_IDs);
    llm->g_ort->ReleaseValue(Ort_Mask);
    return;
}
