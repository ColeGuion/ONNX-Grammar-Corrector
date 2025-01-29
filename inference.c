#include "inference.h"
#include "MoreBin/utils.h"
#include <pthread.h>

// Model paths
char PATH_ENCODER[1000];
char PATH_DECODER[1000];
char PATH_DECODER_PAST[1000];
char PATH_GIBB_TOKENIZER[1000];
char PATH_GIBB[1000];
char PATH_SP_MODEL[1000];
bool USE_GPU = false;

// Decoder Input/Output Names
char* decoder_output_names[51] = {"logits", "present.0.decoder.key", "present.0.decoder.value", "present.0.encoder.key", "present.0.encoder.value", "present.1.decoder.key", "present.1.decoder.value", "present.1.encoder.key", "present.1.encoder.value", "present.2.decoder.key", "present.2.decoder.value", "present.2.encoder.key", "present.2.encoder.value", "present.3.decoder.key", "present.3.decoder.value", "present.3.encoder.key", "present.3.encoder.value", "present.4.decoder.key", "present.4.decoder.value", "present.4.encoder.key", "present.4.encoder.value", "present.5.decoder.key", "present.5.decoder.value", "present.5.encoder.key", "present.5.encoder.value", "present.6.decoder.key", "present.6.decoder.value", "present.6.encoder.key", "present.6.encoder.value", "present.7.decoder.key", "present.7.decoder.value", "present.7.encoder.key", "present.7.encoder.value", "present.8.decoder.key", "present.8.decoder.value", "present.8.encoder.key", "present.8.encoder.value", "present.9.decoder.key", "present.9.decoder.value", "present.9.encoder.key", "present.9.encoder.value", "present.10.decoder.key", "present.10.decoder.value", "present.10.encoder.key", "present.10.encoder.value", "present.11.decoder.key", "present.11.decoder.value", "present.11.encoder.key", "present.11.encoder.value"};
char* decPast_input_names[51] = {"input_ids", "encoder_attention_mask", "encoder_hidden_states", "past_key_values.0.decoder.key", "past_key_values.0.decoder.value", "past_key_values.0.encoder.key", "past_key_values.0.encoder.value", "past_key_values.1.decoder.key", "past_key_values.1.decoder.value", "past_key_values.1.encoder.key", "past_key_values.1.encoder.value", "past_key_values.2.decoder.key", "past_key_values.2.decoder.value", "past_key_values.2.encoder.key", "past_key_values.2.encoder.value", "past_key_values.3.decoder.key", "past_key_values.3.decoder.value", "past_key_values.3.encoder.key", "past_key_values.3.encoder.value", "past_key_values.4.decoder.key", "past_key_values.4.decoder.value", "past_key_values.4.encoder.key", "past_key_values.4.encoder.value", "past_key_values.5.decoder.key", "past_key_values.5.decoder.value", "past_key_values.5.encoder.key", "past_key_values.5.encoder.value", "past_key_values.6.decoder.key", "past_key_values.6.decoder.value", "past_key_values.6.encoder.key", "past_key_values.6.encoder.value", "past_key_values.7.decoder.key", "past_key_values.7.decoder.value", "past_key_values.7.encoder.key", "past_key_values.7.encoder.value", "past_key_values.8.decoder.key", "past_key_values.8.decoder.value", "past_key_values.8.encoder.key", "past_key_values.8.encoder.value", "past_key_values.9.decoder.key", "past_key_values.9.decoder.value", "past_key_values.9.encoder.key", "past_key_values.9.encoder.value", "past_key_values.10.decoder.key", "past_key_values.10.decoder.value", "past_key_values.10.encoder.key", "past_key_values.10.encoder.value", "past_key_values.11.decoder.key", "past_key_values.11.decoder.value", "past_key_values.11.encoder.key", "past_key_values.11.encoder.value"};
char* decPast_output_names[25] = {"logits", "present.0.decoder.key", "present.0.decoder.value", "present.1.decoder.key", "present.1.decoder.value", "present.2.decoder.key", "present.2.decoder.value", "present.3.decoder.key", "present.3.decoder.value", "present.4.decoder.key", "present.4.decoder.value", "present.5.decoder.key", "present.5.decoder.value", "present.6.decoder.key", "present.6.decoder.value", "present.7.decoder.key", "present.7.decoder.value", "present.8.decoder.key", "present.8.decoder.value", "present.9.decoder.key", "present.9.decoder.value", "present.10.decoder.key", "present.10.decoder.value", "present.11.decoder.key", "present.11.decoder.value"};
char* decPast_pkv_inputs[25] = {"", "past_key_values.0.decoder.key", "past_key_values.0.decoder.value", "past_key_values.1.decoder.key", "past_key_values.1.decoder.value", "past_key_values.2.decoder.key", "past_key_values.2.decoder.value", "past_key_values.3.decoder.key", "past_key_values.3.decoder.value", "past_key_values.4.decoder.key", "past_key_values.4.decoder.value", "past_key_values.5.decoder.key", "past_key_values.5.decoder.value", "past_key_values.6.decoder.key", "past_key_values.6.decoder.value", "past_key_values.7.decoder.key", "past_key_values.7.decoder.value", "past_key_values.8.decoder.key", "past_key_values.8.decoder.value", "past_key_values.9.decoder.key", "past_key_values.9.decoder.value", "past_key_values.10.decoder.key", "past_key_values.10.decoder.value", "past_key_values.11.decoder.key", "past_key_values.11.decoder.value"};


void UseGeco() {
    int mem1 = get_memstat("VmRSS");
    GecoConfig config = {false, 2, "0", false}; // {UseGpu, LogLevel, GpuId, doProfile}
    if (load_config("MoreBin/config.json") != 0) {
        Log(ERROR, "Failed to load config!");
        return;
    }

    addMem(mem1);
    
    return;

    // Allocate memory for GECO object
    Geco* geco = (Geco*)malloc(sizeof(Geco));
    if (geco == NULL) {
        Log(ERROR, "Failed to allocate memory for Geco object!");
        return;
    }
    memset(geco, 0, sizeof(Geco));  // Set all fields to 0 or NULL
    getMem("After Creating Geco Object");

    free(geco);
    geco = NULL;
    getMem("After Full Release");
}


int load_config(const char *filename) {
    clock_t startTime = clock();
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
        Log(ERROR, "Error parsing JSON file");
        fclose(file);
        return -1;
    }

    // Assign JSON values to variables   
    struct json_object *path_onnx, *path_gibb;//, *log_level;
    char onnxPath[100];
    char gibbPath[100];

    //if (json_object_object_get_ex(parsed_json, "logLevel", &log_level))
    //    LOG_LEVEL = json_object_get_int(log_level);

    if (json_object_object_get_ex(parsed_json, "gibberish_model_path", &path_gibb))
        strcpy(gibbPath, json_object_get_string(path_gibb));

    if (json_object_object_get_ex(parsed_json, "gec_model_path", &path_onnx)) 
        strcpy(onnxPath, json_object_get_string(path_onnx));

    snprintf(PATH_ENCODER, sizeof(PATH_ENCODER), "%s%s", onnxPath, "encoder_model.onnx");
    snprintf(PATH_DECODER, sizeof(PATH_DECODER), "%s%s", onnxPath, "decoder_model.onnx");
    snprintf(PATH_DECODER_PAST, sizeof(PATH_DECODER_PAST), "%s%s", onnxPath, "decoder_with_past_model.onnx");
    snprintf(PATH_SP_MODEL, sizeof(PATH_SP_MODEL), "%s%s", onnxPath, "spiece.model");
    snprintf(PATH_GIBB, sizeof(PATH_GIBB), "%s%s", gibbPath, "model.onnx");
    snprintf(PATH_GIBB_TOKENIZER, sizeof(PATH_GIBB_TOKENIZER), "%s%s", gibbPath, "tokenizer.json");

    json_object_put(parsed_json);
    fclose(file);
    clock_t endTime = clock();
    setTimerValue("Load Config", startTime, endTime);
    return 0;
}

// Macro to print memory usage
#define GET_MEM(n, item_name) \
    if (n == 11) { \
        Log(DEBUG, "\x1b[1;31mCreated %s\x1b[0m", item_name); \
        get_memory_usage(); \
    } else if (n == 21) { \
        Log(DEBUG, "\x1b[1;31mFreed %s\x1b[0m", item_name); \
        get_memory_usage(); \
    } \

void* NewGeco(GecoConfig config) {
    clock_t startTime = clock();
    Log(DEBUG, "Initializing a new Geco object...");
    if (load_config("MoreBin/config.json") != 0) {
        Log(ERROR, "Failed to load config!");
        return NULL;
    }

    // Allocate memory for GECO object
    Geco* geco = (Geco*)malloc(sizeof(Geco));
    if (geco == NULL) {
        Log(ERROR, "Failed to allocate memory for Geco object!");
        return NULL;
    }

    // Set all fields to 0 or NULL
    memset(geco, 0, sizeof(Geco));

    // Get the ONNX Runtime API handle
    geco->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (geco->g_ort == NULL) {
        Log(ERROR, "Failed to get ONNX Runtime API handle!");
        free(geco);
        geco = NULL;
        return NULL;
    }

    // Set device_id
    if (config.useGpu) USE_GPU = true;
    snprintf(geco->device_id, sizeof(geco->device_id), USE_GPU ? "gpu:%s" : "cpu", config.gpuId);
    //Log(DEBUG, "  - Device: '%s'", geco->device_id);


    GET_MEM(1, "GECO");
    // Create an ONNX Runtime environment 
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateEnv(config.OrtLogLevel, "test", &geco->env));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateCpuMemoryInfo(0, 0, &geco->memory_info)); // Memory Info (OrtArenaAllocator, OrtMemTypeDefault)
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateSessionOptions(&geco->session_options));  // Session options
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateRunOptions(&geco->run_options));          // Run options
    GET_MEM(1, "Env+");

    if (config.doProfile) {
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->EnableProfiling(geco->session_options, "Docs/Profile_Logs/gibb_profile.log"));
        Log(DEBUG, "  - Profiling enabled!");
    }


    // Configuration parameters
    /* const char* keys[] = {"max_mem", "arena_extend_strategy", "max_dead_bytes_per_chunk", "initial_chunk_size_bytes"};
    const size_t values[] = {0, 1, 256, 1024};
    int num_keys = sizeof(keys) / sizeof(keys[0]);
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateArenaCfgV2(keys, values, num_keys, &geco->arena_cfg));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateAndRegisterAllocatorV2(geco->env, "CPUExecutionProvider", geco->memory_info, geco->arena_cfg, NULL, NULL, 0));
    GET_MEM(1, "Arena Config"); */

    // Create Arena Config for GPU / CPU
    if (USE_GPU) { 
        // Enable CUDA for GPU acceleration & set the GPU id
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateCUDAProviderOptions(&geco->cuda_options));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->SetCurrentGpuDeviceId(atoi(config.gpuId)));

        const char* provider_keys[] = {"device_id", "arena_extend_strategy"};
        const char* provider_values[] = {config.gpuId, "kSameAsRequested"};
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->UpdateCUDAProviderOptions(geco->cuda_options, provider_keys, provider_values, 2));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(geco->session_options, geco->cuda_options));
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddRunConfigEntry(geco->run_options, "memory.enable_memory_arena_shrinkage", geco->device_id));
    } else {
        // Huge speed improvement for CPU execution
        ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->SetIntraOpNumThreads(geco->session_options, 1));
    }

    // Session & Run Options
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.use_env_allocators", "1"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->SetSessionGraphOptimizationLevel(geco->session_options, ORT_ENABLE_ALL));           // Enable all graph optimizations
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.dynamic_block_base", "4"));   //* Improves gpu performance

    // Performs asynchronous copies while running inference
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddRunConfigEntry(geco->run_options, "disable_synchronize_execution_providers", "1")); //* Improves gpu performance
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddRunConfigEntry(geco->run_options, "memory.enable_memory_arena_shrinkage", "cpu:0"));
    GET_MEM(1, "Run/Session Options");

    //ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->EnableMemPattern(geco->session_options)); //? Don't know what this does?
    //ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->DisableMemPattern(geco->session_options));
    
    //char* sess_opt = "session.disable_prepacking"; 
    //char* sess_opt = "session.set_denormal_as_zero"; 
    //char* sess_opt = "session.use_ort_model_bytes_directly"; 
    //char* sess_opt = "session.disable_quant_qdq";
    char* sess_opt = "session.use_device_allocator_for_initializers";
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, sess_opt, "1"));
    Log(DEBUG, "Session Option: \x1b[1;32m%s\x1b[0m", sess_opt); 

    //ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.disable_prepacking", "1"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.set_denormal_as_zero", "1"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.use_ort_model_bytes_directly", "1"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.strict_shape_type_inference", "0"));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.disable_quant_qdq", "1"));
    //ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->AddSessionConfigEntry(geco->session_options, "session.force_spinning_stop", "1"));
    printf("Using all sesh options\n");
    
    //* Possible Session Options    (NOTE: These could be different when using GPU)
    // AddSessionConfigEntry(geco->session_options, OPTION, VALUE_STRING));
    //      - ["session.disable_prepacking", "1"]
    //      - ["session.set_denormal_as_zero", "1"]         // Not much harm or gain
    //      - ["session.enable_quant_qdq_cleanup", "1"]     // Not much harm or gain
    //      - ["session.strict_shape_type_inference", "0"]  // Improved across the board
    //      - ["session.inter_op.allow_spinning", "1"]      // Slight improvement
    //      - ["session.intra_op.allow_spinning", "1"]      // Bigger decrease
    //      - ["session.use_ort_model_bytes_directly", "1"]
    //      - ["session.enable_gelu_approximation", "1"]    // Barely worse
    
    // Initialize Allocators & Sessions
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateSession(geco->env, PATH_ENCODER, geco->session_options, &geco->encoder_session));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateAllocator(geco->encoder_session, geco->memory_info, &geco->enc_allocator));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateIoBinding(geco->encoder_session, &geco->enc_io_binding));
    GET_MEM(1, "Encoder");

    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateSession(geco->env, PATH_DECODER, geco->session_options, &geco->decoder_session));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateAllocator(geco->decoder_session, geco->memory_info, &geco->dec_allocator));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateIoBinding(geco->decoder_session, &geco->dec_io_binding));
    GET_MEM(1, "Decoder");

    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateSession(geco->env, PATH_DECODER_PAST, geco->session_options, &geco->decPast_session));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateAllocator(geco->decPast_session, geco->memory_info, &geco->decPast_allocator));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateIoBinding(geco->decPast_session, &geco->decPast_io_binding));
    GET_MEM(1, "Decoder With Past");

    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateSession(geco->env, PATH_GIBB, geco->session_options, &geco->gibb_session));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateAllocator(geco->gibb_session, geco->memory_info, &geco->gibb_allocator));
    ORT_CLEAN_ON_ERROR(clean_setup_fail, geco, geco->g_ort->CreateIoBinding(geco->gibb_session, &geco->gibb_io_binding));
    GET_MEM(1, "Gibberish");

    // Load the SentencePiece model
    geco->processor = initialize_processor(PATH_SP_MODEL);
    Log(DEBUG, "Geco Created!\n");
    setTimerValue("Create Geco", startTime, clock());
    return (void*)geco;

    // If the setup fails, free the allocated memory and return NULL
    clean_setup_fail:
    Log(ERROR, "Failed to create Geco object!");
    FreeGeco((void*)geco);
    return NULL;
}

void FreeGeco(void* objPtr) {
    clock_t startTime = clock();
    Log(DEBUG, "Cleaning up the Geco... ");
    Geco* geco = (Geco*)objPtr;   // Cast the void* to Geco*
    if (geco == NULL) {
        return;
    }


    // Macro to help release resources
    #define RELEASE_RESOURCE(resource, release_func) \
        if (geco->resource) { \
            geco->g_ort->release_func(geco->resource); \
            geco->resource = NULL; \
        } else { \
            Log(WARNING, #resource " is NOT released"); \
        }

    // Check if each component is non-NULL before freeing/releasing
    if (geco->g_ort) {
        geco->g_ort->ClearBoundInputs(geco->enc_io_binding);
        geco->g_ort->ClearBoundOutputs(geco->enc_io_binding);
        geco->g_ort->ClearBoundInputs(geco->dec_io_binding);
        geco->g_ort->ClearBoundOutputs(geco->dec_io_binding);
        geco->g_ort->ClearBoundInputs(geco->decPast_io_binding);
        geco->g_ort->ClearBoundOutputs(geco->decPast_io_binding);
        geco->g_ort->ClearBoundInputs(geco->gibb_io_binding);
        geco->g_ort->ClearBoundOutputs(geco->gibb_io_binding);
        RELEASE_RESOURCE(enc_io_binding, ReleaseIoBinding);
        RELEASE_RESOURCE(dec_io_binding, ReleaseIoBinding);
        RELEASE_RESOURCE(decPast_io_binding, ReleaseIoBinding);
        RELEASE_RESOURCE(gibb_io_binding, ReleaseIoBinding);
        GET_MEM(2, "IO Bindings");

        RELEASE_RESOURCE(enc_allocator, ReleaseAllocator);
        RELEASE_RESOURCE(dec_allocator, ReleaseAllocator);
        RELEASE_RESOURCE(decPast_allocator, ReleaseAllocator);
        RELEASE_RESOURCE(gibb_allocator, ReleaseAllocator);
        GET_MEM(2, "Allocators");

        RELEASE_RESOURCE(encoder_session, ReleaseSession);
        RELEASE_RESOURCE(decoder_session, ReleaseSession);
        RELEASE_RESOURCE(decPast_session, ReleaseSession);
        RELEASE_RESOURCE(gibb_session, ReleaseSession);
        GET_MEM(2, "Sessions");
        
        //RELEASE_RESOURCE(cuda_options, ReleaseCUDAProviderOptions);
        geco->g_ort->ReleaseCUDAProviderOptions(geco->cuda_options);
        geco->cuda_options = NULL;
        RELEASE_RESOURCE(memory_info, ReleaseMemoryInfo);
        GET_MEM(2, "Memory Info");
        RELEASE_RESOURCE(run_options, ReleaseRunOptions);
        RELEASE_RESOURCE(session_options, ReleaseSessionOptions);
        GET_MEM(2, "Run/Sess Options");
        RELEASE_RESOURCE(arena_cfg, ReleaseArenaCfg);
        GET_MEM(2, "Arena Config");
        RELEASE_RESOURCE(env, ReleaseEnv);
        GET_MEM(2, "Env");
    } else {
        Log(WARNING, "g_ort is preventing ANYTHING from being released");
    }

    if (geco->processor != NULL) {
        free_processor(geco->processor);
        GET_MEM(2, "SP Processor");
    } else {
        Log(WARNING, "processor is NOT released");
    }

    // Free the Geco struct itself
    free(geco);
    geco = NULL;
    Log(DEBUG, "Done!");
    setTimerValue("Destroy Geco", startTime, clock());
}



void free_tensor(Geco* geco, OrtValue** tensor) {
    if (tensor != NULL && *tensor != NULL) {
        geco->g_ort->ReleaseValue(*tensor);
        *tensor = NULL;
    }
}

/* void cleanBoundOutputs(Geco* geco, OrtValue** output_tensors, size_t output_len) {
    if (geco == NULL || output_tensors == NULL) {
        Log(WARNING, "Bound Outputs can not be released");
        return;
    }
    for (size_t i = 0; i < output_len; i++) {
        if (output_tensors[i] != NULL) {
            geco->g_ort->ReleaseValue(output_tensors[i]);
            output_tensors[i] = NULL;
        }
    }
    output_tensors = NULL;
} */

void cleanBoundOutputs(Geco* geco, OrtValue*** output_tensors, size_t output_len) {
    if (geco == NULL || output_tensors == NULL || *output_tensors == NULL) {
        //Log(WARNING, "Bound Outputs can not be released");
        return;
    }
    for (size_t i = 0; i < output_len; i++) {
        if ((*output_tensors)[i] != NULL) {
            geco->g_ort->ReleaseValue((*output_tensors)[i]);
            (*output_tensors)[i] = NULL;
        }
    }
    *output_tensors = NULL;
}

bool checkRepeating(int* arr, int lastInd) {
    if (arr == NULL) {
        Log(ERROR, "checkRepeating() Array pointer is NULL");
        return false;
    }
    if (lastInd < 5) {
        return false;
    }

    if (arr[lastInd] == arr[lastInd-2] && arr[lastInd] == arr[lastInd-4] && arr[lastInd-1] == arr[lastInd-3] && arr[lastInd-1] == arr[lastInd-5]) {
        return true;
    }
    return false;
}

int getMaxTokens(Geco* geco, OrtValue* logits, int64_t* newTokens, int batchSize, int* completed_sequences, int runNum) {
    clock_t startTime = clock();
    // Check for NULL pointers
    if (!geco || !logits || !newTokens || !completed_sequences) {
        Log(ERROR, "getMaxTokens(): NULL pointer detected in input arguments");
        return -1;
    }

    // Get the logits data as a readable array
    float* logitData = NULL;
    if (geco->g_ort->GetTensorMutableData(logits, (void**)&logitData) != NULL) {
        Log(ERROR, "getMaxTokens(): Failed to get logits data");
        return -1;
    }

    // Find the most likely next token for each sequence
    for (int seqNum = 0; seqNum < batchSize; seqNum++) {
        if (completed_sequences[seqNum] == 1) {
            // This sequence is already completed, so add a 0 and continue to the next seq
            newTokens[seqNum] = 0;
            continue;
        }

        // If the sequence is starting a repeating loop then mark the indexes we won't allow to be generated next
        int ignore_indexes[2] = {-1, -1};
        if (checkRepeating(geco->generated_tokens[seqNum], runNum-1)) {
            ignore_indexes[0] = geco->generated_tokens[seqNum][runNum-1];
            ignore_indexes[1] = geco->generated_tokens[seqNum][runNum-2];
            //Log(DEBUG, "Ignoring indicies: [%d, %d]\n", ignore_indexes[0], ignore_indexes[1]);
        }

        int start_index = seqNum * LOGIT_SIZE;
        float maxVal = logitData[start_index];
        int nextToken = 0;

        for (int i = (start_index+1); i < (start_index+LOGIT_SIZE); i++) {
            if (logitData[i] > maxVal) {
                if ((i-start_index) == ignore_indexes[0] || (i-start_index) == ignore_indexes[1]) {
                    // Token is in the ignore list, so skip it
                    //Log(DEBUG, "Seq[%d] Skipping token %d\n", seqNum, i-start_index);
                    continue;
                }
                maxVal = logitData[i];
                nextToken = i - start_index;
            }
        }
        // Add to array of generated tokens
        newTokens[seqNum] = (int64_t)nextToken;
        
        // If this sequence reaches its eos token, mark it as completed
        if (nextToken == 1) {
            completed_sequences[seqNum] = 1;
        }
    }

    logitData = NULL;
    setTimerValue("Get Max Tokens", startTime, clock());
    return 0;
}

OrtValue* runEncoder(Geco* geco, OrtValue* Ort_AttnMask, int64_t* tokenIds, int64_t dataShape[2]) {
    clock_t startTime = clock();
    if (tokenIds == NULL || Ort_AttnMask == NULL) {
        return NULL;
    }
    OrtValue* Ort_InputIDs = NULL;
    OrtValue* lastHiddenState = NULL;

    // Bind the input tensors to IO bindings
    size_t data_len = dataShape[0] * dataShape[1] * sizeof(int64_t);
    ORT_CLEAN_ON_ERROR(encoder_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(geco->memory_info, tokenIds, data_len, dataShape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &Ort_InputIDs));
    ORT_CLEAN_ON_ERROR(encoder_cleanup, geco, geco->g_ort->BindInput(geco->enc_io_binding, "input_ids", Ort_InputIDs));
    ORT_CLEAN_ON_ERROR(encoder_cleanup, geco, geco->g_ort->BindInput(geco->enc_io_binding, "attention_mask", Ort_AttnMask));
    
    // Create & Bind the output tensor
    int64_t output_shape[3] = {dataShape[0], dataShape[1], 768};
    ORT_CLEAN_ON_ERROR(encoder_cleanup, geco, geco->g_ort->CreateTensorAsOrtValue(geco->enc_allocator, output_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &lastHiddenState));
    ORT_CLEAN_ON_ERROR(encoder_cleanup, geco, geco->g_ort->BindOutput(geco->enc_io_binding, "last_hidden_state", lastHiddenState));

    // Run the Encoder
    ORT_CLEAN_ON_ERROR(encoder_cleanup, geco, geco->g_ort->RunWithBinding(geco->encoder_session, geco->run_options, geco->enc_io_binding));
    
    encoder_cleanup:
    // Clear bindings
    geco->g_ort->ClearBoundInputs(geco->enc_io_binding);
    geco->g_ort->ClearBoundOutputs(geco->enc_io_binding);
    free_tensor(geco, &Ort_InputIDs);
    setTimerValue("Encoder", startTime, clock());
    return lastHiddenState;
}



void runPast(Geco* geco, int runNum, int64_t* nextToks, int batchSize, int* completed_sequences) {
    OrtValue** decPast_output_tensors = NULL;
    OrtValue* OrtDec_InputIDs = NULL;
    size_t decPast_output_len = 0;

    // Run the Model with IO Bindings and get the output tensors 
    ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->RunWithBinding(geco->decPast_session, geco->run_options, geco->decPast_io_binding));
    ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->GetBoundOutputValues(geco->decPast_io_binding, geco->decPast_allocator, &decPast_output_tensors, &decPast_output_len));

    // Get the most likely next tokens
    int gmtErr = getMaxTokens(geco, decPast_output_tensors[0], nextToks, batchSize, completed_sequences, runNum);
    if (gmtErr != 0) {
        //cleanBoundOutputs(geco, &decPast_output_tensors, decPast_output_len);
        for (int i=0; i<(int)decPast_output_len; i++) {
            free_tensor(geco, &decPast_output_tensors[i]);
        }
        geco->decPast_allocator->Free(geco->decPast_allocator, decPast_output_tensors);
        decPast_output_tensors = NULL;
        return;
    }

    // Add the new tokens to the generated tokens array
    bool sequences_completed = true;
    for (int i = 0; i < batchSize; i++) {
        geco->generated_tokens[i][runNum] = nextToks[i];
        if (completed_sequences[i] != 1) {
            sequences_completed = false;
        }
    }

    // Check if all sequences are finished
    if (sequences_completed || (runNum+1) == MAX_TOKENS) { 
        //Log(DEBUG, "All sequences completed @ Run #%d!\n", runNum); 
        //cleanBoundOutputs(geco, &decPast_output_tensors, decPast_output_len);
        for (int i=0; i<(int)decPast_output_len; i++) {
            free_tensor(geco, &decPast_output_tensors[i]);
        }
        geco->decPast_allocator->Free(geco->decPast_allocator, decPast_output_tensors);
        decPast_output_tensors = NULL;
        return;
    }

    // Bind the input tensor: "input_ids"
    size_t data_len = batchSize * sizeof(int64_t);
    int64_t data_shape[2] = {batchSize, 1}; 
    ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(geco->memory_info, nextToks, data_len, data_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &OrtDec_InputIDs));
    ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, "input_ids", OrtDec_InputIDs));
    free_tensor(geco, &OrtDec_InputIDs);

    // Bind the output tensors to the input tensors & Create new shaped output tensors
    for (int i=0; i<25; i++) {
        // Updates the output tensors shapes
        ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->BindOutputToDevice(geco->decPast_io_binding, decPast_output_names[i], geco->memory_info));
 
        // Update the input tensor bindings with the previous output tensors (Skip 'logits' since it has no correlating input)
        if (i != 0) {
            // Changes output_name from "present.0.decoder.key" to "past_key_values.0.decoder.key"
            /* int idx = (i-1) / 2;
            char inp_name[35];
            if (i % 2 == 0) { 
                snprintf(inp_name, 35, "past_key_values.%d.decoder.value", idx);
            } else {
                snprintf(inp_name, 35, "past_key_values.%d.decoder.key", idx);
            }

            ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, inp_name, decPast_output_tensors[i])); */
            ORT_CLEAN_ON_ERROR(decPast_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, decPast_pkv_inputs[i], decPast_output_tensors[i]));
        }
        free_tensor(geco, &decPast_output_tensors[i]); 
    }
    geco->decPast_allocator->Free(geco->decPast_allocator, decPast_output_tensors);
    decPast_output_tensors = NULL;

        
    // RECURSE
    runPast(geco, runNum+1, nextToks, batchSize, completed_sequences);
    
    // Free memory
    decPast_cleanup:
    free_tensor(geco, &OrtDec_InputIDs);
    //cleanBoundOutputs(geco, &decPast_output_tensors, decPast_output_len);
    if (decPast_output_tensors != NULL) {
        for (int i=0; i<(int)decPast_output_len; i++) {
            free_tensor(geco, &decPast_output_tensors[i]);
        }
        geco->decPast_allocator->Free(geco->decPast_allocator, decPast_output_tensors);
        decPast_output_tensors = NULL;
    }
}

void runDecoders(Geco* geco, OrtValue* lastHiddenState, OrtValue* Ort_AttnMask, int batchSize) {
    malloc_trim(0);
    MemCheck("Start of Decoders");
    clock_t startTime = clock();
    OrtValue* OrtDec_InputIDs = NULL;
    OrtValue** dec_output_tensors = NULL;
    size_t dec_output_len = 0;

    // Token arrays
    int64_t newTokens[MAX_BATCH_SIZE];              // Array to hold the newly generated tokens
    int completed_sequences[MAX_BATCH_SIZE] = {0};  // Mark 1 if a sequence is completed, otherwise 0
    int64_t init_tokens[MAX_BATCH_SIZE] = {0};
    
    size_t inputs_data_len = batchSize * sizeof(int64_t);
    int64_t inputs_shape[2] = {batchSize, 1};
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(geco->memory_info, init_tokens, inputs_data_len, inputs_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &OrtDec_InputIDs));

    // Bind tensors to IO bindings
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->dec_io_binding, "encoder_attention_mask", Ort_AttnMask));
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->dec_io_binding, "input_ids", OrtDec_InputIDs));
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->dec_io_binding, "encoder_hidden_states", lastHiddenState));
    for (int i = 0; i < 49; i++) {
        // Bind the Output Tensors
        ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindOutputToDevice(geco->dec_io_binding, decoder_output_names[i], geco->memory_info));
    }

    // Run the Model
    MemCheck("Before Decoder Run");
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->RunWithBinding(geco->decoder_session, geco->run_options, geco->dec_io_binding));
    MemCheck("After Decoder Run");

    // Get the return output values as OrtValue* Tensors
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->GetBoundOutputValues(geco->dec_io_binding, geco->dec_allocator, &dec_output_tensors, &dec_output_len));
    int gmtErr = getMaxTokens(geco, dec_output_tensors[0], newTokens, batchSize, completed_sequences, 1);
    if (gmtErr != 0) {
        goto decoder_cleanup;
    }
    MemCheck("Get Max Tokens");

    for (int i = 0; i < (int)dec_output_len; i++) {
        free_tensor(geco, &dec_output_tensors[i]);
    }
    geco->dec_allocator->Free(geco->dec_allocator, dec_output_tensors);
    dec_output_tensors = NULL;
    MemCheck("Clear Output tensors");
    geco->g_ort->ClearBoundInputs(geco->dec_io_binding);
    geco->g_ort->ClearBoundOutputs(geco->dec_io_binding);
    MemCheck("Clear IO Bounds");
    free_tensor(geco, &OrtDec_InputIDs);
    MemCheck("Free Input Tensor");

    geco->g_ort->ReleaseIoBinding(geco->dec_io_binding);
    geco->dec_io_binding = NULL;
    MemCheck("Release IO Binding");
    geco->g_ort->ReleaseAllocator(geco->dec_allocator);
    geco->dec_allocator = NULL;
    MemCheck("Release Allocator");
    geco->g_ort->ReleaseMemoryInfo(geco->memory_info);
    geco->memory_info = NULL;
    MemCheck("Release Memory Info");
    geco->g_ort->ReleaseSession(geco->decoder_session);
    geco->decoder_session = NULL;
    MemCheck("Release Session");
    malloc_trim(0);
    MemCheck("Malloc Trim");

    return;




    // Add the new tokens to the generated tokens array
    for (int i = 0; i < batchSize; i++) {
        geco->generated_tokens[i][1] = newTokens[i];
    }
    free_tensor(geco, &OrtDec_InputIDs);
    setTimerValue("Main Decoder", startTime, clock());


    // DECODER_WITH_PAST_MODEL
    // Bind the input tensors to IO bindings
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(geco->memory_info, newTokens, inputs_data_len, inputs_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &OrtDec_InputIDs));
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, "encoder_attention_mask", Ort_AttnMask));
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, "input_ids", OrtDec_InputIDs));
    ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, "encoder_hidden_states", lastHiddenState));
    for (int i = 3; i < 51; i++) {
        // Append the outputs from the decoder model as the inputs to this model
        ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindInput(geco->decPast_io_binding, decPast_input_names[i], dec_output_tensors[i-2]));
        free_tensor(geco, &dec_output_tensors[i-2]);
    }

    // Create & Bind the Output Tensors
    for (int i = 0; i < 25; i++) {
        ORT_CLEAN_ON_ERROR(decoder_cleanup, geco, geco->g_ort->BindOutputToDevice(geco->decPast_io_binding, decPast_output_names[i], geco->memory_info));
    } 

    // Run and recurse
    clock_t rp_startTime = clock();
    MemCheck("Before DecPast Run");
    runPast(geco, 2, newTokens, batchSize, completed_sequences);
    setTimerValue("Past Decoder", rp_startTime, clock());
    MemCheck("After DecPast Run");

    // Clean up
    decoder_cleanup:
    free_tensor(geco, &OrtDec_InputIDs);
    // Clear Bounded Inputs & Outputs
    //cleanBoundOutputs(geco, &dec_output_tensors, dec_output_len);
    if (dec_output_tensors != NULL) {
        for (int i=0; i<(int)dec_output_len; i++) {
            free_tensor(geco, &dec_output_tensors[i]);
        }
    }
    geco->g_ort->ClearBoundInputs(geco->dec_io_binding);
    geco->g_ort->ClearBoundOutputs(geco->dec_io_binding);
    geco->g_ort->ClearBoundInputs(geco->decPast_io_binding);
    geco->g_ort->ClearBoundOutputs(geco->decPast_io_binding);
    geco->dec_allocator->Free(geco->dec_allocator, dec_output_tensors);
    dec_output_tensors = NULL;
    setTimerValue("All Decoders", startTime, clock());
    MemCheck("End of Decoders");
}
 


char* GecoRun(void* context, char** texts, int num_texts) {
    // Check if the context is valid
    if (context == NULL) {
        fprintf(stderr, "Invalid Geco context!\n");
        return NULL;
    }

    // Cast the context to Geco pointer
    Geco* geco = (Geco*)context;

    // Increment run counter & Reinitalize allocators every 2 runs
    /* geco->runCounter++;
    if (geco->runCounter % 2 == 0) {
        get_memory_usage();
        ReinitAllocators(geco);
        get_memory_usage();
    } */

    // Call the InferModel function with the geco and input texts
    return InferModel(geco, texts, num_texts);
}

char* InferModel(Geco* geco, char** texts, int num_texts) {
    OrtValue* Ort_AttnMask = NULL;
    OrtValue* lastHiddenState = NULL;
    char* final_result = NULL;
    clock_t startTime = clock();

    
    TokenizedTexts *tokensObj = prepare_texts(geco->processor, texts, num_texts);
    if (tokensObj == NULL) {
        Log(ERROR, "Failed to create the TokenizedTexts object");
        goto model_run_cleanup;
    }
    setTimerValue("GEC Preproc", startTime, clock());

    // If their are no tokens to process, return NULL
    if (tokensObj->shape[0]*tokensObj->shape[1] == 0) {
        Log(ERROR, "No tokens to process");
        goto model_run_cleanup;
    }

    int batchSize = (int)tokensObj->shape[0];
    if (batchSize > MAX_BATCH_SIZE) {
        Log(ERROR, "batch size is too large to process: %d > %d", batchSize, MAX_BATCH_SIZE);
        goto model_run_cleanup;
    }
    //Log(DEBUG, "Batch Size: %d", batchSize);
    //Log(DEBUG, "Seq Size: %d\nNum Texts: %d", (int)tokensObj->shape[1], num_texts);

    // Reset the generated tokens array to all 0's
    for (int i = 0; i < MAX_BATCH_SIZE; i++) {
        for (int j = 0; j < MAX_TOKENS; j++) {
            geco->generated_tokens[i][j] = 0;
        }
    }

    //GetMemoryInfo(geco, geco->enc_allocator); ["Name": CPU, "ID": 0, "Mem-Type": Default, "Alloc-Type": Arena, "Device-Type": CPU]

    // Tensor: "attention_mask"
    ORT_CLEAN_ON_ERROR(model_run_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(geco->memory_info, tokensObj->attention_mask, tokensObj->data_len, tokensObj->shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &Ort_AttnMask));

    // Run model sessions
    lastHiddenState = runEncoder(geco, Ort_AttnMask, tokensObj->ids, tokensObj->shape);
    if (lastHiddenState == NULL) {
        Log(ERROR, "Failed to run the Encoder model!");
        goto model_run_cleanup;
    }
    //printMem(mem_rss, mem_heap);
    
    
    int mem_rss = get_memstat("VmRSS");
    int mem_heap = get_memstat("VmData");
    runDecoders(geco, lastHiddenState, Ort_AttnMask, batchSize);
    MemCheck("Left Decoder Func");
    printMem(mem_rss, mem_heap);


    // Decode and print the results
    clock_t postTime = clock();
    final_result = decode_texts(geco->processor, geco->generated_tokens, tokensObj);
    setTimerValue("GEC Postproc", postTime, clock());
    
    // CLEAN UP
    model_run_cleanup:
    free_tensor(geco, &Ort_AttnMask);
    free_tensor(geco, &lastHiddenState);
    free_tokenized_texts(tokensObj);
    setTimerValue("GEC Total", startTime, clock());
    //malloc_trim(0);
    //PrintTimes();
    return final_result;
}


void GecoGibb(void* context, double probs[MAX_BATCH_SIZE][GIBB_CLASSES], char** texts, int num_batches) {
    // Check if the context is valid
    if (context == NULL) {
        fprintf(stderr, "Invalid Geco context!\n");
        return;// NULL;
    }

    // Cast the context to Geco pointer
    Geco* geco = (Geco*)context;

    // Call the InferGibb function with the geco and input texts
    InferGibb(geco, probs, texts, num_batches);
}

void InferGibb(Geco* geco, double probs[MAX_BATCH_SIZE][GIBB_CLASSES], char** texts, int num_batches) {
    clock_t startTime = clock();
    OrtValue* Ort_InputIDs = NULL; 
    OrtValue* Ort_AttnMask = NULL;
    OrtValue* logits = NULL;
    Tokenized_WP_Output tokenized_texts;

    // Tokenize the input texts
    tokenized_texts = batch_gibb_texts(PATH_GIBB_TOKENIZER, texts, num_batches);
    setTimerValue("Gibb Preproc", startTime, clock());
    Log(DEBUG, "Gibb Shape: %ld x %ld", tokenized_texts.shape[0], tokenized_texts.shape[1]);
    Log(DEBUG, "Gibb Length: %d", tokenized_texts.length);
    int batchSize = tokenized_texts.shape[0];

    // Reset probs array to 0
    for (int i=0; i<MAX_BATCH_SIZE; i++) {
        for (int j=0; j<GIBB_CLASSES; j++) {
            probs[i][j] = 0.0;
        }
    }
    
    int64_t data_len = tokenized_texts.shape[0] * tokenized_texts.shape[1] * sizeof(int64_t);
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(
        geco->memory_info, 
        tokenized_texts.attention_mask, 
        data_len, 
        tokenized_texts.shape, 
        2, 
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
        &Ort_AttnMask
    ));
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->CreateTensorWithDataAsOrtValue(
        geco->memory_info, 
        tokenized_texts.ids, 
        data_len, 
        tokenized_texts.shape, 
        2, 
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
        &Ort_InputIDs
    ));

    // Bind the input tensors to IO bindings
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->BindInput(geco->gibb_io_binding, "input_ids", Ort_InputIDs));
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->BindInput(geco->gibb_io_binding, "attention_mask", Ort_AttnMask));

    int64_t logit_shape[2] = {batchSize, GIBB_CLASSES};
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->CreateTensorAsOrtValue(geco->gibb_allocator, logit_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &logits));
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->BindOutput(geco->gibb_io_binding, "logits", logits));


    clock_t gibbTime = clock();
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->RunWithBinding(geco->gibb_session, geco->run_options, geco->gibb_io_binding));
    setTimerValue("Gibb Model Run", gibbTime, clock());


    // Perform softmax on an array of logits to convert them to probabilities
    clock_t softmaxTime = clock();
    float* logitData;
    ORT_CLEAN_ON_ERROR(gibberish_cleanup, geco, geco->g_ort->GetTensorMutableData(logits, (void**)&logitData));

    for (int seqNum = 0; seqNum < batchSize; seqNum++) {
        int idx = seqNum * GIBB_CLASSES;
        int64_t max_logit = logitData[idx];

        // Find the maximum logit for numerical stability
        for (int i = 1; i < GIBB_CLASSES; i++) {
            if (logitData[idx+i] > max_logit) {
                max_logit = logitData[idx+i];
            }
        }

        // Compute exponentials and sum them
        double sum = 0.0;
        for (int i = 0; i < GIBB_CLASSES; i++) {
            probs[seqNum][i] = exp(logitData[idx+i] - max_logit); // Subtract max_logit for numerical stability
            sum += probs[seqNum][i];
        }

        // Normalize the probabilities
        for (int i = 0; i < GIBB_CLASSES; i++) {
            probs[seqNum][i] /= sum;
            probs[seqNum][i] *= 100;
        }
    }
    setTimerValue("Calculate Softmax", softmaxTime, clock());
    
    gibberish_cleanup:
    free_tokenized_wp_output(tokenized_texts);
    free_tensor(geco, &Ort_InputIDs);
    free_tensor(geco, &Ort_AttnMask);
    free_tensor(geco, &logits);
    logitData = NULL;
    setTimerValue("Gibberish Total", startTime, clock());
    //PrintTimes();
}

 
void EncText(void* ctx, char* text) {
    Geco* geco = (Geco*)ctx;
    encodeText(geco->processor, text);
}




