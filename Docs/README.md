# Setup
### Run On Windows
In Powershell SSH to the user and IP of your linux machine: (May need to be in WSL)
```
ssh tech@172.21.188.179
```

### Install Dependencies on Machine
Dependency: JSON-C
```
$ sudo apt install libjson-c-dev
```

Dependency: NVCC Toolkit
```
$ sudo apt install nvidia-cuda-toolkit
```


### Install OnnxRuntime for C
Full tutorial on their [website](https://onnxruntime.ai/docs/install/)

1. Download the "onnxruntime-linux-x64-gpu-1.20.0.tgz" file from [this link](https://github.com/microsoft/onnxruntime/releases)
2. Move and include the header files in the include directory.
3. Move the `libonnxruntime.so` dynamic library to a desired path and include it.


### Download Google-SentencePiece
Download SentencePiece by following the instructions found [HERE](https://github.com/google/sentencepiece?tab=readme-ov-file#build-and-install-sentencepiece-command-line-tools-from-c-source)
- This is crucial to running the tokenizer, `spiece.model`, and most all functions in `sentencepiece_wrapper.cpp`
```
% sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
% git clone https://github.com/google/sentencepiece.git 
% cd sentencepiece
% mkdir build
% cd build
% cmake ..
% make -j $(nproc)
% sudo make install
% sudo ldconfig -v
```

### Export Variables
Export the **LD_LIBRARY_PATH** variable in the terminal
- Set it to the path of you CUDA `/lib64` directory
- Set it also to the path of your downloaded onnxruntime folder

For example:
```
$ export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-gpu-1.19.2:/usr/lib/cuda/lib64
```

### Unity for Unit Tests
Build Unity
```
$ git clone https://github.com/ThrowTheSwitch/Unity.git
$ git pull
```

# File Descriptions
### `inference.c & inference.h`
- Sets up the global variables such as the environment, allocator, IO bindings, model sessions, Cpu & Gpu memory arenas, etc.
- Cleans up all the global variables
- Runs the encoder_model.onnx, decoder_model.onnx, and decoder_with_past_model.onnx


### `utils.c & utils.h`
- Mainly for debugging and listing values of variables
- Don't think it has any actual importance to the program as whole



### Directory: `MoreBin/`
Contains files I will not be editing myself

Files in the directory:
- `spiece.model`: The tokenizer SentencePiece model. Does the heavy lifiting decoding and encoding texts to tokens before we sent them over to our ONNX models
- `json.hpp`: Allows `sentencepiece_wrapper.cpp` to read the `tokenizer.json` file


### Directory: `tests/`
Contains the files for some unit tests
`test_tokenizer.c`: Tests the tokenizer encoding functionality
`test_inference.c`: Tests the results of infering texts compared against the expected results







# Unity for Unit Tests
Build Unity
```
$ git clone https://github.com/ThrowTheSwitch/Unity.git
$ git pull
```

# Download Prose
```
$ go get github.com/jdkato/prose/v2
```

