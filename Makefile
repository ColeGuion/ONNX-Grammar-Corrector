#export  LD_LIBRARY_PATH=.
# Targets
TARGET = runner
TEST_TGT = testprog
GO_TGT = goprog

# Flags
CFLAGS = -g -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable
# -Wno-comment -Wno-sign-compare -Wno-unused-but-set-variable
CFLAGS += -I./MoreBin -Itests/Unity/src -I.
CXX_LDFLAGS = -lstdc++ -licuuc -licudata -I/usr/local/include -lonnxruntime -lonnxruntime_providers_shared -lonnxruntime_providers_cuda -lcuda -lcudart 
LDFLAGS = -L/usr/local/lib -ljson-c -lsentencepiece

# Source & Object files
INFER_LIBRARY = libinference.so
LIB_FILES = inference.c utils.c infer.c

C_FILES = $(wildcard *.c) $(wildcard MoreBin/*.c) $(wildcard Llm/*.c)
OBJ_C = $(C_FILES:.c=.o)

# Removed main.o from files
TEST_C_FILES = $(C_FILES) $(wildcard tests/*.c) $(wildcard tests/Unity/src/*.c)
TEST_OBJ_C := $(filter-out main.o, $(TEST_C_FILES:.c=.o))

CPP_FILES = $(wildcard *.cpp)
CPP_FILES += $(wildcard MoreBin/*.cpp)
OBJ_CPP = $(CPP_FILES:.cpp=.o)


all: run

# Build the executable
$(TARGET): $(OBJ_C) $(OBJ_CPP)
	@g++ *.o $(LDFLAGS) -o $(TARGET) $(CXX_LDFLAGS)

$(TEST_TGT): $(OBJ_CPP) $(TEST_OBJ_C)
	@g++ *.o $(LDFLAGS) -o $(TEST_TGT) $(CXX_LDFLAGS)

# Build the shared library
$(INFER_LIBRARY): $(LIB_FILES) $(CPP_FILES)
	gcc -shared -o $(INFER_LIBRARY) $(LIB_FILES) $(CPP_FILES) -I. $(CFLAGS) -fPIC

run: clean $(TARGET)
	./$(TARGET) $(t) $(r) $(lg) $(useCpu)

graph: clean $(TARGET)
	valgrind --tool=massif ./$(TARGET) $(t) $(r) $(lg) $(useCpu)


testme: clean $(TEST_TGT)
	./$(TEST_TGT)

# Build the Go executable
bldgo: $(INFER_LIBRARY)
	go build -o $(GO_TGT) main.go helper.go && ./$(GO_TGT)

memcheck: $(TARGET)
	valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all ./$(TARGET) 1 $(useCpu)

# Compile C and C++ files
.c.o:
	@gcc $(CFLAGS) -c $<
#@echo "Compiled '"$<"' successfully!"
.cpp.o:
	@g++ $(CFLAGS) -c $<

# Clean up build files
clean:
	rm -f $(TARGET) $(TEST_TGT) $(GO_TGT) $(INFER_LIBRARY) *.o MoreBin/*.o tests/*.o Llm/*.o

.PHONY: all clean bldgo run testEnc memcheck


