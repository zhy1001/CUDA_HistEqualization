# environment
SM := 30

GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f
 
# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= /usr/lib/x86_64-linux-gnu
opencvINC= /usr/include/opencv

# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart

NVCCFLAGS :=
GccFLAGS = -fopenmp -O3 

debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = heq

all: build

build: $(TARGET)

$(TARGET): src/dlink.o src/main.o src/$(TARGET).o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_video #-link `pkg-config opencv --cflags --libs`

src/dlink.o: src/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

src/main.o: src/main.cpp
	$(GCC) $(GccFLAGS) -c $< -o $@
	
src/$(TARGET).o: src/$(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 
	
clean:
	$(RM) $(TARGET) src/*.o *.o *.tar* *.core* *out*.jpg *input*.jpg
