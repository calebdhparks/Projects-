PROJ = cuda_tree
CC = g++
CFLAGS = -c -O3 -std=c++11 -I/opt/local/include
NVCC = nvcc

OBJS = cuda_tree.o cuda_dtree.o

all: $(PROJ)

$(PROJ): $(OBJS)
	$(NVCC) $^ -o $@

%.o : %.cu %.h
	$(NVCC) -c $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@ 

clean:
	rm -f $(PROJ) $(OBJS)