CDIR = ../commons
IDIR = ../../lib
MKLROOT=/opt/intel/mkl

DEBUG = -g
#-DNDEBUG
#-g

INTEL_CC = icc
INTEL_CFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -I"${MKLROOT}"/include -O3 -march=core-avx2 -std=c++14 -inline-factor=500 -no-inline-max-size -no-inline-max-total-size -use-intel-optimized-headers -parallel -qopt-prefetch=4 -qopt-mem-layout-trans=2 -pthread -c
INTEL_LFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -I"${MKLROOT}"/include -O3 -march=core-avx2 -std=c++14 -inline-factor=500 -no-inline-max-size -no-inline-max-total-size -use-intel-optimized-headers -parallel -qopt-prefetch=4 -qopt-mem-layout-trans=2 -pthread -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl
INTEL_TFLAGS = -I"$(CDIR)" -fast -DNDEBUG -std=c++14 -inline-factor=500 -no-inline-max-size -no-inline-max-total-size -use-intel-optimized-headers -parallel -qopt-prefetch=4 -qopt-mem-layout-trans=3 -pthread 

GNU_CC = g++
GNU_CFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -O3 -march=core-avx2 -pthread -std=c++14 -c
GNU_LFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -O3 -march=core-avx2 -pthread -std=c++14

LLVM_CC = clang++
# Minimum required LLVM/CLang version is 3.4, in which we have to use -std=c++1y for c++14 support.
# In later versions we could use -std=c++14, but we can also use -std=c++1y still.
LLVM_CFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -O3 -march=core-avx2 -pthread -stdlib=libc++ -std=c++1y -c
LLVM_LFLAGS = $(DEBUG) -I"$(CDIR)" -I"$(IDIR)" -O3 -march=core-avx2 -pthread -stdlib=libc++ -std=c++1y

CC = $(GNU_CC)
CFLAGS = $(GNU_CFLAGS)
LFLAGS = $(GNU_LFLAGS)

SOURCEDIR = .
BUILDDIR = ../build
EXECUTABLE = cover_tree

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

COMMONS = $(wildcard $(SOURCEDIR)/../commons/*.cpp)
COBJECT = $(patsubst $(SOURCEDIR)/../commons/%.cpp,$(BUILDDIR)/%.o,$(COMMONS))

all: $(EXECUTABLE)

gcc: $(EXECUTABLE)

intel: CC=$(INTEL_CC)
intel: CFLAGS=$(INTEL_CFLAGS)
intel: LFLAGS=$(INTEL_LFLAGS)
intel: $(EXECUTABLE)

llvm: CC=$(LLVM_CC)
llvm: CFLAGS=$(LLVM_CFLAGS)
llvm: LFLAGS=$(LLVM_LFLAGS)
llvm: $(EXECUTABLE)
 
$(EXECUTABLE): $(COBJECT)  $(OBJECTS)
	$(CC) $(LFLAGS) $^ -o $@

$(OBJECTS): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

$(COBJECT): $(BUILDDIR)/%.o : $(SOURCEDIR)/../commons/%.cpp
	$(CC) $(CFLAGS) $< -o $@

inteltogether:
	$(INTEL_CC) $(INTEL_TFLAGS) $(SOURCES) $(COMMONS) -o $(EXECUTABLE)

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(EXECUTABLE)
