LIB_DIR := ../lib/

OBJECTS := libfm_api.o

all: libFM

libFM: libfm_api.o
	mkdir -p $(LIB_DIR)
	ar crv $(LIB_DIR)libfm_api.a $(OBJECTS)
	g++ -shared -o $(LIB_DIR)libfm_api.so $(OBJECTS)
	#g++ -O3 -Wall libfm_api.o -o $(BIN_DIR)libFM

%.o: %.cpp
	g++ -O3 -Wall -c $< -o $@

clean:	clean_lib
	mkdir -p $(LIB_DIR)
	rm -rf $(LIB_DIR)

clean_lib:
	rm -f $(OBJECTS)


