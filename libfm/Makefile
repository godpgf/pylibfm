LIB_DIR := ../lib/

OBJECTS := libfm_api.o

all: libFM

libFM: $(OBJECTS)
	mkdir -p $(LIB_DIR)
	ar crv $(LIB_DIR)libfm_api.a $(OBJECTS)
	g++ -fPIC -std=c++11 -shared -o $(LIB_DIR)libfm_api.so $(OBJECTS)
	#g++ -O3 -Wall libfm_api.o -o $(BIN_DIR)libFM

%.o: %.cpp
	g++ -fPIC -std=c++11 -O3 -Wall -c -g -rdynamic $< -o $@

clean:	clean_lib
	mkdir -p $(LIB_DIR)
	rm -rf $(LIB_DIR)

clean_lib:
	rm -f $(OBJECTS)


