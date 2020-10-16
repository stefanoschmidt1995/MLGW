CC := g++
CFLAGS += -O3 -fPIC -std=c++11 -Wall -Wextra  
# compile with the system Python:
INCDIRS = -I /usr/lib64/python2.7/site-packages/numpy/core/include/  -I/usr/include/python2.7/

#The actual compilation is performed in a two logical stages
	#g++ -c -fPIC function.cpp -o function.o
	#g++ -shared -Wl,-soname,library.so -o library.so function.o
#however we use a single line compilation command

#You can add tag ${INCDIRS} to compilation command if you wish to include python libraries (which btw, are useless)

default: postproc.so
	@echo "done $@"

%.so: %.cpp
	${CC} ${CFLAGS} $< -o $@ -shared

clean: 
	rm *.so



