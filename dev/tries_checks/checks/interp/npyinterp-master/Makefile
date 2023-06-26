
CC := gcc
CFLAGS += -O3 -fPIC -std=c99 -Wall -Wextra  
# compile with the system Python:
INCDIRS = -I /usr/lib64/python2.7/site-packages/numpy/core/include/  -I/usr/include/python2.7/
# or, instead compile with the Python in Sherpa:
# INCDIRS = -I /opt/ciao-4.6/ots/lib/python2.7/site-packages/numpy/core/include/ -I /opt/ciao-4.6/ots/include/python2.7/

all: npyinterp.so
	@echo "done: $@"
allplatforms: npyinterp64.so npyinterp32.so
	@echo "done: $@"

%.so: %.c
	${CC} ${CFLAGS} $< -o $@ -shared ${INCDIRS}
%64.so: %.c 
	${CC} ${CFLAGS} -m64 $< -o $@ -shared ${INCDIRS}
%32.so: %.c 
	${CC} ${CFLAGS} -m32 $< -o $@ -shared ${INCDIRS}

clean: 
	rm *.so

.PHONY: all allplatforms clean

