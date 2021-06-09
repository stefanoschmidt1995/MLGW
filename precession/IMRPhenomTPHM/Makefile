##
# This Makefile is to compile lal code
##

#Remember to: source /home/stefano/lscsoft/lalsuite/lalsuite-user-env.sh
#This enables the pyhton environment and you might be able to call import lal from this...

LAL=/home/stefano/opt/lalsuite_python3
LIBS=-L/usr/lib/x86_64-linux-gnu -lm

TARGETS=run_IMR #angles_IMR

all: $(TARGETS)

run_IMR: run_IMR.o
	gcc -g -Wl,-rpath,$(LAL)/lib/ -o $@ $< $(LAL)/lib/*.so*

angles_IMR: angles_IMR.o
	gcc -g -Wl,-rpath,$(LAL)/lib/ -o $@ $< $(LAL)/lib/*.so*
	
%.o: %.c
	gcc -I $(LAL)/include -c $<
	
clean:
	rm -f *.o *.txt

clear: clean
	rm -f  $(TARGETS)
