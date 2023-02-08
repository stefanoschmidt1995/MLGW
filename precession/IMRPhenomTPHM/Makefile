##
# This Makefile is to compile lal code
##

#Remember to: source /home/stefano/lscsoft/lalsuite/lalsuite-user-env.sh
#This enables the pyhton environment and you might be able to call import lal from this...

#LIBS=-L/usr/lib/x86_64-linux-gnu -lm

LAL=/home/stefano/opt/lalsuite_python3
#LAL=/home/stefano.schmidt/lscsoft/lalsuite_python3/_inst

FLAG=-std=c99

TARGETS=run_IMR #angles_IMR

all: $(TARGETS)

run_IMR: run_IMR.o
	gcc $(FLAG) -g -Wl,-rpath,$(LAL)/lib/ -o $@ $< $(LAL)/lib/*.so*

angles_IMR: angles_IMR.o
	gcc $(FLAG) -g -Wl,-rpath,$(LAL)/lib/ -o $@ $< $(LAL)/lib/*.so*
	
%.o: %.c
	gcc $(FLAG) -I $(LAL)/include -c $<
	
clean:
	rm -rf *.o *.txt __pycache__/

clear: clean
	rm -f  $(TARGETS)
