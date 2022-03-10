FLAGS = 


all: a.out

a.out: functions.o main.o 
	nvcc functions.o main.o -o a.out 

functions.o: functions.cu 
	nvcc -c $(FLAGS) functions.cu

main.o: main.cu 
	nvcc -c $(FLAGS) main.cu

clean:
	rm -rf *.0 a.out
