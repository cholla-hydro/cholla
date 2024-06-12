OBJS	:= supernova.o main.o
CC	= g++ -DPRECISION=2 -std=c++11 -DSUPERNOVA
tester: $(OBJS)
	$(CC) $(OBJS) -o tester
	./tester
%.o : %.h
%.o : %.cpp
	$(CC) -c $< -o $@

#cluster_list.o: cluster_list.cpp cluster_list.h
#	$(CC) -c $< -o $@

#main.o: main.cpp
#	$(CC) -c $< -o $@
supernova.o: S99_table.data cluster_list.cpp

cluster_list.cpp: make_clusterlist.py cluster_list.txt
	python make_clusterlist.py

S99_table.data: make_S99.py
	python make_S99.py
