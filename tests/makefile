test_knn:
	nvcc -arch=sm_35 -c test_knn.cpp -o test_knn.o
	nvcc -arch=sm_35 test_knn.o ../knn.o -o test_knn -lcuda -lcublas
test_feature:
	g++ -c test_feature.cpp -o test_feature.o
	g++ test_feature.o ../compute_feature.o -o test_feature
clean:
	rm *.o test_knn
