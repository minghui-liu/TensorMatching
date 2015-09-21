all: synthetic_data real_image

synthetic_data: synthetic_data.o image_matching.o compute_feature.o knn.o tensor_matching.o
	nvcc -arch=sm_35 $^ -o $@ -lcuda -lcurand -lcublas -L ANN/lib -lANN -lnvToolsExt
real_image: real_image.o image_matching.o compute_feature.o tensor_matching.o libsurf.a
	nvcc -arch=sm_35 $^ -o $@ -Xlinker '`pkg-config --libs opencv`' -L ANN/lib -lANN -lnvToolsExt

synthetic_data.o: synthetic_data.cu
	nvcc -arch=sm_35 -c $< -o $@ -I ANN/include 
real_image.o: real_image.cu
	nvcc -arch=sm_35 -c $^ -o $@ -I OpenSURFcpp/src -Xcompiler '`pkg-config --cflags opencv` -D LINUX'
image_matching.o: image_matching.cu
	nvcc -arch=sm_35 -c $< -o $@ -I ANN/include 
compute_feature.o: compute_feature.cpp
	g++ -c $< -o $@ 
knn.o: knn_cublas_with_indexes.cu
	nvcc -arch=sm_35 -c $< -o $@ -lcuda -lcublas -D_CRT_SECURE_NO_DEPRECATE
tensor_matching.o: tensor_matching.cu
	nvcc -arch=sm_35 -c $< -o $@
clean:
	rm *.o synthetic_data real_image

