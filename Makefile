all: cpu

cpu: build_cpu run

gpu: build_gpu run

.PHONY: run

build_cpu:
	g++ -fPIC -c move_particles.cpp
	g++ -fPIC -shared -o move_particles.so move_particles.o
	@# nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libKernel.so --shared project.cu
	@echo "=====================================================\n\n"
run:
	python3 ./particles.py

build_gpu:
	nvcc -Wno-deprecated-gpu-targets --gpu-architecture=compute_35 --gpu-code=sm_35,compute_35 -rdc=true --ptxas-options=-v --compiler-options '-fPIC' -o move_particles_gpu.so --shared move_particles_gpu.cu
	@echo "=====================================================\n\n"
run:
	python3 ./particles.py

clean:
	rm ./*.o
	rm ./*.so
