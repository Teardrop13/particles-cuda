all: cpu

cpu: build_cpu run_cpu

gpu: build_gpu run_cpu

.PHONY: run

build_cpu:
	g++ -fPIC -c move_particles_cpu.cpp
	g++ -fPIC -shared -o move_particles_cpu.so move_particles_cpu.o
	@echo "=====================================================\n\n"
run_cpu:
	python3 ./particles.py --cpu

build_gpu:
	nvcc -Wno-deprecated-gpu-targets --gpu-architecture=compute_35 --gpu-code=sm_35,compute_35 -rdc=true --ptxas-options=-v --compiler-options '-fPIC' -o move_particles_gpu.so --shared move_particles_gpu.cu
	@echo "=====================================================\n\n"

run_gpu:
	python3 ./particles.py --gpu

clean:
	rm ./*.o
	rm ./*.so
