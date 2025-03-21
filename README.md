# Welcome to my ML AMD Notes

This will serve and my notes for setting up and running AMD related projects within PyTorch, Tensorflow, StableDiffusion etc... I just recently (March 2025) built a new system that will serve as the test platform.

System Setup:

- ~~9950X 65c Limit PBO CO -30 [44k Cinebench R23]~~
    - 9950X3D 65c Limit PBO CO -30
- ASRock Taichi RX 9070 XT 16GB
- ASRock B850i ITX Motherboard
- Corsair 96GB (48GB x 2) EXPO DDR5 6000Mhz CL30-36-36-76
    - Around 80ns latency in system
- be quiet! Silent Loop 3 240mm AIO
- Acer Predator GM7000 4TB NVME x 2

Right now I will just look at testing on CPU with the ~~9950X~~9950X3D. So far on my offline testing with BERT Tokenization of JSON data its roughly twice as fast as my 14900K. However this is a bit of apples and oranges since the ~~9950X~~9950X3D has no problem running all 32 threads full-tilt with equal loading and the 14900K can become unstable on a similar tune.

One of the motivators of grabbing a 9950X was native AVX-512, native FP16 (the 14900K has BF16) and to see how the cpu based ML workloads are handled. 

Once ROCm on Windows, WSL2, or Linux is available for the new RDNA4 RX 9070 XT cards we can circle back and test.

<a href="archive/navi31-rocm-zluda-pytorch-sd.md">Old 2023-2024 ROCm Navi31 Notes</a>

<a href="benchmarks/9950x3d-benchmarking.ipynb">9950X3D Benchmarking Notes</a>

<a href="benchmarks/rx9070xt-benchmarking.ipynb">RX 9070 XT Benchmarking Notes</a>

<a href="pytorch-directml-9070xt/README.md">RX 9070 XT PyTorch DirectML Notes</a>

<a href="pytorch-directml-9070xt/rx9070xt-pytorch-directml.ipynb">RX 9070 XT PyTorch DirectML Jupyter Notebook</a>

<a href="tensorflow-directml-9070xt/rx9070xt-tensorflow-directml.ipynb">RX 9070 XT PyTorch DirectML Notes and Jupyter Notebook</a>