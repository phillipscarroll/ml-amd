# Welcome to my ML AMD Notes

This will serve and my notes for setting up and running AMD related projects within PyTorch, Tensorflow, StableDiffusion etc... I just recently (March 2025) built a new system that will serve as the test platform.

I am currently testing with:

- ~~9950X 75c Limit PBO CO -30 [44k Cinebench R23]~~
- 9950X3D is shipping in, I will re-evaluate and refresh with the new processor
- ASRock Taichi RX 9070 XT 16GB [No ROCm support just yet]

Right now I will just look at testing on CPU with the 9950X. So far on my offline testing with BERT Tokenization of JSON data its roughly twice as fast as my 14900K. However this is a bit of apples and oranges since the 9950X has no problem running all 32 threads full-tilt with equal loading and the 14900K can become unstable on a similar tune.

One of the motivators of grabbing a 9950X was native AVX-512, native FP16 (the 14900K has BF16) and to see how the cpu based ML workloads are handled. 

Once ROCm on Windows, WSL2, or Linux is available for the new RDNA4 RX 9070 XT cards we can circle back and test.