# NNToolkitCore

Cross-platform mobile Neural Network library for training and inference on device.
CPU only.

On Apple platforms it uses Accelerate framework as a backend;
On other platforms it uses Eigen for matmul, kissfft for dft, and ARM NEON intrinsics for acceleration.

NN layers:
Conv1d
GRU
RNN
LSTM
BatchNorm
Activation
TimeDistributedDense
Dense

DSP tools:
Spectrogram
Window functions

Algorithms were tested using tf 2.3.0

Examples and tests will be exposed in different repository. 










