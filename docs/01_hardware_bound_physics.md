# 01. Hardware Bound Physics & The Roofline Model

## The Fundamental Roofline Equation
The Roofline Model provides an absolute physical ceiling on the attainable performance of an algorithm on a given accelerator:

$$\text{Attainable Performance (TFLOPS)} = \min\left(\text{Peak Compute (TFLOPS)}, \; I \times \text{Memory Bandwidth (TB/s)}\right)$$

Where **Operational Intensity** (or Arithmetic Intensity, $I$) is defined as:

$$I = \frac{\text{Total Arithmetic Operations (FLOPs)}}{\text{Total Memory Traffic (Bytes Transferred Across Memory Bus)}}$$

## The Knee Point Derivation
The inflection point between memory-bound and compute-bound regimes is the **Knee Point**:

$$I_{\text{knee}} = \frac{\text{Peak Compute Ceiling (TFLOPS)}}{\text{Peak Memory Bandwidth (TB/s)}}$$

### Modern Accelerator Knee Points:
* **NVIDIA H100 SXM5**: $\frac{989 \text{ TFLOPS}}{3.35 \text{ TB/s}} \approx \mathbf{295.2 \text{ FLOPs/Byte}}$
* **NVIDIA A100 SXM4 (80GB)**: $\frac{312 \text{ TFLOPS}}{2.039 \text{ TB/s}} \approx \mathbf{153.0 \text{ FLOPs/Byte}}$
* **NVIDIA L40S**: $\frac{362 \text{ TFLOPS}}{0.864 \text{ TB/s}} \approx \mathbf{418.9 \text{ FLOPs/Byte}}$

## Prefill GEMM vs Autoregressive Decode GEMV
In a linear projection $Y = XW$ ($D_{\text{in}} = 4096, D_{\text{out}} = 4096$, FP16):
* **Prefill Phase ($S = 2048, B = 1$)**: 
  $$\text{FLOPs} = 2 \times 2048 \times 4096 \times 4096 \approx 6.87 \times 10^{10} \text{ FLOPs}$$
  $$\text{Memory Traffic} = W (33.55\text{MB}) + X (16.78\text{MB}) + Y (16.78\text{MB}) \approx 67.1\text{ MB}$$
  $$I_{\text{prefill}} \approx \mathbf{1024 \text{ FLOPs/Byte}} \gg I_{\text{knee}} \implies \text{COMPUTE BOUND}$$
* **Decode Phase ($S = 1, B = 1$)**:
  $$\text{FLOPs} = 2 \times 1 \times 4096 \times 4096 \approx 3.35 \times 10^7 \text{ FLOPs}$$
  $$\text{Memory Traffic} \approx W (33.55\text{MB}) + X (8\text{KB}) + Y (8\text{KB}) \approx 33.57\text{ MB}$$
  $$I_{\text{decode}} \approx \mathbf{1.0 \text{ FLOPs/Byte}} \ll I_{\text{knee}} \implies \text{STRICTLY MEMORY BOUND}$$

## Single-Stream Decode Latency Physics
Because decode must reload all model weights for every single generated token:
$$t_{\text{token}} \ge \frac{\text{Model Weight Size (Bytes)}}{\text{Peak Memory Bandwidth (Bytes/s)}}$$
For Llama-3-70B in FP16 ($140\text{ GB}$):
* On A100 SXM4 ($2039\text{ GB/s}$): $t_{\text{token}} \ge \frac{140}{2039} \approx \mathbf{68.6\text{ ms}} \implies \mathbf{\le 14.5\text{ tok/s}}$
* On H100 SXM5 ($3350\text{ GB/s}$): $t_{\text{token}} \ge \frac{140}{3350} \approx \mathbf{41.8\text{ ms}} \implies \mathbf{\le 23.9\text{ tok/s}}$
