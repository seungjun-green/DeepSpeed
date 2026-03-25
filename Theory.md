# All the things you need to know about DeepSpeed

## Training

### Overview
So to summarize, when you want to train a large model but it doesn’t fit on a single GPU, the first thing you’d think of is: let’s use multiple GPUs.

But simply adding more GPUs doesn’t help you train bigger models. Why? Because each GPU has to hold a full copy of the model’s parameters, gradients, and optimizer states in its VRAM. The only benefit you get from multiple GPUs is data parallelism — bigger effective batch size. For example, if each GPU processes a batch size of 4 and you have 8 GPUs, your effective batch size becomes 32. But the model itself still has to fit on each GPU.

This is the limitation that DeepSpeed solves. It’s nothing crazy. DeepSpeed has two main features that enable training much larger models than before: first, three-stage partitioning, and second, ZeRO offloading.

For three-stage partitioning — previously, every GPU stored a complete copy of all parameters, gradients, and optimizer states. DeepSpeed changes this so each GPU only stores a portion. Specifically, for any given layer, the parameters, gradients, and optimizer states are sliced up so each GPU only holds its zone. For example, if you have 8 GPUs and a layer has parameters of shape (3200, 1000), each GPU only stores (400, 1000) — its 1/8th slice. Same for gradients and optimizer states. They all have the same shape, just different per-element sizes: 2 bytes for parameters, 2 bytes for gradients, and 16 bytes for optimizer states. Through this partitioning, the per-GPU memory drops from about 20 bytes per parameter down to less than 1 byte per parameter.

That alone is impressive, but then comes the second feature. If you’ve ever used Google Colab for training, you’ve probably noticed something — CPU RAM and NVMe storage are huge but just sitting there doing nothing. That’s exactly where the idea comes from. ZeRO-Offload moves optimizer states from GPU to CPU. ZeRO-Infinity goes further and moves parameters, gradients, and optimizer states to CPU and NVMe, then streams them back to GPU only when needed for computation. There’s one catch though — GPU-to-GPU data transfer is fast (~900 GB/s via NVLink), but GPU-to-CPU transfer over PCIe is slow (~16 GB/s). The solution is elegant: instead of one GPU pulling everything from CPU by itself, all 8 GPUs each pull a small portion from CPU simultaneously, then share with each other over the fast GPU-to-GPU network. This minimizes how much data travels over the slow PCIe connection.

Now that we’ve covered the overall concepts, let’s look at the forward pass, backward pass, and weight update step in detail to see exactly what happens.

### Forward Pass
* **layer1:** Each GPU get layer1’s full param from CPU and get different bactch of input and do forward pass, and then discard layer1's params.
* **layer2:** Each GPU get layer2’s full param from CPU and get different bactch of input and do forward pass, and then discard layer2 params.
* **…**
* **layerN:** Each GPU get layerN’s full param from CPU and get different bactch of input and do forward pass, and then discard layerN params.

Like this, repeating this process, we get the activations from the final layer.

### Backward Pass
Previsouly, we got the activations from the last layer, and now we can do the backward pass.

**Layer N:**
Each GPU do followings: with the activations at the last layer(all gpus have different activations since they all got diff batch of input), and params(each gpu has a full copy) do the backward pass get the gradients(here each gpu would get different gradient since they all used different activations, but doing avging one universal gradient is made), and discard params of layer N. Doing these we have one full gradient for all gpus, following this, we slice them into N bocks so each gpu holds subset of gradients for layer N. This gradient part is keep staying on the each gpus.

So we do this for all layers, and as a result, each GPU have a gradient part for all layers.

### Weight Update
GPU send the gradient part to CPU, and since CPU was already holding params and optimizers, it can do weight update right there for layer N.

Like this one cycle of forward pass, backward pass and weight update ends. Following this, the updated parameters stay on CPU, ready to be pulled by GPUs in the next forward pass. and in the each GPU.

---

## Inference

### Overview
DeepSpeed Inference tackles the same core problem as training: models are huge, hardware is limited. But inference has unique challenges — latency matters, workloads are dynamic, and you need to serve many users simultaneously.

DeepSpeed provides different strategies depending on your hardware setup, plus general optimizations that apply everywhere:

### 1. General GPU-Level Optimizations (apply to any setup)
These optimizations work on every individual GPU regardless of whether you have a single GPU, single node, or multi-node cluster.

**Kernel Fusion**
Normally each operation (attention, bias, activation, layer norm, etc.) is a separate GPU program (“kernel”). Between each kernel, GPU writes results to memory and reads them back. This is slow.

```text
Before (separate kernels):
  Op1 → write to memory → read → Op2 → write to memory → read → Op3 → write → read → Op4 → write → read → Op5

After (fused):
  Op1+Op2+Op3+Op4+Op5 → single write to memory
```

DeepSpeed fuses 5–7 operations together, eliminating intermediate memory round trips. Result: 8–12x speedup.

Adaptive fusion based on batch size:
* **Small batch (1 request):** GPU is memory-bandwidth bound — computation is tiny but you still read all parameters. Fuse to minimize memory reads.
* **Large batch (100 requests):** GPU is compute bound — enough data to keep GPU cores busy. Fuse differently to maximize computational efficiency.

**INT8 Quantization**
Replace 16-bit matrix multiplications with 8-bit versions. Matrices are half the size → GPU processes them ~2x faster.

**KV Cache**
During token generation, the model caches key-value pairs from previous tokens so it doesn’t reprocess the entire sequence each time:

```text
Naive (without KV cache):
  Step 1: process [t1, t2, ... t20]           → generates t21
  Step 2: process [t1, t2, ... t20, t21]      → generates t22  (reprocesses everything!)

With KV cache:
  Step 1 (prompt): process [t1, t2, ... t20] → generates t21, saves KV for all 20 tokens
  Step 2: process ONLY [t21], use cached KV   → generates t22, adds t21's KV to cache
  Step 3: process ONLY [t22], use cached KV   → generates t23, adds t22's KV to cache
```
Each generation step only processes ONE new token. Much faster, but the cache grows with sequence length and eats GPU memory.

### 2. Single Node (one machine, multiple GPUs) → Tensor Parallelism
**What it does:** Splits individual weight matrices across GPUs within the same node.

**Example:** A layer has a weight matrix of shape [4096 × 4096]. With 4 GPUs:
```text
GPU 0 holds: [4096 × 1024]  ← 1/4 of the weight matrix
GPU 1 holds: [4096 × 1024]  ← 1/4 of the weight matrix
GPU 2 holds: [4096 × 1024]  ← 1/4 of the weight matrix
GPU 3 holds: [4096 × 1024]  ← 1/4 of the weight matrix
```
All GPUs compute their portion of the matrix multiply simultaneously, then combine results.

* **Why this works within a node:** GPUs inside the same machine are connected by NVLink (~900 GB/s). Tensor parallelism requires constant communication during every matrix multiplication, but NVLink is fast enough to handle it.
* **Why NOT use this across nodes:** Inter-node network is much slower. The constant back-and-forth communication would kill performance.

### 3. Multiple Nodes (multiple machines) → Pipeline Parallelism
**What it does:** Assigns whole layers to different nodes.
```text
Node 1: layers 1-10
Node 2: layers 11-20
Node 3: layers 21-30
```
* **Why this works across nodes:** Pipeline parallelism only communicates between nodes when one stage finishes and passes output to the next. Much less frequent than tensor parallelism’s constant communication.
* **Why NOT use this within a node:** Pipeline parallelism has “bubbles” (idle time) where GPUs wait. Tensor parallelism keeps all GPUs busy simultaneously, which is better when NVLink allows fast communication.

But pipeline parallelism has problems. DeepSpeed solves four of them:

**Problem 1: Bubbles (idle GPUs)**
When Node 1 finishes and passes data to Node 2, Node 1 sits idle.

```text
Without optimization (huge idle gaps):
Time →
Node 1:  [Batch1] [idle......] [idle......] [Batch2] [idle......] [idle......]
Node 2:           [Batch1]     [idle......]           [Batch2]     [idle......]
Node 3:                        [Batch1]                             [Batch2]
```
**DeepSpeed’s insight:** Different batches (different user requests) have ZERO dependency on each other. While Batch 1 is at Node 2, start Batch 2 at Node 1 immediately.

```text
With optimization (overlapping independent batches):
Time →
Node 1 (layers 1-10):   [Batch1] [Batch2] [Batch3] [Batch4]
Node 2 (layers 11-20):          [Batch1] [Batch2] [Batch3] [Batch4]
Node 3 (layers 21-30):                   [Batch1] [Batch2] [Batch3] [Batch4]
```
All nodes stay busy by continuously feeding new batches into the pipeline.

**Problem 2: Prompt vs Token Generation Imbalance**
LLM inference has two distinct phases:
* **Prompt phase:** Process all input tokens at once (e.g., 20 tokens). Compute-heavy, slow.
* **Token generation:** Generate one token at a time. Memory-heavy (read full params for just 1 token), fast.

This creates an imbalanced pipeline:
```text
Without optimization:
Node 1:  [=======LONG PROMPT=======] [tok] [tok] [tok]
Node 2:  [idle......................]  [=======LONG PROMPT=======] [tok] [tok]
```
Node 2 sits idle while Node 1 crunches through the entire prompt.

**DeepSpeed’s hybrid schedule — two parts:**

**Part 1: Smaller micro-batches during prompt phase**
Break the prompt into smaller chunks. Node 1 finishes the first chunk faster and starts feeding Node 2 sooner.
```text
With micro-batches (smaller bubbles):
Node 1: [5 tok][5 tok][5 tok][5 tok]
Node 2:        [5 tok][5 tok][5 tok][5 tok]
```
Node 2 starts working after just the first 5 tokens, not all 20. It’s like a restaurant: instead of cooking the entire 4-course meal then bringing everything at once (big wait), bring each course as it’s ready.

**Part 2: Fuse batches during token generation**
Token generation is memory-bound — reading parameters is the expensive part, not computation. So processing 1 token or 8 tokens costs almost the same:
```text
Without fusion (reading params twice, wasteful):
  Read all params (5ms) → process Batch A's token (0.1ms) → Read all params again (5ms) → process Batch B's token (0.1ms)
  Total: ~10.2ms

With fusion (read params once, process both):
  Read all params (5ms) → process Batch A's token + Batch B's token together (0.15ms)
  Total: ~5.15ms
```
It’s like opening a textbook — if two students need the same page, open it once and let both read, rather than opening it twice.
The hybrid: Small batches during prompt (reduce bubbles) + fused batches during generation (maximize throughput). Different strategy for each phase.

**Problem 3: KV Cache Memory**
During generation, the KV cache grows with every token and every batch. This eats GPU memory and limits how many requests you can serve simultaneously.

**DeepSpeed’s solution:** Offload parts of the KV cache to CPU memory using double buffering — prefetch the next batch’s cache while computing the current one.
```text
Without offload:
  GPU holds: model params + KV cache for all batches → runs out of memory

With offload:
  GPU holds: model params + KV cache for current batch
  CPU holds: KV cache for other batches (prefetching next one)
```
Result: ~30% larger batch size → 25% speedup in throughput.

**Problem 4: Dynamic Tensor Communication**
Between pipeline stages, nodes send tensors to each other. But in LLM inference, tensor sizes are unpredictable:
```text
Request 1: "Hi"                                → 2 tokens
Request 2: "Explain quantum physics in detail" → 8 tokens
Request 3: "Why?"                              → 3 tokens
```

Standard approach uses serialization/deserialization to handle variable sizes:
```text
Standard:
  Node 1: compute → serialize tensor (slow) → send
  Node 2: receive → deserialize (slow) → compute
```

**DeepSpeed’s solution:** Send metadata first (“I’m about to send shape [4, 512, 4096]”), then send raw data. The receiving node pre-allocates memory and receives directly:
```text
DeepSpeed:
  Node 1: compute → send metadata (tiny, fast) → send raw data (fast)
  Node 2: receive metadata → pre-allocate memory → receive raw data directly → compute
```
This eliminates serialization overhead and unnecessary CPU-to-GPU transfers.

### 4. Single GPU → ZeRO Inference
Same core idea as ZeRO training — use GPU + CPU + NVMe memory together. Model parameters are too big for one GPU, so they live on CPU/NVMe and get streamed to GPU layer by layer.
Result: Run 4x larger models on a single GPU compared to vanilla PyTorch.
This is the “democratization” angle — data scientists without multi-GPU clusters can still run massive model inference.

### 5. Sparse Models (MoE) Optimization
For Mixture of Experts models (where only some “expert” sub-networks activate per input), DeepSpeed combines all the dense optimizations above with specialized sparse parallelism and kernels.
Result: Real-time inference for models up to 3 trillion parameters at 25ms latency.

### Combined Results
* **Pipeline scheduling (filling bubbles):** ~2x
* **KV cache memory offload:** +25%
* **Communication optimization:** additional speedup
* **Transformer kernel fusion:** up to 4.3x
* **Total for 530B model:** ~4.3x throughput

For single GPU: 4x larger model support with 50%+ hardware utilization. For sparse/MoE: 7x improvement, trillion-scale models at real-time speed.

---

## Quantization

### Training Compression
Three techniques to make training faster:

1. **Sparse Attention Kernels:** In transformer models, the self-attention layer computes attention between every token and every other token. But most of these attention scores are actually close to zero — there’s intrinsic sparsity. DeepSpeed exploits this by skipping the near-zero computations. Result: you can train with 10x longer sequences at the same cost.
2. **Progressive Layer Dropping:** Training is iterative — you do forward and backward pass on every layer, every mini-batch. But do you really need to update every single layer every iteration? No. By randomly skipping some layers during training (sparsely updating), you get 2.5x faster training with similar accuracy on downstream tasks. Think of it like: not every layer needs to learn something new from every single batch.
3. **1-bit Compression (for communication):** When training across many GPUs, they need to communicate gradients. This communication is expensive. DeepSpeed’s 1-bit Adam compresses the communication by 26x, leading to 7x faster end-to-end training. Instead of sending full precision gradients between GPUs, you send heavily compressed versions.

### Inference Compression
This is where it gets really interesting. The goal is to make models smaller and faster for deployment.

**The DeepSpeed Compression Library**
It has two components:
* **Compression Composer:** Automatically combines different compression techniques (distillation, quantization, pruning) to get the best compressed model. Finding the right combination manually is hard — this automates it.
* **Inference Engine:** Generates optimized code for compressed models (like the INT8 fused kernels Reza talked about earlier).

**ZeroQuant — The Star of the Show**
Traditional quantization (Quantization-Aware Training / QAT) has a painful pipeline:
```text
Traditional QAT:
1. Need the original training data     → often unavailable (private/confidential)
2. Need many GPUs                       → expensive
3. Need hyperparameter tuning           → time-consuming
4. After all that → get quantized model
```

ZeroQuant solves this by being data-free AND GPU-free:
```text
ZeroQuant:
1. No training data needed
2. No GPUs needed for compression
3. No hyperparameter tuning
4. Zero training cost
```

How? Two key techniques:
* **Fine-grained quantization:** Instead of using one single scaling factor to convert an entire weight matrix from fp16 to int8, split the matrix into small groups and use a different scaling factor per group. This dramatically reduces quantization error.

```text
Traditional: entire matrix → one scaling factor → big error
ZeroQuant:   split into groups → scaling factor per group → small error
```

Results on GPT-NeoX (20B parameters):
* Traditional QAT would need 256 GPUs for 20 days — not affordable for most people
* ZeroQuant needs zero GPUs, zero data, zero training time
* Reduces inference from 2 GPUs to 1 GPU (latency from 65ms to 25ms)
* Accuracy degradation within 0.1% — negligible

**PR-MoE — Parameter Efficient Mixture of Experts**
The problem with standard MoE models: they have 8x more parameters than equivalent dense models. More parameters = more memory needed for training, slower inference (more parameters to load).

DeepSpeed introduces PR-MoE with two innovations:
1. **Pyramid structure:** Instead of using the same number of experts in every layer (standard MoE), use more experts in later layers, fewer in earlier layers. Inspired by CNNs which use more channels in deeper layers. This reduces the total number of experts needed.
2. **Residual MoE:** Instead of routing tokens to 1 or 2 experts (which requires expensive communication to find the right expert), use a fixed MLP + one expert. The fixed MLP handles the base computation, and the expert corrects/refines it. No extra communication cost.

Combined with MoS (MoE-specific knowledge distillation), PR-MoE reduces MoE parameters by 1.9x to 3.7x while maintaining similar accuracy.

---

## Additional Questions

**Q1: Yes, GPUs share gradients to get the average.** Remember from training — after backward pass, all GPUs do reduce-scatter to average their gradients. This requires sending gradient data between GPUs. Normally each gradient value is 16 or 32 bits. 1-bit Adam compresses each value down to just 1 bit (essentially just the sign — positive or negative direction) before sending. The receiving GPU then decompresses it. So the actual data traveling between GPUs is 16–32x smaller, which is why communication is 26x cheaper. The accuracy barely suffers because the sign of the gradient (which direction to move) is more important than the exact magnitude.

**Q2: This is the key misunderstanding.** Simply converting fp16 → int8 (naive quantization) is indeed fast and CPU can do it in seconds. But the problem is it destroys model accuracy. When you just chop precision from 16 bits to 8 bits, the rounding errors accumulate and the model outputs garbage.

That’s why traditional Quantization-Aware Training (QAT) exists. It doesn’t just compress — it retrains the model to learn to be accurate despite the lower precision:
```text
Naive quantization (seconds, but bad accuracy):
  Just round fp16 weights → int8. Done. Model accuracy drops significantly.

QAT (days, but good accuracy):
  1. Load model
  2. Run training with simulated int8 precision
  3. Model gradually learns to adjust its weights to work well in int8
  4. Try different hyperparameters (group sizes, which layers to quantize, etc.)
  5. Repeat until accuracy is acceptable
  6. Save quantized model
```
Step 2–5 is essentially retraining — you’re running forward pass, backward pass, weight updates, for many epochs. For a 20B model, that needs 256 GPUs for 20 days because it’s doing actual training, not just converting numbers.

ZeroQuant’s breakthrough is that its fine-grained group quantization is precise enough that naive quantization barely loses accuracy — no retraining needed. By using different scaling factors per small group of weights instead of one for the whole matrix, the rounding errors stay tiny enough that the model still works well.

**Q3: Residual MoE explained.**
First, understand standard MoE. In a standard MoE layer, each token gets routed to specific experts:
```text
Standard MoE (top-2 routing):
  Input token → Router says "send to Expert 3 and Expert 7"
  → Expert 3 processes token (on GPU 3)
  → Expert 7 processes token (on GPU 7)
  → Combine results
```
The problem: Expert 3 might be on GPU 3 and Expert 7 on GPU 7. So you need to send the token across GPUs to reach the right experts, then send results back. This communication is expensive, especially with top-2 routing where every token visits two experts on potentially different GPUs.

Residual MoE redesigns this:
```text
Residual MoE:
  Input token → Fixed MLP (exists on every GPU, no communication needed)
               → produces base output
             → ONE expert (local or minimal communication)
               → produces correction/refinement
             → Final output = base + correction
```
The fixed MLP is the same on every GPU — it’s not an expert, it’s a shared layer. Since every GPU has a copy, no communication needed for this part. It handles the bulk of the computation.
Then only ONE expert is used (not two), and it just refines what the MLP already computed. Less routing = less communication.

```text
Standard MoE:    route to 2 experts across GPUs → heavy communication
Residual MoE:    local MLP (no communication) + 1 expert → minimal communication
```
The expert acts like a “residual correction” — similar to skip connections in ResNet. The MLP gets you 90% of the way, the expert fine-tunes the remaining 10%. That’s why it’s called “residual” MoE.
```
