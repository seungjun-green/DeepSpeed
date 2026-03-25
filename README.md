# DeepSpeed Implementation Suite

This repository provides comprehensive guides, theoretical foundations, and practical implementation notebooks for leveraging **DeepSpeed** to optimize Large Language Model (LLM) training and inference.

---

## Contents

### 1. Theory & Fundamentals
* **[Theory.md](Theory.md)**: A deep dive into the underlying mechanisms of DeepSpeed, including ZeRO (Zero Redundancy Optimizer) stages 1, 2, and 3, memory optimization techniques, and parallelism strategies.

### 2. Training Examples
* **[Fine-tuning Llama-7B with GSM8K](https://github.com/seungjun-green/DeepSpeed/blob/main/Deep_Speed_Sample_Script_train_7B_model_with_GSM8K.ipynb)**
    * **Goal**: Train a Llama-7B Instruct model on the GSM8K dataset.
    * **Focus**: Efficient memory management during the fine-tuning process using DeepSpeed ZeRO stages.

### 3. Inference & Optimization
* **[Large Model Inference on Limited Hardware](https://github.com/seungjun-green/DeepSpeed/blob/main/DeepSpeed_Inference_test_V1.ipynb)**
    * **Goal**: Run models larger than standard VRAM capacity (e.g., fitting a massive model on a single A10 40GB GPU).
    * **Focus**: Utilizing DeepSpeed-Inference to enable model sharding and offloading.

* **[Performance Benchmarking (V2)](https://github.com/seungjun-green/DeepSpeed/blob/main/Deep_Speed_Inference_Test_V2.ipynb)**
    * **Goal**: Measure the efficiency and speed of DeepSpeed Inference.
    * **Metrics**: 
        * TTFT (Time To First Token)
        * Throughput (tokens/sec)
        * Latency and memory consumption analysis.
