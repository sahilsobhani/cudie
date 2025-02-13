# 🚀 CUDA Programming Journey

Welcome to my **CUDA Programming Journey**! This repository is a step-by-step guide documenting my learning and experiments with **NVIDIA CUDA** for parallel programming on GPUs. Here, you will find examples, explanations, and code to get started with CUDA programming from scratch.

## 🌟 Goals of This Repository
- Learn the basics of CUDA programming
- Solve computational problems using GPUs
- Optimize performance using CUDA techniques
- Share my progress and help others learn

---

## 📚 What is CUDA?

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It enables developers to use NVIDIA GPUs for general-purpose computing.

CUDA allows us to leverage the massive parallel processing power of GPUs to accelerate computations that would be slower on CPUs.

---

## 📁 Repository Structure

```
📦 cuda-learning-repo
 ┣ 📂 basics
 ┃ ┣ 📜 01_hello_world.cu        # First CUDA program (Hello, GPU!)
 ┃ ┣ 📜 02_matrix_multiplication.cu # Basic matrix multiplication on GPU
 ┃ ┗ 📜 ...
 ┣ 📂 optimized
 ┃ ┣ 📜 shared_memory_example.cu # Using shared memory for performance
 ┃ ┗ 📜 ...
 ┗ 📜 README.md                 # This file
```

---

## 🛠️ Prerequisites

Before running any of the scripts, ensure you have the following installed:
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** (Download it [here](https://developer.nvidia.com/cuda-downloads))
- **nvcc** compiler (comes with the CUDA Toolkit)
- Basic knowledge of C/C++ programming

To verify your CUDA installation, run:
```bash
nvcc --version
```

---

## 🚀 How to Compile and Run CUDA Programs

1. **Write your CUDA code** in a `.cu` file (e.g., `my_program.cu`).
2. **Compile the code** using the `nvcc` compiler:
   ```bash
   nvcc my_program.cu -o my_program
   ```
3. **Run the executable**:
   ```bash
   ./my_program
   ```

### Example:
```bash
nvcc 01_hello_world.cu -o hello_world
./hello_world
```

---

## ✨ Current Progress

| **Program**               | **Description**                              | **Status** |
|---------------------------|----------------------------------------------|------------|
| `01_hello_world.cu`        | First CUDA program                           | ✅ Complete |
| `02_matrix_multiplication.cu` | Matrix multiplication on GPU                | ✅ Complete |
| `shared_memory_example.cu` | Optimized matrix multiplication using shared memory | 🚧 In Progress |

---

## 🔥 Learning Milestones
- [x] **Understand CUDA architecture**
- [x] **Write a basic CUDA kernel**
- [x] **Learn about thread hierarchy (blocks and threads)**
- [x] **Matrix multiplication using CUDA**
- [ ] Optimize with shared memory
- [ ] Use CUDA streams for concurrency
- [ ] Explore advanced CUDA features

---

## 📖 Resources
Here are some helpful resources that I'm following:
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Parallel Programming with CUDA (Udacity Course)](https://www.udacity.com/course/intro-to-parallel-programming--cs344)

---

## 🙌 Contributing
This is a personal learning project, but feel free to open an issue or suggest improvements if you find something interesting!

---

## 🏆 Acknowledgments
- NVIDIA for providing the CUDA Toolkit
- OpenAI's ChatGPT for helping with explanations and code

---

## 📄 License
This repository is licensed under the MIT License. Feel free to use and modify the code.
