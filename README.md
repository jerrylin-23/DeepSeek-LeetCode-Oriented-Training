# LeetCode Model Benchmark

Benchmark and compare LLMs on LeetCode coding problems. Fine-tune your own model and measure improvement against the base model.

## 🏆 Results

| Model | Overall | Easy | Medium | Hard |
|-------|---------|------|--------|------|
| deepseek-coder:6.7b-base | 24% | 30.3% | 32.4% | 9.1% |
| **deepseek-leetcode (3-epoch fine-tuned)** | **34%** | **52%** | **27.8%** | **28.6%** |

Fine-tuning improved overall accuracy by **42%** (24% → 34%), with major gains on Easy (+72%) and Hard (+214%) problems.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
brew install ollama
ollama serve  # Keep running in background
```

### 2. Pull Base Model
```bash
ollama pull deepseek-coder:6.7b-base
```

### 3. Run Benchmark
```bash
python run_benchmark.py -n 10  # Test 10 problems
python run_benchmark.py -n 100  # Full benchmark
```

## 🎓 Fine-Tune Your Own Model

### Option 1: Kaggle (Recommended for 6.7B model)
1. Upload `notebooks/train_kaggle.ipynb` to [Kaggle](https://www.kaggle.com/)
2. Enable GPU (P100 or T4x2)
3. Run all cells (~1-2 hours for 3 epochs)
4. Download the merged model from Hugging Face

### Option 2: Google Colab
Works for smaller models or with Colab Pro for larger memory.

### Convert & Load Fine-Tuned Model
```bash
# Convert to GGUF (requires llama.cpp)
python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile model.gguf --outtype q8_0

# Import to Ollama
ollama create deepseek-leetcode -f Modelfile

# Benchmark your model
python run_benchmark.py -m deepseek-leetcode -n 100
```

## Project Structure

```
├── run_benchmark.py          # CLI entry point
├── config.py                 # Model & benchmark configuration
├── Modelfile                 # Ollama import config
├── merge_and_convert.py      # Model merging utilities
├── notebooks/
│   └── train_kaggle.ipynb    # Kaggle training notebook
├── src/
│   ├── benchmark.py          # Benchmark orchestrator
│   ├── dataset.py            # LeetCode problem loader
│   ├── executor.py           # Sandboxed code execution
│   ├── ollama_client.py      # Local Ollama inference
│   ├── deepseek_client.py    # DeepSeek API client
│   └── reporter.py           # Results & report generator
├── results/                  # Benchmark outputs
└── data/                     # Cached problems (auto-generated)
```

## Configuration

Edit `config.py` to customize:
- Models to benchmark
- Number of problems per difficulty
- Execution timeouts
- Ollama base URL

## 📝 Key Takeaways

### Dataset Quality Matters More Than Size

My first attempt used a **general coding dataset** (mix of languages, styles, and tasks). The result? **Performance actually got worse** than the base model. The model learned generic patterns that conflicted with LeetCode's specific format.

Switching to [LongQ/leetcode_python](https://huggingface.co/datasets/LongQ/leetcode_python) — a curated dataset of 2,369 LeetCode problems with Python solutions — made all the difference. The model learned:
- LeetCode's exact input/output format
- Python-specific idioms for competitive programming
- Common algorithmic patterns (DP, BFS/DFS, two pointers, etc.)

**Lesson:** Domain-specific, high-quality data beats larger generic datasets.

### Training Configuration Insights

| Setting | Value | Why |
|---------|-------|-----|
| `dataset_text_field="text"` | Required | SFTTrainer needs explicit field mapping — missing this caused garbage output |
| `max_seq_length=2048` | Prevents OOM | LeetCode problems + solutions fit within this limit |
| `packing=False` | Cleaner training | Packing multiple examples caused format confusion |
| `gradient_checkpointing=True` | Memory savings | Essential for 6.7B model on 16GB GPU |
| `load_in_4bit=True` (QLoRA) | 4x memory reduction | Enables training on consumer GPUs |

### What Improved (and What Didn't)

**Big wins:**
- **Easy problems:** 30% → 52% (+72%) — model learned common patterns well
- **Hard problems:** 9% → 29% (+214%) — surprising improvement on complex algorithms

**Unexpected:**
- **Medium problems:** 32% → 28% (-13%) — slight regression, possibly overfitting to extremes

### LoRA vs Full Fine-Tuning

Used **LoRA (Low-Rank Adaptation)** to train only ~40M parameters (1.1% of the model) instead of all 6.7B:
- 🚀 Training time: ~5 hours on a single T4 GPU
- 💾 VRAM usage: ~14GB (vs 50GB+ for full fine-tuning)
- 📦 Adapter size: ~160MB (easily shareable)

The merged FP16 model performs identically to a fully fine-tuned version for this task.

## License

MIT
