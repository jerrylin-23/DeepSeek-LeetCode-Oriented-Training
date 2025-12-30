# LeetCode Model Benchmark

Benchmark and compare LLMs on LeetCode coding problems. Fine-tune your own model and measure improvement against the base model.

## 🏆 Results

| Model | Overall | Easy | Medium | Hard |
|-------|---------|------|--------|------|
| deepseek-coder:6.7b-base | 28% | 28% | 31.5% | 19% |
| **deepseek-leetcode (3-epoch fine-tuned)** | **34%** | **52%** | **27.8%** | **28.6%** |

Fine-tuning improved performance by **21%** overall, with significant gains on Easy (+85%) and Hard (+50%) problems.

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

## License

MIT
