# VLMEvalKit Evaluation Tutorial

This guide explains how to use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for Vision-Language Model evaluation.

## 1. Installation

```bash
# Recommended: clone the latest repository
git clone https://github.com/open-compass/VLMEvalKit.git

cd VLMEvalKit
pip install -e .

# For Qwen models, check if there are additional dependencies in the Qwen repository
```

## 2. Environment Configuration

```bash
cd VLMEvalKit
vim .env

# Add the following content:
OPENAI_API_KEY=<your_api_key>
```

## 3. Prepare Data

Reference data downloading scripts in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

## 4. Eval
```bash
bash eval_all.sh
```

# To reuse previous results (from the previous day), add --reuse flag
```
