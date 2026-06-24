# Candidate Discovery & Ranking System (PoC)

This repository contains the code and resources for a robust, high-performance candidate ranking system developed for the **Senior AI Engineer — Founding Team** role at Redrob AI.

The system is engineered to run **completely offline** within a **5-minute wall-clock CPU limit** (using under 8 GB RAM) on a dataset of 100,000 candidates.

## Features & Architecture

The ranker uses a **Two-Stage Pipeline** to balance high-quality semantic understanding with extreme efficiency:

1.  **Stage 1: Coarse Filters & Heuristics**
    -   Filters out anomalies and physical reality violations (honeypots, overlapping current roles, dates in the future).
    -   Discards candidates with non-technical trap titles or target locations outside India (without relocation intent).
    -   Scores the remaining pool using a fast keyword and experience heuristic, selecting the **top 1,500 candidates** for Stage 2.
2.  **Stage 2: Deep Semantic Scoring**
    -   Computes dense semantic embeddings using a local `all-MiniLM-L6-v2` sentence-transformer model (cached offline in `./model_cache`).
    -   Computes specific sub-scores for critical skill match (Vector DBs, embedding models, NLP, MLOps) with exponential recency decay, experience fit, and employer quality.
    -   Applies custom modifiers: consulting career penalty, pure academic/research penalty, LangChain-only/shallow AI penalty, and non-coding tech lead penalty.
    -   Applies behavioral signals (response rate, notice period, active recency, profile completeness) as a multiplier.

## Installation & Setup

Ensure Python 3.11 is installed, then run:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Ranker

### General Command (For Judges / General Execution)
To run the ranking step, use the following command by passing the path to the candidates file and your desired output path:

```bash
python rank.py --candidates <path_to_candidates.jsonl> --out <path_to_submission.csv>
```

### Local Workspace Execution (For Your Machine)
From the repository folder (`C:\Users\Ayush\Desktop\Ai recruiter candidate ranking System`), you can run the ranker on your local dataset using:

```bash
python rank.py --candidates "..\Redrob\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --out "submission.csv"
```

This will output two files:
-   `submission.csv`: Exactly 100 candidates ranked, containing the columns: `candidate_id,rank,score,reasoning`.
-   `submission_detailed.csv`: A comprehensive 12-column file containing sub-score decompositions, reliability metrics, and debug flags for explainability.

## Validation

### General Command (For Judges / General Execution)
To run the official submission validator:

```bash
python <path_to_validate_submission.py> <path_to_submission.csv>
```

### Local Workspace Execution (For Your Machine)
From the repository folder, run the validator using:

```bash
python "..\Redrob\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py" "submission.csv"
```
