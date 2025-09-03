# Resume AI-Detection: Identifying Language Model-Assisted CVs

A machine learning project that detects whether resumes have been rewritten using Large Language Models (LLMs) to promote fairness and transparency in hiring processes.

## Problem Statement

Large Language Models like GPT-4 are increasingly used by job candidates to rewrite resumes, creating potential issues:
- **Factual inaccuracies** from AI hallucinations
- **Systemic hiring disparities** - candidates with access to advanced LLMs gain unfair advantages
- **Lack of visibility** for employers into resume authenticity

Our solution provides employers and ATS providers with tools to identify AI-assisted resumes and apply consistent hiring standards.

## Dataset

**Source**: [Djinni Recruitment Dataset](https://huggingface.co/datasets) (Hugging Face)
- **Focus**: Project Manager roles (most frequent in dataset)
- **Original CVs**: 3,622 human-written resumes
- **AI-Generated CVs**: 3,253 LLM-rewritten versions using 10 distinct prompts
- **Total Dataset**: ~7,000 labeled resume samples

### Prompt Diversity Strategy
We created 10 different rewriting prompts to ensure model robustness:
- Eye-catching resume style
- Anti-LLM tone (humble, matter-of-fact)
- Tailored to AI startups
- Soft skill focused
- ATS-friendly keyword optimization
- Professional fluency enhancement

## Models Implemented

| Model | Description | Best Accuracy |
|-------|-------------|---------------|
| **Baseline** | Majority class predictor | ~67% |
| **TF-IDF + Logistic Regression** | Shallow lexical features | 97.24% |
| **GloVe + MLP** | 300D embeddings + neural network | 89.97% |
| **DistillBERT + MLP (Fully Tuned)** | Fine-tuned transformer | **98.99%** |
| **DistillBERT + MLP (Last 2 Layers)** | Efficient fine-tuning | 97.33% |

## Key Results

### Performance Summary (9-Prompt Mixed Test Set)
| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| **DistillBERT + MLP (Fully Tuned)** | **0.9899** | **0.9899** | **0.9992** |
| DistillBERT + MLP (Reduced Classifier) | 0.9834 | 0.9834 | 0.9985 |
| DistillBERT + MLP (Last 2 Layers) | 0.9733 | 0.9733 | 0.9980 |
| TF-IDF + Logistic Regression | 0.9724 | 0.9724 | 0.9978 |
| GloVe + MLP | 0.8997 | 0.8997 | 0.9642 |

### Critical Discovery: Prompt Diversity Impact
- **Without prompt diversity**: Models achieved ~99% accuracy but failed on novel prompts (dropped to 71%)
- **With 9-prompt training**: Robust performance across diverse writing styles
- **Key insight**: Model robustness depends as much on training data diversity as model complexity

## Project Structure

```
├── 242b_final_project.ipynb          # Main project notebook & EDA (176 KB)
├── combine_data.ipynb                # Data combination and labeling (133 KB)
├── New_PATHA_Training.ipynb          # GloVe + MLP model training (264 KB)
├── PATHB_Training.ipynb              # TF-IDF + DistillBERT model training (6,187 KB)
├── Prompt_Process_Shuffle.ipynb      # Prompt processing and data shuffling (256 KB)
├── Rewrite_Resume.ipynb              # Resume rewriting with LLM prompts (49 KB)
├── 242B_Visualization.ipynb          # Results visualization and analysis (2,480 KB)
└── README.md
```

### Workflow Organization

**Data Preprocessing**
1. **EDA**: `242b_final_project.ipynb`
2. **Resume Rewriting**: `Rewrite_Resume.ipynb`
3. **Data Processing**: 
   - `combine_data.ipynb`
   - `Prompt_Process_Shuffle.ipynb`

**Model Training**
1. **Path-A** (GloVe + AvgPool + MLP): `New_PATHA_Training.ipynb`
2. **Path-B** (TF-IDF + Logistic / DistillBERT + MLP): `PATHB_Training.ipynb`

**Analysis & Results**
- **Visualization**: `242B_Visualization.ipynb`

## Model Insights

### Feature Importance by Model Type
- **Logistic Regression**: Geographic indicators (state-based features)
- **Random Forest**: Clinical flags (bipolar, schizophrenia, education, marital status)
- **XGBoost**: Anxiety and trauma-related disorders, education level
- **DistillBERT**: Contextual patterns in tone, fluency, and professional terminology

### Token Attribution Analysis
Our transformer models identified key linguistic patterns:
- **LLM-generated**: "eager", "passionate", "fluent", "assessment", "mitigation"
- **Human-written**: "liaison", "evaluation", varied but precise language

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn transformers torch
pip install matplotlib seaborn plotly shap
```

### Quick Start
1. **Data Preprocessing**:
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train Models**:
   ```bash
   # Baseline models
   python src/model_training.py --model baseline
   
   # Transformer models
   python src/model_training.py --model distilbert --fine_tune full
   ```

3. **Evaluate Performance**:
   ```bash
   python src/evaluation.py --model_path results/best_model.pkl
   ```

## Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: Avoiding false positives (important for protecting authentic candidates)
- **Recall**: Detecting AI-generated resumes consistently
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Model's ability to distinguish between classes

## Visualizations & Analysis

The project includes comprehensive visualizations and analysis:
- Confusion matrices for all models
- PCA/UMAP embeddings analysis
- SHAP feature attribution plots
- Prompt-wise performance breakdowns
- Token importance rankings
- Model comparison across different prompt styles

## Future Work

- **Multi-class detection**: Distinguish between different LLM providers (GPT-4, Claude, etc.)
- **Adversarial testing**: Evaluate against sophisticated prompt engineering
- **Real-time API**: Deploy for integration with ATS platforms
- **Continual learning**: Adapt to evolving LLM writing patterns
- **Attention visualization**: Enhanced interpretability for hiring decisions

## Applications

### For Employers & Recruiters
- Maintain visibility into resume authenticity
- Apply consistent hiring standards
- Identify potential factual inaccuracies from AI hallucinations

### For ATS Providers
- Integrate AI-detection capabilities
- Support fair candidate screening
- Provide transparency tools for hiring teams

### For Candidates
- Protect authentic job seekers from AI-driven disadvantages
- Promote equitable hiring practices

## Ethical Considerations

This tool aims to promote fairness in hiring by:
- Preventing systematic disadvantages for candidates without LLM access
- Maintaining transparency in the application process  
- Supporting authentic representation of candidate qualifications

**Important**: False positives could unfairly penalize legitimate candidates. Our high-precision models minimize this risk while maintaining strong detection capabilities.

## Technical Performance

**Best Model**: DistillBERT + MLP (Fully Tuned)
- **Accuracy**: 98.99%
- **F1-Score**: 98.99% 
- **AUC**: 99.92%
- **Robust across**: 9 different prompt styles and writing approaches

---

**Key Takeaway**: Combining advanced transformer architectures with diverse, prompt-aware training data enables reliable detection of AI-generated resume content while maintaining fairness for all candidates.
