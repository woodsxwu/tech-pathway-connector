# tech-pathway-connector
a novel hybrid approach combining graph theory and machine learning to identify  critical skill pathways across technical job roles.
# Dataset
https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024?select=job_skills.csv

#Data Cleaning

NLP-enhanced preprocessing for analyzing skill pathways between tech roles.

## Setup

```bash
pip install pandas numpy spacy scikit-learn nltk
python -m spacy download en_core_web_md
```

## Usage

```python
from data_cleaning import nlp_enhanced_analysis
import pandas as pd

# Load data
job_postings = pd.read_csv('linkedin_job_postings.csv')
job_skills = pd.read_csv('job_skills.csv')

# Specify transitions to analyze
focus_transitions = [
    ('backend_developer', 'devops_engineer'),
    ('mobile_developer', 'fullstack_developer')
]

# Run analysis
results = nlp_enhanced_analysis(job_postings, job_skills, focus_transitions)

# Save results
results['integrated_data'].to_csv('cleaned_data/integrated_data.csv', index=False)
```

## Input Format

- `linkedin_job_postings.csv`: Contains `job_link`, `job_title`, `company`, etc.
- `job_skills.csv`: Contains `job_link` and `job_skills` (comma-separated)

## Output

- `job_postings_clean`: Standardized job titles and classifications
- `job_skills_clean`: Normalized skills
- `integrated_data`: Jobs linked with skills
- `role_skills`: Skills mapped to roles with importance scores
- `transition_analyses`: Pathway analysis between specified role pairs

## Next Steps

Use cleaned data to implement:
1. Baseline frequency analysis
2. MST algorithm
3. TechBridge hybrid algorithm