# tech-pathway-connector
a novel hybrid approach combining graph theory and machine learning to identify  critical skill pathways across technical job roles.
# Dataset
https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024?select=job_skills.csv

# Data Cleaning

1. Copy the script to your environment
2. Run it:
   ```python
   %run clean_tech_jobs_final.py
   ```
3. If prompted for file paths, enter them
4. Get your cleaned data in `tech_jobs_with_skills.csv`

That's it.

## Tech Role Categories

The script specifically identifies these tech roles:
- frontend_developer
- backend_developer
- fullstack_developer
- data_scientist
- data_engineer
- devops_engineer
- mobile_developer
- security_engineer
- qa_engineer
- software_engineer
- cloud_architect
- data_analyst
- product_manager
- technical_support

Jobs outside these specific categories are excluded from the output.

## Output

The script generates a CSV file with these columns:
- `job_id`: The job identifier
- `title`: The job title
- `role_category`: The specific tech role category
- `skills`: The list of normalized skills for each job