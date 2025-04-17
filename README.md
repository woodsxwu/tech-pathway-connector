# Tech Pathway Connector

A graph-based approach to identify critical skill pathways across technical job roles using network analysis and skill similarity metrics.

## Overview

This project provides tools for analyzing and visualizing career transition paths in the tech industry. It helps identify optimal skill acquisition strategies when transitioning between different technical roles through graph-based analysis and skill similarity calculations.

## Features

- **Job Similarity Analysis**: Calculate similarity between different tech roles based on required skills using TF-IDF and cosine similarity
- **Career Path Planning**: Identify optimal skill acquisition paths for career transitions using graph-based algorithms
- **Interactive Visualization**: Visualize career paths and skill requirements using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tech-pathway-connector.git
cd tech-pathway-connector
```

2. Create and activate a virtual environment:
```bash
python -m venv tech_pathway_env
source tech_pathway_env/bin/activate  # On Windows: tech_pathway_env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the LinkedIn Jobs and Skills dataset from Kaggle:
https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024?select=job_skills.csv

## Data Processing

1. Place the dataset in the `data` directory
2. Run the data cleaning script:
```bash
python src/data_cleaning.py
```
3. The cleaned data will be saved as `tech_jobs_with_skills.csv`

## Tech Role Categories

The system analyzes the following technical roles:
- Frontend Developer
- Backend Developer
- Fullstack Developer
- Data Scientist
- Data Engineer
- DevOps Engineer
- Mobile Developer
- Security Engineer
- QA Engineer
- Software Engineer
- Cloud Architect
- Data Analyst
- Product Manager
- Technical Support

## Usage

### Job Similarity Analysis

To analyze similarity between different tech roles:
```bash
python src/job_similarity.py
```

### Career Path Planning

To use the interactive career path planning tool:
```bash
streamlit run src/algorithms_compare.py
```

## Output

The system generates various outputs:

1. **Cleaned Data** (`tech_jobs_with_skills.csv`):
   - `job_id`: Job identifier
   - `title`: Job title
   - `role_category`: Tech role category
   - `skills`: List of normalized skills

2. **Analysis Results**:
   - Role similarity scores
   - Skill acquisition paths
   - Skill distribution visualizations

## Project Structure

```
tech-pathway-connector/
├── data/                    # Data files
├── src/                     # Source code
│   ├── algorithms_compare.py    # Career path comparison
│   ├── data_cleaning.py         # Data processing
│   ├── job_similarity.py        # Role similarity
│   └── strategies/              # Strategy implementations
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue in the repository.