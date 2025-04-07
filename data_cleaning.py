import pandas as pd
import re
import os
import kagglehub
from collections import Counter
import time

def identify_tech_roles(job_title):
    """
    Identify if a job title belongs to a specific tech role
    Returns the tech category if found, None otherwise
    """
    # Convert to lowercase for case-insensitive matching
    title = str(job_title).lower()

    non_tech_patterns = [
        r'\b(mechanic|automotive|technician|repair|oil change)\b',
        r'\b(nurse|medical|healthcare|patient|clinical)\b',
        r'\b(sales associate|cashier|retail|store)\b',
        r'\b(sales|account executive|business development|customer success|talent acquisition)\b',
        r'\b(manufacturing|industrial|production|factory|assembly|fabrication)\b',
        r'\b(food|manufacturing|production|industrial|factory)\b',
        r'\b(gmp|haccp|fda|safety|selenium|food safety)\b',
    ]
    
    for pattern in non_tech_patterns:
        if re.search(pattern, title):
            return None

    # Define tech role patterns with clear boundaries to prevent partial matches
    tech_patterns = {
        'ai_engineer': r'\b(ai engineer|generative ai|llm engineer|machine learning engineer|ml engineer|ml ops|mlops|ai infrastructure)\b',
        'frontend_developer': r'\b(front.?end|ui developer|react developer|angular developer|vue developer|javascript developer|typescript developer|ui engineer)\b',
        'backend_developer': r'\b(back.?end|api developer|django developer|flask developer|node developer|java developer|python developer|golang developer|ruby developer|php developer|scala developer|rust developer|c\+\+ developer|api engineer)\b',
        'fullstack_developer': r'\b(full.?stack|mean stack|mern stack|lamp stack|web developer|full stack)\b',
        'mobile_developer': r'\b(mobile|ios|android|flutter|react native|swift developer|mobile app)\b',
        'devops_engineer': r'\b(devops|site reliability|platform engineer)\b',
        'cloud_engineer': r'\b(cloud engineer|aws engineer|azure engineer|gcp engineer)\b',
        'qa_engineer': r'\b(qa engineer|quality assurance|test engineer|automation test|sdet)\b',
        'embedded_engineer': r'\b(embedded|firmware|hardware engineer|iot engineer)\b',
        'database_engineer': r'\b(database engineer|dba|database administrator|sql developer)\b',
        'data_analyst': r'\b(data analyst|business intelligence|bi developer|analytics)\b',
        # 'technical_support': r'\b(technical support|it support|help desk|system admin)\b',
    }

    # Order matters - check specialized roles first
    for role, pattern in tech_patterns.items():
        if re.search(pattern, title):
            return role

    # No specific tech role found - return None
    return None

def is_generic_skill(skill):
    """
    Identify if a skill is generic or non-technical
    Returns True if the skill is generic, False otherwise
    """
    # Make sure we're comparing lowercase strings
    skill_lower = skill.lower().strip()

    # More comprehensive generic skills set
    generic_skills = {
        # Soft skills
        'communication', 'communication skill', 'communication skills',
        'effective communication', 'clear communication', 'verbal communication',
        'written communication', 'teamwork', 'team work', 'team player', 'team building',
        'problem solving', 'problem-solving', 'problem solver',
        'leadership', 'time management', 'creativity', 'critical thinking',
        'adaptability', 'collaboration', 'attention to detail', 'project management',
        'organization', 'self-motivated', 'self motivated', 'analytical skills',
        'analytical thinking', 'interpersonal skills', 'interpersonal', 'problemsolving',
        'troubleshooting',

        # Employment conditions & requirements
        'cleanbackground', 'backgroundcheck', 'drugtesting', 'drugtest',
        'cleandriving', 'cleanrecord', 'availabletowork', 'workweekends',
        'reliability', 'dependability', 'punctuality', 'flexibility',

        # Generic business terms
        'business', 'strategy', 'analytics', 'reporting', 'scheduling', 'training',
        'mentoring', 'consulting', 'operations', 'planning', 'documentation',
        'management', 'customer service', 'client relations', 'relationship building',

        # Too generic tech terms
        'computer', 'technology', 'technical', 'it', 'software', 'hardware', 'programming',
        'coding', 'development', 'engineering', 'testing', 'debugging', 'computer science',
        'software development', 'risk management', 'quality assurance',

        # Generic tools
        'microsoft office', 'office', 'word', 'excel', 'powerpoint', 'outlook', 'email',
        'presentation', 'spreadsheet', 'microsoft', 'office suite',

        # Employee benefits
        'paid time off', 'pto', 'benefits', 'vacation', 'health insurance',
        'retirement', '401k', 'bonus', 'compensation',

        # existing items...
        'bachelors degree', 'bachelor degree', 'bachelor', 'masters degree', 
        'phd', 'doctorate', 'associates degree', 'certification',
        'degree', 'diploma', 'education', 
    }

    # Check if skill contains common generic skill keywords
    generic_keywords = ['skill', 'ability', 'proficiency', 'knowledge', 'experience', 'expertise']
    contains_generic_keyword = any(keyword in skill_lower for keyword in generic_keywords)

    # Check if skill is in the generic list, is punctuation, or is very short
    return (skill_lower in generic_skills or
            len(skill_lower) <= 2 or  # Very short abbreviations
            skill_lower.isdigit() or   # Just numbers
            skill_lower in [',', '.', ';', ':', '-'] or  # Common punctuation
            contains_generic_keyword)  # Contains generic keywords

def normalize_skill(skill):
    """
    Normalize and standardize skill names to group related skills
    Returns the normalized skill name
    """
    # Map of skill variants to their canonical names
    skill_mappings = {
        # JavaScript ecosystem
        'js': 'javascript',
        'javascript': 'javascript',
        'ecmascript': 'javascript',
        'typescript': 'typescript',
        'ts': 'typescript',

        # React ecosystem
        'react': 'react',
        'react.js': 'react',
        'reactjs': 'react',
        'react native': 'react native',  # Keep React Native separate due to mobile focus

        # Vue ecosystem
        'vue': 'vue',
        'vue.js': 'vue',
        'vuejs': 'vue',
        'vuex': 'vue',

        # Angular ecosystem
        'angular': 'angular',
        'angular.js': 'angular',
        'angularjs': 'angular',
        'ng': 'angular',

        # Node.js ecosystem
        'node': 'node.js',
        'node.js': 'node.js',
        'nodejs': 'node.js',
        'express': 'express.js',
        'express.js': 'express.js',
        'expressjs': 'express.js',

        # Python ecosystem
        'py': 'python',
        'python': 'python',
        'django': 'django',
        'flask': 'flask',
        'fastapi': 'fastapi',

        # Java ecosystem
        'java': 'java',
        'spring': 'spring',
        'spring boot': 'spring boot',
        'springboot': 'spring boot',
        'j2ee': 'java ee',
        'java ee': 'java ee',
        'javaee': 'java ee',

        # Cloud platforms
        'aws': 'aws',
        'amazon web services': 'aws',
        'ec2': 'aws ec2',
        's3': 'aws s3',
        'lambda': 'aws lambda',
        'azure': 'azure',
        'microsoft azure': 'azure',
        'msft azure': 'azure',
        'gcp': 'gcp',
        'google cloud': 'gcp',
        'google cloud platform': 'gcp',

        # DevOps tools
        'kubernetes': 'kubernetes',
        'k8s': 'kubernetes',
        'docker': 'docker',
        'terraform': 'terraform',
        'ansible': 'ansible',
        'jenkins': 'jenkins',
        'ci/cd': 'ci/cd',
        'cicd': 'ci/cd',
        'continuous integration': 'ci/cd',
        'continuous deployment': 'ci/cd',

        # Databases
        'sql': 'sql',
        'mysql': 'mysql',
        'postgresql': 'postgresql',
        'postgres': 'postgresql',
        'postgre': 'postgresql',
        'oracle': 'oracle db',
        'oracle database': 'oracle db',
        'mongodb': 'mongodb',
        'mongo': 'mongodb',
        'nosql': 'nosql',
        'redis': 'redis',

        # Mobile development
        'ios': 'ios',
        'android': 'android',
        'swift': 'swift',
        'kotlin': 'kotlin',
        'flutter': 'flutter',
        'dart': 'dart',

        # Front-end technologies
        'html': 'html',
        'html5': 'html',
        'css': 'css',
        'css3': 'css',
        'sass': 'sass/scss',
        'scss': 'sass/scss',
        'less': 'less',

        # Data Science & ML
        'machine learning': 'machine learning',
        'ml': 'machine learning',
        'deep learning': 'deep learning',
        'dl': 'deep learning',
        'tensorflow': 'tensorflow',
        'tf': 'tensorflow',
        'pytorch': 'pytorch',
        'scikit-learn': 'scikit-learn',
        'sklearn': 'scikit-learn',
        'nlp': 'nlp',
        'natural language processing': 'nlp',
        'computer vision': 'computer vision',
        'cv': 'computer vision',

        # Data Analysis
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'jupyter': 'jupyter',
        'tableau': 'tableau',
        'power bi': 'power bi',
        'powerbi': 'power bi',

        # Version Control
        'git': 'git',
        'github': 'github',
        'gitlab': 'gitlab',
        'bitbucket': 'bitbucket',

        # Testing
        'unit testing': 'unit testing',
        'integration testing': 'integration testing',
        'e2e testing': 'e2e testing',
        'end-to-end testing': 'e2e testing',
        'selenium': 'selenium',
        'cypress': 'cypress',
        'jest': 'jest',
        'mocha': 'mocha',
        'chai': 'chai',

        # API
        'rest': 'rest api',
        'rest api': 'rest api',
        'restful': 'rest api',
        'restful api': 'rest api',
        'graphql': 'graphql',
        'soap': 'soap api',
        'soap api': 'soap api',

        # AI/ML specialized
        'llm': 'large language models',
        'large language model': 'large language models',
        'large language models': 'large language models',
        'generative ai': 'generative ai',
        'transformers': 'transformers',
        'gpt': 'gpt',
        'chatgpt': 'gpt',
        'bert': 'bert'
    }

    # Return the canonical skill name if found, otherwise the original
    return skill_mappings.get(skill.lower(), skill)

def is_non_tech_skill(skill):
    """
    Identify if a skill is clearly not related to technology
    Returns True if the skill is non-tech, False otherwise
    """
    # Make sure we're comparing lowercase strings
    skill_lower = skill.lower().strip()

    # List of skills from non-tech domains
    non_tech_skills = {
        # Medical
        'phlebotomy', 'medical', 'healthcare', 'clinical', 'patient care',
        'vital signs', 'medical terminology', 'cpr', 'first aid', 'haccp', 'gmp',
        'selenium', 'food safety',
        
        # Automotive
        'automotive repair', 'ase certifications', 'basic automotive tools',
        'vehicle maintenance', 'automotive diagnostic', 'brake repair',
        
        # Retail/Sales
        'sales', 'retail', 'cashiering', 'merchandising', 'inventory management',
        'point of sale', 'customer service', 'sales experience',
        
        # Transportation
        'reliable transportation', 'valid drivers license', 'cdl',
        'driving', 'transportation', 'delivery', 'logistics', 'drivers license', 'driver\'s license', 'driving license',
    }
    
    return skill_lower in non_tech_skills

def clean_skills(skills_text):
    """Clean and normalize a comma-separated skills string"""
    if not isinstance(skills_text, str) or not skills_text.strip():
        return []

    # Replace multiple instances of commas with a single comma and split
    cleaned_text = re.sub(r',+', ',', skills_text.strip())
    # Handle other common separators that might be used
    cleaned_text = re.sub(r'[;|]', ',', cleaned_text)

    # Split on commas
    raw_skills = [skill.strip().lower() for skill in cleaned_text.split(',')]

    # Filter out empty skills, normalize, and remove generic skills
    cleaned_skills = []
    for skill in raw_skills:
        if not skill:
            continue

        # Remove any potential leftover punctuation
        skill = re.sub(r'[^\w\s]', '', skill).strip()
        if not skill:
            continue

        # Skip very short terms (likely abbreviations without context)
        if len(skill) <= 2:
            continue

        # Normalize the skill
        normalized_skill = normalize_skill(skill)

        # Filter out generic skills
        if not is_generic_skill(normalized_skill) and not is_non_tech_skill(normalized_skill):
            cleaned_skills.append(normalized_skill)

    # Remove duplicates while preserving order
    unique_skills = []
    for skill in cleaned_skills:
        if skill not in unique_skills and skill.strip():  # Extra check for empty strings
            unique_skills.append(skill)

    return unique_skills

def post_process_skills(df, skills_column='skills'):
    """
    Perform a final cleaning pass on the skills column to remove any remaining
    generic skills or empty entries that might have slipped through
    """
    def clean_skill_list(skill_list):
        if not isinstance(skill_list, list):
            return []

        # Apply the is_generic_skill filter again
        return [skill for skill in skill_list if skill and not is_generic_skill(skill)]

    # Apply the cleaning function to the skills column
    df[skills_column] = df[skills_column].apply(clean_skill_list)

    # Remove rows with no skills after cleaning
    filtered_df = df[df[skills_column].apply(lambda x: len(x) > 0)]

    print(f"Removed {len(df) - len(filtered_df)} rows with empty skills after post-processing")
    return filtered_df

def validate_skills_cleaning(df, skills_column='skills'):
    """
    Validate the skills cleaning process by checking for any remaining generic skills
    or problematic entries in the dataset
    """
    print("Validating skills cleaning...")

    # Check for empty skill lists
    empty_skills = df[df[skills_column].apply(lambda x: not x or len(x) == 0)]
    print(f"Rows with empty skill lists: {len(empty_skills)}")

    # Flatten all skills to check for generic ones
    all_skills = []
    for skill_list in df[skills_column]:
        if isinstance(skill_list, list):
            all_skills.extend(skill_list)

    # Count all skills
    skill_counts = Counter(all_skills)

    # Check for potential remaining generic skills
    potential_generic = []
    for skill, count in skill_counts.most_common(50):  # Check top 50 most common skills
        if is_generic_skill(skill):
            potential_generic.append((skill, count))

    if potential_generic:
        print("\nPotential generic skills still in the dataset:")
        for skill, count in potential_generic:
            print(f"  - '{skill}': {count} occurrences")
    else:
        print("No common generic skills found in the dataset.")

    # Check for suspiciously short skills (might be abbreviations without context)
    short_skills = []
    for skill, count in skill_counts.items():
        if len(skill) <= 3 and count > 10:  # Short and appears multiple times
            short_skills.append((skill, count))

    if short_skills:
        print("\nPotentially problematic short skills:")
        for skill, count in sorted(short_skills, key=lambda x: x[1], reverse=True):
            print(f"  - '{skill}': {count} occurrences")

    # Check for punctuation or strange characters
    punctuation_skills = []
    for skill, count in skill_counts.items():
        if any(char in skill for char in ',;.:-()[]{}'):
            punctuation_skills.append((skill, count))

    if punctuation_skills:
        print("\nSkills with punctuation:")
        for skill, count in punctuation_skills:
            print(f"  - '{skill}': {count} occurrences")

    return potential_generic, short_skills, punctuation_skills

def sample_equal_roles(df, role_column='role_category', sample_size=None):
    """Sample an equal number of rows for each role category"""
    # Count occurrences of each role
    role_counts = df[role_column].value_counts()

    # Determine sample size (minimum count or specified value)
    if sample_size is None:
        sample_size = min(role_counts)
    else:
        # Ensure sample size doesn't exceed the minimum available
        sample_size = min(sample_size, min(role_counts))

    print(f"Sampling {sample_size} jobs per role category")

    # Sample equal number from each role
    sampled_dfs = []
    for role in role_counts.index:
        role_df = df[df[role_column] == role]
        if len(role_df) > sample_size:
            sampled_dfs.append(role_df.sample(sample_size, random_state=42))
        else:
            sampled_dfs.append(role_df)

    # Combine all samples
    return pd.concat(sampled_dfs, ignore_index=True)

def process_data(jobs_file, skills_file, output_file, sample_per_role=100):
    """Process job postings and skills to create a cleaned tech jobs dataset"""
    start_time = time.time()

    print(f"Reading job postings from: {jobs_file}")
    jobs_df = pd.read_csv(jobs_file)
    print(f"Job file read complete: {len(jobs_df)} job postings")

    print(f"Reading skills data from: {skills_file}")
    skills_df = pd.read_csv(skills_file)
    print(f"Skills file read complete: {len(skills_df)} skill entries")

    # Get column names
    job_id_col = 'job_link'  # Based on your logs, this is the common ID column
    job_title_col = 'job_title'
    skills_col = 'job_skills'

    print("Identifying tech roles (this might take a minute)...")
    jobs_df['tech_role'] = jobs_df[job_title_col].apply(identify_tech_roles)

    # Filter to tech jobs only
    tech_jobs_df = jobs_df[jobs_df['tech_role'].notna()].copy()
    print(f"Found {len(tech_jobs_df)} tech jobs from {len(jobs_df)} total job postings")

    # Show distribution of tech roles
    role_counts = tech_jobs_df['tech_role'].value_counts()
    print("\nDistribution of tech roles:")
    print("===========================")
    for role, count in role_counts.items():
        print(f"{role}: {count} ({count/len(tech_jobs_df)*100:.1f}%)")

    # Get tech job IDs
    tech_job_ids = set(tech_jobs_df[job_id_col])
    print(f"Filtering skills to {len(tech_job_ids)} tech job IDs...")

    # Filter skills to only include tech jobs (more efficient than join)
    tech_skills_df = skills_df[skills_df[job_id_col].isin(tech_job_ids)].copy()
    print(f"Found {len(tech_skills_df)} skill entries for tech jobs")

    # Clean skills column (only for tech jobs)
    print("Cleaning and normalizing skills...")
    tech_skills_df['clean_skills'] = tech_skills_df[skills_col].apply(clean_skills)

    # Verify if a job has skills
    print("Identifying jobs with non-empty skills...")
    # Get job IDs that have at least one skill
    jobs_with_skills = tech_skills_df[tech_skills_df['clean_skills'].apply(lambda x: len(x) > 0)][job_id_col].unique()
    print(f"Found {len(jobs_with_skills)} tech jobs that have at least one skill")

    # Filter to only jobs with skills
    valid_tech_jobs_df = tech_jobs_df[tech_jobs_df[job_id_col].isin(jobs_with_skills)].copy()
    print(f"Filtered to {len(valid_tech_jobs_df)} tech jobs with non-empty skills")

    # Prepare for join - optimize by using only needed columns
    tech_jobs_slim = valid_tech_jobs_df[[job_id_col, job_title_col, 'tech_role']].copy()
    tech_skills_slim = tech_skills_df[[job_id_col, 'clean_skills']].copy()

    # For performance, create dictionaries for skills lookup
    print("Creating skills lookup for faster processing...")
    skills_dict = {}
    for _, row in tech_skills_slim.iterrows():
        job_id = row[job_id_col]
        skills = row['clean_skills']
        if job_id not in skills_dict:
            skills_dict[job_id] = skills
        else:
            # Merge skills lists for jobs with multiple skill entries
            skills_dict[job_id].extend(skills)
            # Remove duplicates
            skills_dict[job_id] = list(dict.fromkeys(skills_dict[job_id]))

    # Add skills to jobs dataframe without a join operation
    print("Merging skills with jobs data...")
    tech_jobs_slim['skills'] = tech_jobs_slim[job_id_col].map(skills_dict)

    # Verify no empty skills after mapping
    empty_skills_count = tech_jobs_slim['skills'].isna().sum()
    print(f"Jobs with missing skills after mapping: {empty_skills_count}")

    # Remove any jobs that still have empty skills
    if empty_skills_count > 0:
        tech_jobs_slim = tech_jobs_slim[tech_jobs_slim['skills'].notna()].copy()
        print(f"Removed {empty_skills_count} jobs with missing skills")

    # Apply post-processing to catch any remaining generic skills
    print("Performing final skills cleaning...")
    tech_jobs_slim = post_process_skills(tech_jobs_slim, skills_column='skills')

    # Sample equal number from each role category
    if sample_per_role is not None:
        print("Sampling equal roles...")
        final_df = sample_equal_roles(tech_jobs_slim, role_column='tech_role', sample_size=sample_per_role)
    else:
        final_df = tech_jobs_slim

    # Show distribution after sampling
    role_counts_after = final_df['tech_role'].value_counts()
    print("\nDistribution of tech roles after sampling:")
    print("=========================================")
    for role, count in role_counts_after.items():
        target_met = "✓" if sample_per_role is None or count == sample_per_role else "✗"
        print(f"{role}: {count} ({count/len(final_df)*100:.1f}%) {target_met}")

    # Rename columns for output
    result_df = pd.DataFrame({
        'job_id': final_df[job_id_col],
        'title': final_df[job_title_col],
        'role_category': final_df['tech_role'],
        'skills': final_df['skills']
    })

    # Save the result
    result_df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")

    # Calculate top skills for each role
    print("\nTop skills by role category:")
    for role in sorted(result_df['role_category'].unique()):
        role_skills = []
        for skills_list in result_df[result_df['role_category'] == role]['skills']:
            if isinstance(skills_list, list):
                role_skills.extend(skills_list)

        # Count skill occurrences
        skill_counts = Counter(role_skills)

        # Print top skills for this role
        if skill_counts:
            top_skills = skill_counts.most_common(10)
            print(f"\n{role}:")
            for skill, count in top_skills:
                print(f"  - {skill}: {count} occurrences")

    # Validate the cleaning process
    print("\nValidating final dataset...")
    potential_generic, short_skills, punctuation_skills = validate_skills_cleaning(result_df)

    # Report validation results
    if not potential_generic and not short_skills and not punctuation_skills:
        print("Validation passed: No significant issues found in skills data.")
    else:
        print("Validation identified some potential issues. Review the output above.")

    # Report processing time
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

    return result_df

def main():
    """Main function to download and process job data"""
    print("Downloading LinkedIn jobs and skills dataset...")
    try:
        # Download dataset using kagglehub
        data_path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")
        print(f"Dataset downloaded to: {data_path}")

        # Specify the exact files we need
        job_postings_file = os.path.join(data_path, "linkedin_job_postings.csv")
        skills_file = os.path.join(data_path, "job_skills.csv")

        # Verify the files exist
        if not os.path.exists(job_postings_file):
            print(f"Error: Job postings file not found at {job_postings_file}")
            files = os.listdir(data_path)
            print(f"Available files in directory: {files}")
            raise FileNotFoundError(f"Missing file: linkedin_job_postings.csv")

        if not os.path.exists(skills_file):
            print(f"Error: Skills file not found at {skills_file}")
            files = os.listdir(data_path)
            print(f"Available files in directory: {files}")
            raise FileNotFoundError(f"Missing file: job_skills.csv")

        print(f"Job postings file: {job_postings_file}")
        print(f"Job skills file: {skills_file}")

        # Output file path
        output_file = os.path.join(data_path, "tech_jobs_with_skills.csv")

        # Process the data with 100 samples per role
        processed_data = process_data(
            jobs_file=job_postings_file,
            skills_file=skills_file,
            output_file=output_file,
            sample_per_role=100
        )

        # Display sample of processed data
        if processed_data is not None:
            print("\nSample of processed data:")
            print(processed_data.head())

            print(f"\nFull dataset saved to: {output_file}")
            print(f"Total processed records: {len(processed_data)}")

    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()