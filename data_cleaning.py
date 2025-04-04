import pandas as pd
import re
import os
from tqdm import tqdm


def identify_tech_roles(job_title):
    """
    Identify if a job title belongs to a specific tech role
    Returns the tech category if found, None otherwise
    """
    # Convert to lowercase for case-insensitive matching
    title = str(job_title).lower()

    # Define tech role patterns with clear boundaries to prevent partial matches
    tech_patterns = {
        'frontend_developer': r'\b(front.?end|ui developer|react developer|angular developer|vue developer)\b',
        'backend_developer': r'\b(back.?end|api developer|django developer|flask developer|node developer|java developer|python developer)\b',
        'fullstack_developer': r'\b(full.?stack|mean stack|mern stack|lamp stack|web developer)\b',
        'data_scientist': r'\b(data scien|machine learning|ml engineer|ai researcher|ml scientist)\b',
        'data_engineer': r'\b(data engineer|etl developer|data pipeline|big data)\b',
        'devops_engineer': r'\b(devops|sre|site reliability|infrastructure|platform engineer|cloud engineer)\b',
        'mobile_developer': r'\b(mobile|ios|android|flutter|react native|swift developer)\b',
        'security_engineer': r'\b(security|cyber|penetration tester|ethical hacker|infosec)\b',
        'qa_engineer': r'\b(qa engineer|quality assurance|test engineer|automation test|sdet)\b',
        'software_engineer': r'\b(software engineer|programmer|coder|developer)\b',
        'cloud_architect': r'\b(cloud architect|aws architect|azure architect|gcp architect)\b',
        'data_analyst': r'\b(data analyst|business intelligence|bi developer|analytics)\b',
        'product_manager': r'\b(product manager|technical product manager|product owner)\b',
        'technical_support': r'\b(technical support|it support|help desk|system admin)\b'
    }

    # Check each pattern
    for role, pattern in tech_patterns.items():
        if re.search(pattern, title):
            return role

    # No specific tech role found - return None instead of 'other_tech'
    return None


def clean_skills(skills_text):
    """Clean and normalize a comma-separated skills string"""
    if not isinstance(skills_text, str):
        return []

    # Split by comma and clean each skill
    skills = [skill.strip().lower() for skill in skills_text.split(',')]

    # Filter out empty skills and normalize
    cleaned_skills = []
    for skill in skills:
        if not skill:
            continue

        # Basic normalization for common tech skills
        skill_mappings = {
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'react.js': 'react',
            'reactjs': 'react',
            'vue.js': 'vue',
            'vuejs': 'vue',
            'angular.js': 'angular',
            'angularjs': 'angular',
            'node.js': 'node',
            'nodejs': 'node',
            'k8s': 'kubernetes',
            'aws cloud': 'aws',
            'msft azure': 'azure',
            'microsoft azure': 'azure',
            'postgre': 'postgresql',
            'postgres': 'postgresql',
            'mongo': 'mongodb'
        }

        # Apply mappings
        for old, new in skill_mappings.items():
            if skill == old:
                skill = new
                break

        cleaned_skills.append(skill)

    # Remove duplicates while preserving order
    unique_skills = []
    for skill in cleaned_skills:
        if skill not in unique_skills:
            unique_skills.append(skill)

    return unique_skills


def find_csv_file(directory, keywords, fallback_to_current_dir=True):
    """Find a CSV file in a directory containing all the given keywords"""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith('.csv') and all(keyword.lower() in file.lower() for keyword in keywords):
                return os.path.join(directory, file)

    # Fallback to current directory
    if fallback_to_current_dir and directory != '.':
        for file in os.listdir('.'):
            if file.endswith('.csv') and all(keyword.lower() in file.lower() for keyword in keywords):
                return file

    return None


def merge_and_clean_tech_jobs(jobs_file, skills_file, output_file, batch_size=10000, debug=False):
    """
    Process job postings and skills to create a cleaned tech jobs dataset

    Parameters:
    -----------
    jobs_file : str
        Path to the job postings CSV file
    skills_file : str
        Path to the job skills CSV file
    output_file : str
        Path for the output CSV file
    batch_size : int
        Size of batches for processing
    debug : bool
        Enable debug output
    """
    # Set debug mode
    if debug:
        print("DEBUG MODE ENABLED")
        print(f"Job postings file: {jobs_file}")
        print(f"Skills file: {skills_file}")

    print(f"Loading job postings from: {jobs_file}")

    # Count total rows for progress reporting
    total_rows = sum(1 for _ in open(jobs_file, 'r')) - 1  # Subtract header

    # Try different encodings and delimiters for CSV files
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    delimiters = [',', ';', '\t']

    # Try to find the right encoding and delimiter for jobs file
    read_successful = False
    for encoding in encodings:
        if read_successful:
            break
        for delimiter in delimiters:
            try:
                # Try reading a small sample
                sample_df = pd.read_csv(jobs_file, nrows=5, encoding=encoding, sep=delimiter)
                if debug:
                    print(f"Successfully read job file with encoding={encoding}, delimiter='{delimiter}'")
                    print(f"Sample columns: {sample_df.columns.tolist()}")
                read_successful = True
                break
            except Exception as e:
                if debug:
                    print(f"Failed reading with encoding={encoding}, delimiter='{delimiter}': {str(e)}")
                continue

    if not read_successful:
        print("Error: Could not read job postings file with any encoding/delimiter combination")
        return

    # Read job postings in chunks with the detected encoding and delimiter
    tech_jobs = []
    for chunk in tqdm(pd.read_csv(jobs_file, chunksize=batch_size, encoding=encoding, sep=delimiter),
                      total=total_rows // batch_size + 1,
                      desc="Processing job postings"):

        # Extract relevant columns
        required_cols = ['job_title']
        id_cols = ['job_id', 'job_link', 'id', 'link']

        # First check if any of our possible ID columns exist
        found_id_col = None
        for id_col in id_cols:
            if id_col in chunk.columns:
                found_id_col = id_col
                break

        if found_id_col is None:
            print(f"Error: Could not find any job ID column. Looking for any of: {id_cols}")
            print(f"Available columns: {chunk.columns.tolist()}")
            return None

        # Check for title column
        missing_cols = [col for col in required_cols if col not in chunk.columns]

        if missing_cols:
            print(f"Error: Missing required columns in job postings file: {missing_cols}")
            print(f"Available columns: {chunk.columns.tolist()}")

            # Try to find alternative title column name
            if 'job_title' in missing_cols and any(col for col in chunk.columns if 'title' in col.lower()):
                alt_title_col = next(col for col in chunk.columns if 'title' in col.lower())
                print(f"Using '{alt_title_col}' instead of 'job_title'")
                chunk['job_title'] = chunk[alt_title_col]

            # Check again if we still have missing columns
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                print(f"Error: Still missing required columns: {missing_cols}")
                if debug:
                    print("First few rows of the data:")
                    print(chunk.head())
                return None

        # Create a new DataFrame with the columns we need
        subset = pd.DataFrame()
        subset['job_id'] = chunk[found_id_col]  # Use the found ID column but map to job_id
        subset['job_title'] = chunk['job_title']

        # Identify tech roles
        subset['tech_role'] = subset['job_title'].apply(identify_tech_roles)

        # Filter tech roles
        tech_subset = subset[subset['tech_role'].notna()].copy()
        tech_jobs.append(tech_subset)

    # Combine all tech jobs
    if not tech_jobs:
        print("No specific tech jobs found!")
        return

    tech_jobs_df = pd.concat(tech_jobs, ignore_index=True)
    print(f"Found {len(tech_jobs_df)} specifically categorized tech jobs")

    # Show distribution of tech roles
    role_counts = tech_jobs_df['tech_role'].value_counts()
    print("\nDistribution of tech roles:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} jobs ({count / len(tech_jobs_df) * 100:.1f}%)")

    # Read skills data
    print(f"Loading skills data from: {skills_file}")

    # Find the right encoding and delimiter for skills file
    skills_read_successful = False
    for encoding in encodings:
        if skills_read_successful:
            break
        for delimiter in delimiters:
            try:
                # Try reading a small sample
                skills_sample = pd.read_csv(skills_file, nrows=5, encoding=encoding, sep=delimiter)
                if debug:
                    print(f"Successfully read skills file with encoding={encoding}, delimiter='{delimiter}'")
                    print(f"Sample columns: {skills_sample.columns.tolist()}")
                skills_read_successful = True
                break
            except Exception as e:
                if debug:
                    print(f"Failed reading skills with encoding={encoding}, delimiter='{delimiter}': {str(e)}")
                continue

    if not skills_read_successful:
        print("Error: Could not read skills file with any encoding/delimiter combination")
        return

    # Determine if job_id is in the skills file
    job_id_cols = ['job_id', 'job_link', 'id', 'link']
    job_id_col = None

    for col in job_id_cols:
        if col in skills_sample.columns:
            job_id_col = col
            if debug:
                print(f"Found job identifier column in skills file: '{job_id_col}'")
            break

    if job_id_col is None:
        print(
            f"Error: Could not find job ID column in skills file. Available columns: {skills_sample.columns.tolist()}")
        print(f"Looking for any of these columns: {job_id_cols}")
        return

    # Read skills in chunks and merge with tech jobs
    skills_data = []

    # Find skills column
    skills_cols = ['job_skills', 'skills', 'skill']
    skills_col = None

    for col in skills_cols:
        if col in skills_sample.columns:
            skills_col = col
            if debug:
                print(f"Found skills column in skills file: '{skills_col}'")
            break

    if skills_col is None:
        print(
            f"Error: Could not find skills column in skills file. Available columns: {skills_sample.columns.tolist()}")
        print(f"Looking for any of these columns: {skills_cols}")
        return

    try:
        for chunk in tqdm(pd.read_csv(skills_file, chunksize=batch_size, encoding=encoding, sep=delimiter),
                          desc="Processing skills data"):

            # Filter skills for tech jobs
            if skills_col in chunk.columns and job_id_col in chunk.columns:
                tech_skills = chunk[chunk[job_id_col].isin(tech_jobs_df['job_id'])]
                skills_data.append(tech_skills[[job_id_col, skills_col]])
    except Exception as e:
        print(f"Error reading skills data: {str(e)}")
        return

    # Combine all skills data
    if not skills_data:
        print("No skills data found for tech jobs!")
        return

    skills_df = pd.concat(skills_data, ignore_index=True)
    print(f"Found skills data for {len(skills_df)} tech jobs")

    # Clean skills
    skills_df['clean_skills'] = skills_df[skills_col].apply(clean_skills)

    # Merge jobs and skills
    merged_df = pd.merge(
        tech_jobs_df,
        skills_df,
        left_on='job_id',
        right_on=job_id_col,
        how='inner'
    )

    # Print debug info if merge resulted in few rows
    if len(merged_df) < min(len(tech_jobs_df), len(skills_df)) * 0.5:
        print(f"Warning: Merge resulted in only {len(merged_df)} rows out of {len(tech_jobs_df)} tech jobs")
        print(f"Sample job_id values from tech_jobs_df: {tech_jobs_df['job_id'].head().tolist()}")
        print(f"Sample {job_id_col} values from skills_df: {skills_df[job_id_col].head().tolist()}")

    # Select and rename final columns
    try:
        result_df = merged_df[['job_id', 'job_title', 'tech_role', 'clean_skills']].copy()
        result_df.rename(columns={
            'job_title': 'title',
            'tech_role': 'role_category',
            'clean_skills': 'skills'
        }, inplace=True)
    except KeyError as e:
        print(f"Error selecting columns: {e}")
        print(f"Available columns in merged_df: {merged_df.columns.tolist()}")

        # Create a new DataFrame with the essential columns
        result_df = pd.DataFrame({
            'job_id': merged_df['job_id'],
            'title': merged_df['job_title'],
            'role_category': merged_df['tech_role'],
            'skills': merged_df['clean_skills']
        })

    # Save the result
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(result_df)} specific tech jobs with skills to {output_file}")

    # Show top skills for each role category
    print("\nTop skills by role category:")
    for role in sorted(result_df['role_category'].unique()):
        role_skills = []
        for skills_list in result_df[result_df['role_category'] == role]['skills']:
            try:
                # Skills might be stored as string representation of list
                if isinstance(skills_list, str):
                    if skills_list.startswith('[') and skills_list.endswith(']'):
                        # Try to evaluate as a list
                        import ast
                        try:
                            skills = ast.literal_eval(skills_list)
                            role_skills.extend(skills)
                        except:
                            # If parsing fails, just split by comma
                            skills = [s.strip().strip("'\"") for s in skills_list.strip("[]").split(',')]
                            role_skills.extend(skills)
                    else:
                        # Just split by comma
                        skills = [s.strip() for s in skills_list.split(',')]
                        role_skills.extend(skills)
                elif isinstance(skills_list, list):
                    role_skills.extend(skills_list)
            except:
                continue

        # Count skill occurrences
        from collections import Counter
        skill_counts = Counter(role_skills)

        # Print top 5 skills for this role
        if skill_counts:
            top_skills = skill_counts.most_common(5)
            print(f"  {role}:")
            for skill, count in top_skills:
                print(f"    - {skill}: {count} occurrences")

    # Return a sample of the data
    return result_df.head(10)


def main():
    """Main function to process job data"""
    # Check for Kaggle dataset path
    path = "/root/.cache/kagglehub/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/versions/2"

    # Alternative paths to check
    alternative_paths = [
        "./data",
        "./dataset",
        "."
    ]

    try:
        alternative_paths.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass

    # Find the existing path
    if not os.path.exists(path):
        print(f"Dataset path not found: {path}")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                path = alt_path
                print(f"Using alternative path: {path}")
                break

    # Look for data files using smarter file detection
    job_postings_file = find_csv_file(path, ["job", "posting"])
    if not job_postings_file:
        job_postings_file = find_csv_file(path, ["linkedin"])

    skills_file = find_csv_file(path, ["job", "skill"])
    if not skills_file:
        skills_file = find_csv_file(path, ["skill"])

    # Check if files exist
    if not job_postings_file:
        print("Error: Could not find job postings CSV file!")
        print("Please specify the path to the job postings file:")
        job_postings_file = input().strip()
        if not os.path.exists(job_postings_file):
            print(f"File not found: {job_postings_file}")
            return

    if not skills_file:
        print("Error: Could not find job skills CSV file!")
        print("Please specify the path to the job skills file:")
        skills_file = input().strip()
        if not os.path.exists(skills_file):
            print(f"File not found: {skills_file}")
            return

    # Output file path
    output_file = os.path.join(path, "tech_jobs_with_skills.csv")

    # Enable debug mode for troubleshooting
    debug_mode = True  # Set to True for verbose output

    # Process the data
    sample_data = merge_and_clean_tech_jobs(
        jobs_file=job_postings_file,
        skills_file=skills_file,
        output_file=output_file,
        debug=debug_mode
    )

    # Display sample data
    if sample_data is not None:
        print("\nSample of processed data:")
        print(sample_data)


if __name__ == "__main__":
    main()