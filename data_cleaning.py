import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model (you'll need to install this with: python -m spacy download en_core_web_md)
nlp = spacy.load('en_core_web_md')


def clean_job_titles_with_nlp(df, title_column='job_title'):
    """
    Clean and standardize job titles using NLP techniques

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing job postings
    title_column : str
        Column name containing job titles

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cleaned and categorized job titles
    """
    print(f"Cleaning {len(df)} job titles...")

    # Create a copy of the dataframe
    df_clean = df.copy()

    # Basic cleaning
    df_clean['title_clean'] = df_clean[title_column].str.lower()
    df_clean['title_clean'] = df_clean['title_clean'].str.replace(r'[^\w\s]', ' ', regex=True)  # Remove special chars
    df_clean['title_clean'] = df_clean['title_clean'].str.replace(r'\s+', ' ',
                                                                  regex=True).str.strip()  # Normalize spaces

    # Remove common words that don't add meaning to job roles
    common_words = ['senior', 'junior', 'lead', 'staff', 'principal', 'associate',
                    'intern', 'contractor', 'freelance', 'i', 'ii', 'iii', 'iv', 'v',
                    'level', 'position', 'job', 'role', 'remote', 'hybrid', 'onsite']

    pattern = r'\b(' + '|'.join(common_words) + r')\b'
    df_clean['title_role'] = df_clean['title_clean'].str.replace(pattern, '', regex=True)
    df_clean['title_role'] = df_clean['title_role'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Define common tech role patterns
    tech_role_patterns = {
        'frontend_developer': [r'front.?end', r'ui developer', r'react developer', r'angular developer',
                               r'vue developer'],
        'backend_developer': [r'back.?end', r'api developer', r'django developer', r'flask developer',
                              r'node developer'],
        'fullstack_developer': [r'full.?stack', r'mean stack', r'mern stack', r'lamp stack', r'web developer'],
        'data_scientist': [r'data scien', r'machine learning', r'ml engineer', r'ai researcher', r'ml scientist'],
        'data_engineer': [r'data engineer', r'etl developer', r'data pipeline', r'big data'],
        'devops_engineer': [r'devops', r'sre', r'site reliability', r'infrastructure', r'platform', r'cicd', r'cloud'],
        'mobile_developer': [r'mobile', r'ios', r'android', r'flutter', r'react native', r'swift'],
        'security_engineer': [r'security', r'cyber', r'penetration tester', r'ethical hacker', r'infosec'],
        'qa_engineer': [r'qa engineer', r'quality assurance', r'test engineer', r'automation test', r'sdet']
    }

    # Categorize job titles based on patterns
    for role, patterns in tech_role_patterns.items():
        pattern = '|'.join(patterns)
        df_clean[f'is_{role}'] = df_clean['title_role'].str.contains(pattern, case=False, regex=True).astype(int)

    # Determine primary role category
    role_columns = [f'is_{role}' for role in tech_role_patterns.keys()]

    # Default is 'other'
    df_clean['primary_role'] = 'other'

    # If exactly one role category matches, use that
    for idx, row in df_clean[role_columns].iterrows():
        if sum(row) == 1:
            # Find the matching role
            for i, val in enumerate(row):
                if val == 1:
                    role_name = tech_role_patterns.keys()[i]
                    df_clean.at[idx, 'primary_role'] = role_name
                    break

    # For titles with multiple or no matches, use NLP similarity to find closest category
    ambiguous_indices = df_clean.index[(df_clean[role_columns].sum(axis=1) != 1)]
    print(f"Found {len(ambiguous_indices)} ambiguous job titles requiring NLP analysis")

    # Create reference role descriptions
    role_descriptions = {
        'frontend_developer': 'frontend web development HTML CSS JavaScript React Angular Vue UI UX',
        'backend_developer': 'backend server API database SQL NoSQL Django Flask Node Express Java Python',
        'fullstack_developer': 'fullstack frontend backend web development JavaScript Node React Angular',
        'data_scientist': 'data science machine learning ML AI statistics modeling Python R algorithms',
        'data_engineer': 'data engineering ETL pipeline Hadoop Spark SQL database warehouse',
        'devops_engineer': 'DevOps CI CD cloud AWS Azure infrastructure Kubernetes Docker containers',
        'mobile_developer': 'mobile iOS Android Swift Kotlin React Native Flutter app development',
        'security_engineer': 'security cybersecurity encryption penetration testing firewall protection',
        'qa_engineer': 'quality assurance testing automation test cases Selenium test-driven development'
    }

    # Process ambiguous titles with NLP
    for idx in ambiguous_indices:
        title = df_clean.at[idx, 'title_role']

        # Create vectors for similarity comparison
        title_vec = nlp(title).vector.reshape(1, -1)

        max_sim = -1
        best_role = 'other'

        # Compare with each role description
        for role, desc in role_descriptions.items():
            desc_vec = nlp(desc).vector.reshape(1, -1)
            sim = cosine_similarity(title_vec, desc_vec)[0][0]

            if sim > max_sim and sim > 0.3:  # Threshold for similarity
                max_sim = sim
                best_role = role

        df_clean.at[idx, 'primary_role'] = best_role

    return df_clean


def normalize_skills_with_nlp(skills_list):
    """
    Normalize skills using NLP techniques including:
    - Entity recognition
    - Skill name standardization
    - Removing duplicates and similar terms

    Parameters:
    -----------
    skills_list : list of strings
        List of skills to normalize

    Returns:
    --------
    list
        Normalized list of skills
    """
    if not skills_list or not isinstance(skills_list, list):
        return []

    # Convert all to lowercase and strip whitespace
    clean_skills = [skill.lower().strip() for skill in skills_list if isinstance(skill, str)]

    # Remove special characters except hyphens
    clean_skills = [re.sub(r'[^\w\s-]', '', skill) for skill in clean_skills]

    # Standardize common variations
    skill_mapping = {
        # Programming languages
        r'\bjs\b': 'javascript',
        r'\bnodejs\b': 'node.js',
        r'\bts\b': 'typescript',
        r'\bpy\b': 'python',
        r'golang': 'go',
        r'objective-?c': 'objective-c',

        # Frameworks and libraries
        r'react\.?js': 'react',
        r'angular(?:js|2\+)': 'angular',
        r'vue\.?js': 'vue',
        r'tensorflow': 'tensorflow',
        r'pytorch': 'pytorch',

        # Technologies and tools
        r'k8s': 'kubernetes',
        r'docker-?compose': 'docker compose',
        r'github actions': 'ci/cd',
        r'gitlab ci': 'ci/cd',
        r'jenkins': 'ci/cd',
        r'aws cloud': 'aws',
        r'amazon web services': 'aws',
        r'google cloud platform': 'gcp',
        r'msft azure': 'azure',
        r'microsoft azure': 'azure',

        # Databases
        r'postgres(?:ql)?': 'postgresql',
        r'mongo(?:db)?': 'mongodb',
        r'mysql': 'mysql',
        r'ms sql server': 'sql server',
        r'oracle db': 'oracle',

        # Concepts
        r'object(?:-|\s)?oriented': 'oop',
        r'machine learning': 'machine learning',
        r'deep learning': 'deep learning',
        r'[\w\s]*microservices[\w\s]*': 'microservices',
        r'[\w\s]*agile[\w\s]*': 'agile',
        r'[\w\s]*scrum[\w\s]*': 'scrum'
    }

    for pattern, replacement in skill_mapping.items():
        clean_skills = [re.sub(pattern, replacement, skill, flags=re.IGNORECASE) for skill in clean_skills]

    # Process skills through spaCy to identify entities and normalize forms
    processed_skills = []
    for skill in clean_skills:
        doc = nlp(skill)

        # If skill is recognized as a product, organization, or language, keep it as is
        if len(doc.ents) > 0 and doc.ents[0].label_ in ['PRODUCT', 'ORG', 'LANGUAGE']:
            processed_skills.append(skill)
        # Otherwise use the lemmatized form for general terms
        else:
            lemma = ' '.join([token.lemma_ for token in doc if not token.is_stop])
            if lemma:
                processed_skills.append(lemma)

    # Remove duplicates while preserving order
    unique_skills = []
    for skill in processed_skills:
        if skill and len(skill) > 1 and skill not in unique_skills:
            unique_skills.append(skill)

    return unique_skills


def identify_related_skills(skills_data, threshold=0.5):
    """
    Use NLP to identify semantically related skills

    Parameters:
    -----------
    skills_data : list of lists
        Lists of skills from job postings
    threshold : float
        Similarity threshold for considering skills related

    Returns:
    --------
    dict
        Dictionary of related skill clusters
    """
    # Flatten the list of skills
    all_skills = [skill for sublist in skills_data for skill in sublist if isinstance(skill, str)]

    # Count skill occurrences
    skill_counter = Counter(all_skills)

    # Keep only skills that appear more than once
    common_skills = [skill for skill, count in skill_counter.items() if count > 1]

    # Create skill vectors
    skill_vectors = {}
    for skill in common_skills:
        skill_vectors[skill] = nlp(skill).vector

    # Find related skills
    related_skills = {}
    for skill1 in common_skills:
        vec1 = skill_vectors[skill1]

        for skill2 in common_skills:
            if skill1 == skill2:
                continue

            vec2 = skill_vectors[skill2]
            similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

            if similarity >= threshold:
                if skill1 not in related_skills:
                    related_skills[skill1] = []
                related_skills[skill1].append((skill2, similarity))

    # Sort related skills by similarity
    for skill, related in related_skills.items():
        related_skills[skill] = sorted(related, key=lambda x: x[1], reverse=True)

    return related_skills


def extract_key_skills_by_role(integrated_data, min_freq=5):
    """
    Extract key skills for each role using NLP to identify important terms

    Parameters:
    -----------
    integrated_data : pandas.DataFrame
        Integrated data with job titles and skills
    min_freq : int
        Minimum frequency for a skill to be considered

    Returns:
    --------
    dict
        Dictionary mapping roles to their key skills
    """
    role_skills = {}

    # Group data by primary role
    role_groups = integrated_data.groupby('primary_role')

    for role, group in role_groups:
        # Skip 'other' category
        if role == 'other':
            continue

        # Collect all skills for this role
        all_role_skills = []
        for skills in group['skills_list']:
            if isinstance(skills, list):
                all_role_skills.extend(skills)

        # Count skill frequencies
        skill_counts = Counter(all_role_skills)

        # Keep skills above minimum frequency
        key_skills = {skill: count for skill, count in skill_counts.items()
                      if count >= min_freq}

        if key_skills:
            # Calculate TF-IDF to find distinctive skills
            role_skills[role] = {
                'skill_counts': key_skills,
                'distinctive_skills': []  # Will fill this later
            }

    # Create a document for each role - the concatenation of all its skills
    role_documents = {}
    for role in role_skills:
        role_documents[role] = ' '.join(list(role_skills[role]['skill_counts'].keys()))

    # Calculate TF-IDF to identify distinctive skills
    if len(role_documents) > 1:  # Need at least 2 roles for comparison
        roles = list(role_documents.keys())
        documents = [role_documents[role] for role in roles]

        # Create TF-IDF matrix
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # For each role, find skills that are common in this role but uncommon in others
        for i, role in enumerate(roles):
            # Sum counts for all other roles
            other_roles_count = X.toarray().sum(axis=0) - X[i].toarray()[0]

            # Calculate distinctiveness score: (this_role_count / other_roles_count)
            distinctiveness = []
            for j, feature in enumerate(feature_names):
                if feature in role_skills[role]['skill_counts']:
                    this_count = role_skills[role]['skill_counts'][feature]
                    other_count = other_roles_count[j]

                    # Avoid division by zero
                    if other_count == 0:
                        score = this_count * 10  # High score if unique to this role
                    else:
                        score = this_count / (other_count + 1)  # +1 smoothing

                    distinctiveness.append((feature, score))

            # Sort by distinctiveness score
            distinctiveness.sort(key=lambda x: x[1], reverse=True)
            role_skills[role]['distinctive_skills'] = [skill for skill, _ in distinctiveness[:20]]

    return role_skills


def identify_transition_skills(role_skills, role1, role2):
    """
    Identify skills that facilitate transitions between roles

    Parameters:
    -----------
    role_skills : dict
        Dictionary with role skills from extract_key_skills_by_role
    role1 : str
        Source role
    role2 : str
        Target role

    Returns:
    --------
    dict
        Dictionary with transition analysis
    """
    if role1 not in role_skills or role2 not in role_skills:
        return None

    # Get skills sets
    role1_skills = set(role_skills[role1]['skill_counts'].keys())
    role2_skills = set(role_skills[role2]['skill_counts'].keys())

    # Find common and unique skills
    common_skills = role1_skills.intersection(role2_skills)
    role1_unique = role1_skills - role2_skills
    role2_unique = role2_skills - role1_skills

    # Calculate skill importance
    role1_important = {skill: role_skills[role1]['skill_counts'][skill]
                       for skill in role1_unique if skill in role_skills[role1]['skill_counts']}
    role2_important = {skill: role_skills[role2]['skill_counts'][skill]
                       for skill in role2_unique if skill in role_skills[role2]['skill_counts']}

    # Sort by frequency
    role1_important = dict(sorted(role1_important.items(), key=lambda x: x[1], reverse=True))
    role2_important = dict(sorted(role2_important.items(), key=lambda x: x[1], reverse=True))

    # Find distinctive skills for the target role
    target_distinctive = role_skills[role2]['distinctive_skills']

    # Identify potential bridge skills (skills needed for role2 that are semantically similar to some in role1)
    bridge_skills = []

    # Check semantic similarity between skills
    for skill1 in list(role1_unique)[:20]:  # Top 20 for efficiency
        vec1 = nlp(skill1).vector

        for skill2 in list(role2_unique)[:20]:
            vec2 = nlp(skill2).vector
            similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

            if similarity >= 0.5:  # Threshold for similarity
                bridge_skills.append({
                    'source_skill': skill1,
                    'target_skill': skill2,
                    'similarity': similarity
                })

    transition_analysis = {
        'source_role': role1,
        'target_role': role2,
        'common_skills': list(common_skills),
        'skills_to_transfer': list(role1_important.keys())[:10],  # Top 10 skills from role1
        'skills_to_acquire': list(role2_important.keys())[:10],  # Top 10 skills needed for role2
        'distinctive_target_skills': target_distinctive[:10],  # Most distinctive skills for role2
        'potential_bridge_skills': bridge_skills
    }

    return transition_analysis


# Main function integrating NLP-based cleaning and analysis
def nlp_enhanced_analysis(job_postings_df, job_skills_df, focus_transitions=None):
    """
    Perform NLP-enhanced cleaning and analysis of job data

    Parameters:
    -----------
    job_postings_df : pandas.DataFrame
        DataFrame with job postings
    job_skills_df : pandas.DataFrame
        DataFrame with job skills
    focus_transitions : list of tuples
        List of role pairs to focus analysis on, e.g. [('backend_developer', 'devops_engineer')]

    Returns:
    --------
    dict
        Dictionary with cleaned data and analysis results
    """
    # Step 1: Clean job titles with NLP
    print("Cleaning job titles with NLP...")
    job_postings_clean = clean_job_titles_with_nlp(job_postings_df)

    # Step 2: Process and normalize skills
    print("Normalizing skills with NLP...")
    if 'job_skills' in job_skills_df.columns:
        # Process comma-separated skills
        job_skills_df['skills_list'] = job_skills_df['job_skills'].fillna('').apply(
            lambda x: normalize_skills_with_nlp(x.split(',')) if isinstance(x, str) else []
        )
    elif 'skills_list' in job_skills_df.columns:
        # Process list skills
        job_skills_df['skills_list'] = job_skills_df['skills_list'].apply(
            lambda x: normalize_skills_with_nlp(x) if isinstance(x, list) else []
        )

    # Step 3: Merge job postings with skills
    print("Integrating job postings with skills...")
    if 'job_link' in job_postings_clean.columns and 'job_link' in job_skills_df.columns:
        # Check that job_link is the key field
        integrated_data = job_postings_clean.merge(
            job_skills_df[['job_link', 'skills_list']],
            on='job_link',
            how='left'
        )

        # Handle missing skills
        integrated_data['skills_list'] = integrated_data['skills_list'].apply(
            lambda x: [] if isinstance(x, float) and np.isnan(x) else x
        )
    else:
        print("Warning: Could not integrate datasets. Missing job_link column.")
        integrated_data = job_postings_clean.copy()
        integrated_data['skills_list'] = [[]] * len(integrated_data)

    # Step 4: Extract key skills by role
    print("Extracting key skills by role...")
    role_skills = extract_key_skills_by_role(integrated_data)

    # Step 5: Analyze transitions between roles
    transition_analyses = {}

    if focus_transitions:
        print(f"Analyzing {len(focus_transitions)} focused role transitions...")
        for source_role, target_role in focus_transitions:
            print(f"Analyzing transition: {source_role} → {target_role}")
            analysis = identify_transition_skills(role_skills, source_role, target_role)
            if analysis:
                transition_analyses[(source_role, target_role)] = analysis
    else:
        # Analyze a few high-demand transitions
        high_demand_transitions = [
            ('backend_developer', 'devops_engineer'),
            ('frontend_developer', 'fullstack_developer'),
            ('data_scientist', 'data_engineer'),
            ('mobile_developer', 'fullstack_developer')
        ]

        print(f"Analyzing {len(high_demand_transitions)} high-demand role transitions...")
        for source_role, target_role in high_demand_transitions:
            if source_role in role_skills and target_role in role_skills:
                print(f"Analyzing transition: {source_role} → {target_role}")
                analysis = identify_transition_skills(role_skills, source_role, target_role)
                if analysis:
                    transition_analyses[(source_role, target_role)] = analysis

    return {
        'job_postings_clean': job_postings_clean,
        'job_skills_clean': job_skills_df,
        'integrated_data': integrated_data,
        'role_skills': role_skills,
        'transition_analyses': transition_analyses
    }