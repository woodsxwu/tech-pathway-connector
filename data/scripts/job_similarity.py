import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from itertools import combinations

def load_job_data(file_path):
    """Load job data"""
    df = pd.read_csv(file_path)
    # Convert skills string to list
    df['skills'] = df['skills'].apply(ast.literal_eval)
    return df

def calculate_job_similarity(job1_skills, job2_skills):
    """Calculate similarity between two jobs based on their skills"""
    # Convert skills list to string
    skills1 = ' '.join(job1_skills)
    skills2 = ' '.join(job2_skills)
    
    # Use TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([skills1, skills2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def find_similar_jobs(target_job_id, df, top_n=5):
    """查找与目标职位最相似的职位"""
    target_job = df[df['job_id'] == target_job_id].iloc[0]
    target_skills = target_job['skills']
    
    similarities = []
    for _, job in df.iterrows():
        if job['job_id'] != target_job_id:
            similarity = calculate_job_similarity(target_skills, job['skills'])
            similarities.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'similarity': similarity
            })
    
    # 按相似度排序
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_n]

def calculate_all_pairs_similarity(df, top_n=20):
    """计算所有职位对之间的相似度并排序"""
    job_pairs = []
    
    # 获取所有职位ID的组合
    job_ids = df['job_id'].tolist()
    for job1_id, job2_id in combinations(job_ids, 2):
        job1 = df[df['job_id'] == job1_id].iloc[0]
        job2 = df[df['job_id'] == job2_id].iloc[0]
        
        similarity = calculate_job_similarity(job1['skills'], job2['skills'])
        
        job_pairs.append({
            'job1_id': job1_id,
            'job1_title': job1['title'],
            'job2_id': job2_id,
            'job2_title': job2['title'],
            'similarity': similarity
        })
    
    # 按相似度排序
    job_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return job_pairs[:top_n]

def merge_category_skills(df):
    """Merge all skills for each role category"""
    category_skills = {}
    
    # Merge skills for each category
    for category in df['role_category'].unique():
        # Get all jobs in this category
        category_jobs = df[df['role_category'] == category]
        all_skills = []
        
        # Merge all skills lists
        for skills in category_jobs['skills']:
            all_skills.extend(skills)
        
        # Remove duplicates and sort
        category_skills[category] = sorted(list(set(all_skills)))
    
    return category_skills

def calculate_category_similarity(category_skills):
    """Calculate similarity between role categories"""
    category_pairs = []
    
    # Get all categories
    categories = list(category_skills.keys())
    
    # Calculate similarity for each category pair
    for cat1, cat2 in combinations(categories, 2):
        similarity = calculate_job_similarity(category_skills[cat1], category_skills[cat2])
        
        category_pairs.append({
            'category1': cat1,
            'category2': cat2,
            'similarity': similarity,
            'skills1_count': len(category_skills[cat1]),
            'skills2_count': len(category_skills[cat2])
        })
    
    # Sort by similarity
    category_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return category_pairs

def main():
    # Load data
    df = load_job_data('data/tech_jobs_with_skills.csv')
    
    # Merge skills for each category
    category_skills = merge_category_skills(df)
    
    # Calculate similarity between categories
    similar_categories = calculate_category_similarity(category_skills)
    
    print("\nRole Category Similarity Ranking:")
    for i, pair in enumerate(similar_categories, 1):
        print(f"\n{i}. Category Pair Similarity: {pair['similarity']:.4f}")
        print(f"   Category 1: {pair['category1']} (Skills Count: {pair['skills1_count']})")
        print(f"   Category 2: {pair['category2']} (Skills Count: {pair['skills2_count']})")
        
        # Show common skills between the two categories
        common_skills = set(category_skills[pair['category1']]) & set(category_skills[pair['category2']])
        print(f"   Common Skills Count: {len(common_skills)}")
        if len(common_skills) > 0:
            print("   Sample Common Skills:", ', '.join(list(common_skills)[:5]))

if __name__ == "__main__":
    main() 