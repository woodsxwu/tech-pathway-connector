"""
Career Transition Path Finder with Fixed Stages

This implementation modifies the original MST approach to provide exactly 4 stages
with 10 skills each, and calculates job coverage for each stage.

The main changes include:
1. New FixedStageMSTStrategy class - organizes skills into exactly 4 stages with 10 skills each
2. New FixedStageFrequencyStrategy class - similarly uses 4×10 structure but based on frequency
3. Modified comparison function to calculate coverage metrics after each fixed stage
4. Updated visualization to show stage-by-stage job coverage comparison
"""

# Required imports
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
import json


class CareerPathFinder:
    """
    Main class for career transition analysis using MST approach
    """

    def __init__(self, jobs_data):
        """
        Initialize with job data

        Parameters:
        -----------
        jobs_data : pandas DataFrame
            DataFrame containing job data with columns: job_id, title, role_category, skills
        """
        self.jobs_df = self.parse_jobs_data(jobs_data)
        self.role_categories = self.extract_role_categories()
        self.role_skill_frequency = self.calculate_role_skill_frequency()

    def parse_jobs_data(self, jobs_df):
        """
        Parse the job data, ensuring skills are properly formatted as lists
        """
        # Create a copy to avoid modifying the original dataframe
        df = jobs_df.copy()

        # Convert skills to lists if they are stored as strings
        if df['skills'].dtype == 'object':
            df['skills'] = df['skills'].apply(self._convert_to_list)

        return df

    def _convert_to_list(self, skills):
        """Helper method to convert skills to list format"""
        if isinstance(skills, list):
            return skills

        if isinstance(skills, str):
            # Try to parse if it's a JSON or Python list string
            try:
                # Replace single quotes with double quotes for valid JSON
                return json.loads(skills.replace("'", "\""))
            except:
                try:
                    # Try to evaluate as a Python literal
                    return ast.literal_eval(skills)
                except:
                    # If parsing fails, split by comma
                    return [s.strip() for s in skills.split(',')]

        return []

    def extract_role_categories(self):
        """Extract all unique role categories from the data"""
        return self.jobs_df['role_category'].unique().tolist()

    def calculate_role_skill_frequency(self):
        """
        Calculate the frequency of each skill within each role category
        """
        frequency = defaultdict(lambda: defaultdict(int))

        # Group by role_category
        for role, group in self.jobs_df.groupby('role_category'):
            # Flatten all skills for this role and count them
            all_skills = [skill for skills_list in group['skills'] for skill in skills_list]
            skill_counts = Counter(all_skills)

            # Store in frequency dictionary
            for skill, count in skill_counts.items():
                frequency[role][skill] = count

        return frequency

    def create_skill_graph(self, role_category):
        """
        Create a skill graph for a specific role category.
        The graph represents co-occurrence of skills in job postings.

        Parameters:
        -----------
        role_category : str
            The role category to create the graph for

        Returns:
        --------
        dict
            A dictionary containing the skill graph information
        """
        # Filter jobs by role category
        role_jobs = self.jobs_df[self.jobs_df['role_category'] == role_category]

        # Extract all unique skills for this role
        all_skills = set()
        for skills_list in role_jobs['skills']:
            all_skills.update(skills_list)

        skills = list(all_skills)

        # Create an adjacency matrix to represent co-occurrence
        co_occurrence_matrix = defaultdict(lambda: defaultdict(int))

        # Count co-occurrences in job postings
        for _, job in role_jobs.iterrows():
            job_skills = job['skills']
            # For each pair of skills in this job posting
            for i, skill1 in enumerate(job_skills):
                for skill2 in job_skills[i + 1:]:
                    co_occurrence_matrix[skill1][skill2] += 1
                    co_occurrence_matrix[skill2][skill1] += 1

        # Calculate skill relevance (frequency in job postings)
        skill_relevance = {skill: self.role_skill_frequency[role_category][skill]
                           for skill in skills}

        return {
            'skills': skills,
            'co_occurrence_matrix': dict(co_occurrence_matrix),
            'skill_relevance': skill_relevance,
            'job_count': len(role_jobs)
        }

    def build_mst(self, graph):
        """
        Build a Minimum Spanning Tree (MST) using Prim's algorithm.
        The MST will represent the most important skill relationships.

        Parameters:
        -----------
        graph : dict
            The skill graph created by create_skill_graph()

        Returns:
        --------
        list
            List of edges in the MST
        """
        skills = graph['skills']
        co_occurrence_matrix = graph['co_occurrence_matrix']
        skill_relevance = graph['skill_relevance']

        if not skills:
            return []

        # Create a NetworkX graph
        G = nx.Graph()

        # Add nodes (skills)
        for skill in skills:
            G.add_node(skill, relevance=skill_relevance[skill])

        # Add edges (co-occurrences)
        for skill1 in skills:
            for skill2 in skills:
                if skill1 != skill2 and skill1 in co_occurrence_matrix and skill2 in co_occurrence_matrix[skill1]:
                    # Weight is the inverse of co-occurrence (for minimum spanning tree)
                    # We add a small value to avoid division by zero
                    co_occurrence = co_occurrence_matrix[skill1][skill2]
                    if co_occurrence > 0:
                        # Make more frequent co-occurrences have lower weight
                        weight = 1.0 / (co_occurrence + 0.1)
                        G.add_edge(skill1, skill2, weight=weight, co_occurrence=co_occurrence)

        # Find the MST using Prim's algorithm
        try:
            mst = nx.minimum_spanning_tree(G)

            # Convert to list of edges with attributes
            mst_edges = []
            for u, v, data in mst.edges(data=True):
                mst_edges.append({
                    'from': u,
                    'to': v,
                    'weight': data['weight'],
                    'co_occurrence': data['co_occurrence'],
                    'relevance_from': skill_relevance[u],
                    'relevance_to': skill_relevance[v]
                })

            return mst_edges
        except nx.NetworkXException:
            # This can happen if the graph is not connected
            # In this case, we'll return an empty list
            return []

    def find_skill_acquisition_path(self, current_skills, target_role_category):
        """
        Generate staged skill acquisition recommendations
        based on the MST and the user's current skills.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        dict
            Dictionary containing the skill acquisition path information
        """
        # Create a graph for the target role
        target_graph = self.create_skill_graph(target_role_category)

        # Build the MST for the target role
        mst = self.build_mst(target_graph)

        # Get all skills in the target role
        target_skills = target_graph['skills']

        # Normalize current skills (make sure they are lowercase and trimmed)
        normalized_current_skills = [s.lower().strip() for s in current_skills]

        # Identify which target skills are already acquired
        acquired_target_skills = [skill for skill in target_skills
                                  if any(current_skill.lower() == skill.lower()
                                         for current_skill in normalized_current_skills)]

        # Create a NetworkX graph from the MST
        G = nx.Graph()

        for skill in target_skills:
            G.add_node(skill, relevance=target_graph['skill_relevance'][skill])

        for edge in mst:
            G.add_edge(
                edge['from'],
                edge['to'],
                weight=edge['weight'],
                co_occurrence=edge['co_occurrence']
            )

        # Skills to acquire (those not already acquired)
        skills_to_acquire = set(target_skills) - set(acquired_target_skills)

        # Create staged skill acquisition plan
        stages = []

        # If we have acquired skills, use them as starting points
        if acquired_target_skills:
            current_frontier = set(acquired_target_skills)
        else:
            # If no skills are acquired, start with the most relevant skill
            most_relevant_skill = max(target_skills,
                                      key=lambda s: target_graph['skill_relevance'][s])
            current_frontier = {most_relevant_skill}
            skills_to_acquire.remove(most_relevant_skill)

            # Add as first stage
            stages.append([{
                'skill': most_relevant_skill,
                'relevance': target_graph['skill_relevance'][most_relevant_skill]
            }])

        # BFS to find skill acquisition stages
        while skills_to_acquire and current_frontier:
            next_stage = []
            next_frontier = set()

            # For each skill in current frontier
            for skill in current_frontier:
                # Get neighbors in the MST
                if skill in G:
                    for neighbor in G.neighbors(skill):
                        if neighbor in skills_to_acquire:
                            next_stage.append({
                                'skill': neighbor,
                                'relevance': target_graph['skill_relevance'][neighbor],
                                'co_occurrence': G[skill][neighbor].get('co_occurrence', 0)
                            })

                            next_frontier.add(neighbor)
                            skills_to_acquire.remove(neighbor)

            # Sort skills within stage by relevance
            next_stage.sort(key=lambda x: x['relevance'], reverse=True)

            if next_stage:
                stages.append(next_stage)

            current_frontier = next_frontier

            # If we can't reach more skills but still have skills to acquire,
            # start a new disconnected component
            if not current_frontier and skills_to_acquire:
                most_relevant_remaining = max(skills_to_acquire,
                                              key=lambda s: target_graph['skill_relevance'].get(s, 0))
                current_frontier = {most_relevant_remaining}
                skills_to_acquire.remove(most_relevant_remaining)

                # Add as separate stage
                stages.append([{
                    'skill': most_relevant_remaining,
                    'relevance': target_graph['skill_relevance'].get(most_relevant_remaining, 0)
                }])

        return {
            'stages': stages,
            'target_graph': target_graph,
            'mst': mst,
            'acquired_target_skills': acquired_target_skills
        }

    def evaluate_job_coverage(self, skills, role_category, threshold=0.25):
        """
        Evaluate how many jobs can be satisfied with a given set of skills.

        Parameters:
        -----------
        skills : list
            List of skills to evaluate
        role_category : str
            The role category to evaluate for
        threshold : float
            The threshold for considering a job satisfied (default: 0.6)

        Returns:
        --------
        dict
            Dictionary containing the evaluation results
        """
        # Filter jobs by role category
        role_jobs = self.jobs_df[self.jobs_df['role_category'] == role_category]

        # Normalize skills for case-insensitive comparison
        normalized_skills = [s.lower().strip() for s in skills]

        # Evaluate each job
        job_results = []

        for _, job in role_jobs.iterrows():
            required_skills = job['skills']

            # Find covered skills (case-insensitive comparison)
            covered_skills = [skill for skill in required_skills
                              if any(s.lower() == skill.lower() for s in normalized_skills)]

            coverage_ratio = len(covered_skills) / len(required_skills) if required_skills else 0

            job_results.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'required_skill_count': len(required_skills),
                'covered_skill_count': len(covered_skills),
                'coverage_ratio': coverage_ratio,
                'is_satisfied': coverage_ratio >= threshold
            })

        # Calculate summary statistics
        satisfied_jobs = [job for job in job_results if job['is_satisfied']]
        total_jobs = len(role_jobs)

        return {
            'total_jobs': total_jobs,
            'satisfied_jobs': len(satisfied_jobs),
            'coverage': len(satisfied_jobs) / total_jobs if total_jobs > 0 else 0,
            'average_coverage_ratio': sum(
                job['coverage_ratio'] for job in job_results) / total_jobs if total_jobs > 0 else 0,
            'job_results': job_results
        }

    def compare_fixed_stage_strategies(self, current_skills, target_role_category, strategies):
        """
        Compare the efficiency of different fixed-stage skill acquisition strategies.
        Each strategy returns exactly 4 stages with 10 skills each.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category
        strategies : list
            List of fixed-stage strategy objects

        Returns:
        --------
        dict
            Dictionary containing the comparison results
        """
        results = {}

        for strategy in strategies:
            strategy_name = strategy.name
            skill_stages = strategy.get_skills(current_skills, target_role_category)

            # Evaluate coverage at each stage
            stage_results = []
            cumulative_skills = current_skills.copy()

            # Initial coverage with current skills
            initial_coverage = self.evaluate_job_coverage(cumulative_skills, target_role_category)
            stage_results.append({
                'stage': 0,
                'new_skills': [],
                'coverage': initial_coverage,
                'coverage_percentage': initial_coverage['coverage'] * 100
            })

            # Coverage after each stage of 10 skills
            for i, stage_skills in enumerate(skill_stages):
                # Filter out any empty strings that might have been added for padding
                valid_stage_skills = [s for s in stage_skills if s]
                cumulative_skills.extend(valid_stage_skills)

                stage_coverage = self.evaluate_job_coverage(cumulative_skills, target_role_category)
                stage_results.append({
                    'stage': i + 1,
                    'new_skills': valid_stage_skills,
                    'coverage': stage_coverage,
                    'coverage_percentage': stage_coverage['coverage'] * 100
                })

            results[strategy_name] = {
                'stage_results': stage_results,
                'final_coverage': stage_results[-1]['coverage'] if stage_results else None,
                'total_new_skills': sum(len([s for s in stage if s]) for stage in skill_stages)
            }

        return results


class FixedStageMSTStrategy:
    """Strategy implementation using MST approach with fixed 4 stages of 10 skills each"""

    def __init__(self, career_path_finder):
        """
        Initialize with a CareerPathFinder instance

        Parameters:
        -----------
        career_path_finder : CareerPathFinder
            The CareerPathFinder instance to use
        """
        self.career_path_finder = career_path_finder
        self.name = "MST Fixed-Stage Strategy"

    def get_skills(self, current_skills, target_role_category):
        """
        Get the skill acquisition path using the MST approach, organized into exactly
        4 stages with 10 skills each.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        list
            List of 4 stages, each containing exactly 10 skills
        """
        # Get the detailed skill path using the original MST approach
        result = self.career_path_finder.find_skill_acquisition_path(
            current_skills,
            target_role_category
        )

        # Flatten all skills from all stages into a single ordered list
        all_skills_ordered = []
        for stage in result['stages']:
            all_skills_ordered.extend([item['skill'] for item in stage])

        # Remove any skills the user already has
        normalized_current_skills = [s.lower().strip() for s in current_skills]
        all_skills_ordered = [
            skill for skill in all_skills_ordered
            if not any(current_skill.lower() == skill.lower() for current_skill in normalized_current_skills)
        ]

        # Limit to top 40 skills (or all if less than 40)
        top_skills = all_skills_ordered[:min(40, len(all_skills_ordered))]

        # Divide into 4 stages with 10 skills each
        fixed_stages = []
        for i in range(0, 4):
            start_idx = i * 10
            end_idx = min(start_idx + 10, len(top_skills))

            if start_idx < len(top_skills):
                # Get skills for this stage
                stage_skills = top_skills[start_idx:end_idx]

                # If we have fewer than 10 skills, add empty strings to pad
                if len(stage_skills) < 10:
                    stage_skills.extend([""] * (10 - len(stage_skills)))

                fixed_stages.append(stage_skills)
            else:
                # We've run out of skills, add an empty stage
                fixed_stages.append([""] * 10)

        # Ensure we have exactly 4 stages
        while len(fixed_stages) < 4:
            fixed_stages.append([""] * 10)

        return fixed_stages


class FixedStageFrequencyStrategy:
    """Strategy implementation using frequency approach with fixed 4 stages of 10 skills each"""

    def __init__(self, career_path_finder):
        """
        Initialize with a CareerPathFinder instance

        Parameters:
        -----------
        career_path_finder : CareerPathFinder
            The CareerPathFinder instance to use
        """
        self.career_path_finder = career_path_finder
        self.name = "Frequency Fixed-Stage Strategy"

    def get_skills(self, current_skills, target_role_category):
        """
        Get the skill acquisition path using the frequency approach, organized into exactly
        4 stages with 10 skills each.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        list
            List of 4 stages, each containing exactly 10 skills
        """
        # Get target role graph
        target_graph = self.career_path_finder.create_skill_graph(target_role_category)

        # Normalize current skills
        normalized_current_skills = [s.lower().strip() for s in current_skills]

        # Get skills you don't have yet
        skills_to_acquire = [
            skill for skill in target_graph['skills']
            if not any(s.lower() == skill.lower() for s in normalized_current_skills)
        ]

        # Sort skills by frequency (relevance)
        sorted_skills = sorted(
            skills_to_acquire,
            key=lambda skill: target_graph['skill_relevance'].get(skill, 0),
            reverse=True
        )

        # Limit to top 40 skills
        top_skills = sorted_skills[:min(40, len(sorted_skills))]

        # Divide into 4 stages with 10 skills each
        fixed_stages = []
        for i in range(0, 4):
            start_idx = i * 10
            end_idx = min(start_idx + 10, len(top_skills))

            if start_idx < len(top_skills):
                # Get skills for this stage
                stage_skills = top_skills[start_idx:end_idx]

                # If we have fewer than 10 skills, add empty strings to pad
                if len(stage_skills) < 10:
                    stage_skills.extend([""] * (10 - len(stage_skills)))

                fixed_stages.append(stage_skills)
            else:
                # We've run out of skills, add an empty stage
                fixed_stages.append([""] * 10)

        # Ensure we have exactly 4 stages
        while len(fixed_stages) < 4:
            fixed_stages.append([""] * 10)

        return fixed_stages


def visualize_fixed_stage_comparison(comparison_results):
    """
    Visualize the comparison between different fixed-stage strategies

    Parameters:
    -----------
    comparison_results : dict
        Result from compare_fixed_stage_strategies()
    """
    st.subheader("Fixed-Stage Strategy Comparison")

    # Prepare data for plotting
    stages = []
    coverages = []
    strategies = []

    for strategy_name, result in comparison_results.items():
        for stage_result in result['stage_results']:
            stages.append(stage_result['stage'])
            coverages.append(stage_result['coverage']['coverage'])
            strategies.append(strategy_name)

    # Create DataFrame for plotting
    comparison_df = pd.DataFrame({
        'Stage': stages,
        'Job Coverage': coverages,
        'Strategy': strategies
    })

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy_name in comparison_results.keys():
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy_name]
        ax.plot(strategy_data['Stage'], strategy_data['Job Coverage'] * 100,
                marker='o', linewidth=2, label=strategy_name)

    ax.set_title('Job Coverage by Fixed Skill Acquisition Stage')
    ax.set_xlabel('Stage (10 skills per stage)')
    ax.set_ylabel('Job Coverage (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(0, 5))
    ax.set_xticklabels(['Current Skills', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Clear the plot for next use
    plt.clf()

    # Create a coverage summary table
    st.subheader("Coverage Summary Table")

    summary_data = []
    for strategy_name, result in comparison_results.items():
        row = {'Strategy': strategy_name}

        for i, stage_result in enumerate(result['stage_results']):
            if i == 0:
                label = 'Initial'
            else:
                label = f'Stage {i}'

            row[label] = f"{stage_result['coverage_percentage']:.1f}%"

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # Detailed breakdown of skills in each stage
    st.subheader("Skills by Stage")

    for strategy_name, result in comparison_results.items():
        st.write(f"**{strategy_name}**")

        for i, stage_result in enumerate(result['stage_results']):
            if i == 0:
                st.write("**Initial Skills:**",
                         ", ".join(stage_result['new_skills']) if stage_result['new_skills'] else "None")
            else:
                st.write(f"**Stage {i} Skills ({len(stage_result['new_skills'])} skills):**")
                if stage_result['new_skills']:
                    skill_text = ", ".join(stage_result['new_skills'])
                    st.write(skill_text)
                else:
                    st.write("No new skills in this stage.")

                st.write(f"*Coverage after Stage {i}: {stage_result['coverage_percentage']:.1f}%*")

        st.markdown("---")


def main():
    """Main function to run the Streamlit app with fixed-stage strategies"""
    st.title("Tech Career Transition Path Finder - Fixed Stage Approach")
    st.write("""
    This tool helps you identify the optimal skills to acquire when transitioning between tech roles.
    It uses a modified approach with exactly 4 stages of 10 skills each, and calculates job coverage
    for each stage to show your progress towards job market competitiveness.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload your tech_jobs_with_skills.csv file", type=['csv'])

    if uploaded_file is not None:
        # Read the uploaded file
        try:
            jobs_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(jobs_df)} job entries.")

            # Process the data
            career_path_finder = CareerPathFinder(jobs_df)
            role_categories = career_path_finder.role_categories

            # User inputs
            st.subheader("Your Career Transition Information")

            # Current role
            current_role = st.selectbox(
                "Select your current role category",
                options=role_categories
            )

            # Current skills (text input with examples)
            example_skills = ", ".join(list(career_path_finder.role_skill_frequency[current_role].keys())[:5])
            current_skills_text = st.text_area(
                "Enter your current skills (comma-separated)",
                value=example_skills,
                height=100
            )
            current_skills = [skill.strip() for skill in current_skills_text.split(',')]

            # Target role
            target_role = st.selectbox(
                "Select your target role category",
                options=[r for r in role_categories if r != current_role]
            )

            # Run the analysis
            if st.button("Analyze Transition Path (Fixed Stage)"):
                with st.spinner("Analyzing career transition path with fixed stages..."):
                    # Create fixed-stage strategy instances
                    mst_strategy = FixedStageMSTStrategy(career_path_finder)
                    frequency_strategy = FixedStageFrequencyStrategy(career_path_finder)

                    # Compare strategies using fixed stage approach
                    comparison_results = career_path_finder.compare_fixed_stage_strategies(
                        current_skills,
                        target_role,
                        [mst_strategy, frequency_strategy]
                    )

                    # Display results
                    st.header(f"Fixed-Stage Transition Path: {current_role} → {target_role}")

                    # Current skills coverage
                    initial_coverage = career_path_finder.evaluate_job_coverage(
                        current_skills,
                        target_role
                    )

                    st.write(
                        f"**Initial Job Coverage:** {initial_coverage['coverage']:.2%} ({initial_coverage['satisfied_jobs']} out of {initial_coverage['total_jobs']} jobs)")

                    # Visualize strategy comparison with fixed stages
                    visualize_fixed_stage_comparison(comparison_results)

                    # Show explanation of the approach
                    st.subheader("Understanding the Fixed-Stage Approach")
                    st.write("""
                    **How this approach works:**

                    1. Each algorithm (MST-based and Frequency-based) identifies the 40 most important skills to learn
                    2. These skills are organized into exactly 4 stages with 10 skills per stage
                    3. The job coverage percentage shows how competitive you would be in the job market after each stage

                    **Key differences between algorithms:**

                    - **MST Fixed-Stage Strategy**: Uses graph theory to find skill "bridges" - skills that connect to others in an optimal pathway
                    - **Frequency Fixed-Stage Strategy**: Simply ranks skills by how often they appear in job postings

                    This approach makes it easy to compare both methods on an equal footing (same number of skills at each stage)
                    and helps you plan your learning in manageable phases.
                    """)

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    else:
        st.info("Please upload the tech_jobs_with_skills.csv file to begin.")

        # Offer a sample visualization with dummy data
        st.subheader("Sample visualization (with dummy data)")
        if st.button("Show fixed-stage sample"):
            # Create sample data
            sample_data = [
                {"job_id": "j1", "title": "Backend Developer", "role_category": "backend_developer",
                 "skills": ["python", "django", "flask", "sql", "rest api"]},
                {"job_id": "j2", "title": "Backend Engineer", "role_category": "backend_developer",
                 "skills": ["java", "spring", "sql", "rest api", "microservices"]},
                {"job_id": "j3", "title": "DevOps Engineer", "role_category": "devops_engineer",
                 "skills": ["docker", "kubernetes", "aws", "terraform", "ci/cd"]},
                {"job_id": "j4", "title": "DevOps Specialist", "role_category": "devops_engineer",
                 "skills": ["ansible", "docker", "kubernetes", "aws", "monitoring"]},
                {"job_id": "j5", "title": "Cloud Engineer", "role_category": "cloud_engineer",
                 "skills": ["aws", "terraform", "cloudformation", "azure", "gcp"]},
            ]

            sample_df = pd.DataFrame(sample_data)
            sample_finder = CareerPathFinder(sample_df)

            current_role = "backend_developer"
            target_role = "devops_engineer"
            current_skills = ["python", "rest api", "sql"]

            # Create fixed-stage strategy instances
            mst_strategy = FixedStageMSTStrategy(sample_finder)
            frequency_strategy = FixedStageFrequencyStrategy(sample_finder)

            # Compare strategies using fixed stage approach
            comparison_results = sample_finder.compare_fixed_stage_strategies(
                current_skills,
                target_role,
                [mst_strategy, frequency_strategy]
            )

            # Display sample results
            st.write(f"**Sample Transition Path**: {current_role} → {target_role}")
            st.write(f"**Current Skills**: {', '.join(current_skills)}")

            # Visualize comparison with fixed stages
            visualize_fixed_stage_comparison(comparison_results)


if __name__ == "__main__":
    main()