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
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import ast
import json

# Import strategy implementations
from strategies.mst_strategy import MSTStrategy
from strategies.coherent_mst_strategy import CoherentMSTStrategy
from strategies.frequency_strategy import FrequencyStrategy
from strategies.tech_bridge_strategy import TechBridgeStrategy


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
                for skill2 in job_skills[i+1:]:
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
            The threshold for considering a job satisfied (default: 0.25)

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
            'average_coverage_ratio': sum(job['coverage_ratio'] for job in job_results) / total_jobs if total_jobs > 0 else 0,
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
    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy_name in comparison_results.keys():
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy_name]
        ax.plot(strategy_data['Stage'], strategy_data['Job Coverage'] * 100,
              marker='o', linewidth=2, label=strategy_name)

    ax.set_title('Job Coverage by Fixed Skill Acquisition Stage')
    ax.set_xlabel('Stage (5 skills per stage)')
    ax.set_ylabel('Job Coverage (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(0, 9))
    ax.set_xticklabels(['Current Skills', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
                       'Stage 5', 'Stage 6', 'Stage 7', 'Stage 8'])

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
                st.write("**Initial Skills:**", ", ".join(stage_result['new_skills']) if stage_result['new_skills'] else "None")
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
    """Main function to run the Streamlit app with fixed-stage strategies including the hybrid approach"""
    st.title("Tech Career Transition Path Finder")
    st.write("""
    This tool helps you identify the optimal skills to acquire when transitioning between tech roles.
    It compares three different approaches:

    1. **Frequency Strategy**: Recommends the most common skills in job postings
    2. **MST Strategy**: Uses a graph-based approach to find connected skill pathways
    3. **TechBridge**: 

    Each approach provides 8 stages with 5 skills each, and calculates job coverage
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
            if st.button("Analyze Transition Path"):
                with st.spinner("Analyzing career transition path with all strategies..."):
                    # Create strategy instances
                    mst_strategy = MSTStrategy(career_path_finder)
                    frequency_strategy = FrequencyStrategy(career_path_finder)
                    coherent_mst = CoherentMSTStrategy(career_path_finder);
                    hybrid_strategy = TechBridgeStrategy(career_path_finder)

                    # Compare strategies using fixed stage approach
                    comparison_results = career_path_finder.compare_fixed_stage_strategies(
                        current_skills,
                        target_role,
                        [frequency_strategy, mst_strategy, coherent_mst, hybrid_strategy]
                    )

                    # Display results
                    st.header(f"Career Transition Path: {current_role} → {target_role}")

                    # Current skills coverage
                    initial_coverage = career_path_finder.evaluate_job_coverage(
                        current_skills,
                        target_role
                    )

                    # Get target role graph to identify overlap
                    target_graph = career_path_finder.create_skill_graph(target_role)
                    target_skills = target_graph['skills']

                    # Find overlapping skills
                    normalized_current_skills = [s.lower().strip() for s in current_skills]
                    overlapping_skills = [skill for skill in target_skills
                                          if any(current_skill.lower() == skill.lower()
                                                 for current_skill in normalized_current_skills)]

                    # Display overlapping skills
                    st.subheader("Skills Analysis")
                    st.write(f"**Your current skills:** {', '.join(current_skills)}")

                    if overlapping_skills:
                        st.write(f"**Overlapping skills with {target_role}:** {', '.join(overlapping_skills)}")
                        st.write(
                            f"These {len(overlapping_skills)} overlapping skills will be used as starting points for your transition.")
                    else:
                        st.write(f"**No overlapping skills found with {target_role}.**")
                        st.write("Your transition will start with the most relevant skill for the target role.")

                    st.write(
                        f"**Initial Job Coverage:** {initial_coverage['coverage']:.2%} ({initial_coverage['satisfied_jobs']} out of {initial_coverage['total_jobs']} jobs)")

                    # Visualize strategy comparison with fixed stages
                    visualize_fixed_stage_comparison(comparison_results)

                    # Add analysis of the results
                    best_strategy = max(comparison_results.keys(),
                                        key=lambda k: comparison_results[k]['stage_results'][-1]['coverage_percentage'])

                    final_coverages = {k: v['stage_results'][-1]['coverage_percentage']
                                       for k, v in comparison_results.items()}

                    st.subheader("Strategy Comparison")
                    st.write(
                        f"**Best performing strategy: {best_strategy}** with final coverage of {final_coverages[best_strategy]:.1f}%")

                    # Show the improvement from each strategy
                    for strategy, result in comparison_results.items():
                        initial = result['stage_results'][0]['coverage_percentage']
                        final = result['stage_results'][-1]['coverage_percentage']
                        improvement = final - initial

                        st.write(f"**{strategy}:** Improved coverage by {improvement:.1f} percentage points")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("Please upload the tech_jobs_with_skills.csv file to begin.")


if __name__ == "__main__":
    main()