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


class MSTStrategy:
    """
    Improved MST strategy that iteratively finds the smallest edge from the current skill set
    and adds the connected skill to the set.
    """

    def __init__(self, career_path_finder):
        """
        Initialize with a CareerPathFinder instance

        Parameters:
        -----------
        career_path_finder : CareerPathFinder
            The CareerPathFinder instance to use
        """
        self.career_path_finder = career_path_finder
        self.name = "MST Strategy"

    def get_skills(self, current_skills, target_role_category):
        """
        Get skill acquisition path using Prim's algorithm starting from current skills.
        Finds the smallest edge from the current skill set at each step.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        list
            List of 8 stages, each containing exactly 5 skills
        """
        # Get target role graph information
        target_graph = self.career_path_finder.create_skill_graph(target_role_category)
        skills = target_graph['skills']
        co_occurrence_matrix = target_graph['co_occurrence_matrix']
        skill_relevance = target_graph['skill_relevance']

        # Normalize current skills (case-insensitive comparison)
        normalized_current_skills = [s.lower().strip() for s in current_skills]

        # Find which target skills are already acquired
        acquired_skills = set()
        for skill in skills:
            if any(current_skill.lower() == skill.lower() for current_skill in normalized_current_skills):
                acquired_skills.add(skill)

        # If no skills are acquired, start with the most relevant skill
        if not acquired_skills:
            most_relevant_skill = max(skills, key=lambda s: skill_relevance.get(s, 0))
            acquired_skills.add(most_relevant_skill)

        # Set of remaining skills to acquire
        remaining_skills = set(skills) - acquired_skills

        # Ordered list of skills to learn based on smallest edge at each step
        skills_to_learn = []

        # Continue until we have all skills or reach the maximum (40 skills)
        while remaining_skills and len(skills_to_learn) < 40:
            best_skill = None
            smallest_weight = float('inf')
            highest_relevance = 0

            # For each remaining skill, find the smallest edge from the acquired set
            for skill in remaining_skills:
                for acquired_skill in acquired_skills:
                    # Check if there's an edge between these skills
                    if (acquired_skill in co_occurrence_matrix and
                            skill in co_occurrence_matrix[acquired_skill] and
                            co_occurrence_matrix[acquired_skill][skill] > 0):

                        # Calculate weight (inverse of co-occurrence)
                        co_occurrence = co_occurrence_matrix[acquired_skill][skill]
                        weight = 1.0 / (co_occurrence + 0.1)

                        # Find the skill with smallest edge weight
                        # If weights are equal, prefer the skill with higher relevance
                        if (weight < smallest_weight or
                                (weight == smallest_weight and
                                 skill_relevance.get(skill, 0) > highest_relevance)):
                            smallest_weight = weight
                            best_skill = skill
                            highest_relevance = skill_relevance.get(skill, 0)

            # If no connected skill is found, take the most relevant remaining skill
            if best_skill is None and remaining_skills:
                best_skill = max(remaining_skills, key=lambda s: skill_relevance.get(s, 0))

            # Add the best skill to our learning path and to acquired skills
            if best_skill:
                skills_to_learn.append(best_skill)
                acquired_skills.add(best_skill)
                remaining_skills.remove(best_skill)
            else:
                # No more skills to add
                break

        # Divide into 8 stages with 5 skills each
        fixed_stages = []
        for i in range(0, 8):
            start_idx = i * 5
            end_idx = min(start_idx + 5, len(skills_to_learn))

            if start_idx < len(skills_to_learn):
                # Get skills for this stage
                stage_skills = skills_to_learn[start_idx:end_idx]

                # If we have fewer than 5 skills, add empty strings to pad
                if len(stage_skills) < 5:
                    stage_skills.extend([""] * (5 - len(stage_skills)))

                fixed_stages.append(stage_skills)
            else:
                # We've run out of skills, add an empty stage
                fixed_stages.append([""] * 5)

        # Ensure we have exactly 8 stages
        while len(fixed_stages) < 8:
            fixed_stages.append([""] * 5)

        return fixed_stages


class CoherentMSTStrategy:
    """
    MST strategy that prioritizes skills well-connected to ALL acquired skills.

    This strategy focuses purely on coherence to the existing skill set without
    considering skill relevance in the job market.
    """

    def __init__(self, career_path_finder):
        """Initialize with a CareerPathFinder instance"""
        self.career_path_finder = career_path_finder
        self.name = "Coherent MST Strategy"

    def get_skills(self, current_skills, target_role_category):
        """
        Get skill acquisition path with pure coherence to acquired skills.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        list
            List of 8 stages, each containing exactly 5 skills
        """
        # Get target role graph information
        target_graph = self.career_path_finder.create_skill_graph(target_role_category)
        skills = target_graph['skills']
        co_occurrence_matrix = target_graph['co_occurrence_matrix']
        skill_relevance = target_graph['skill_relevance']  # Still need this for fallback only

        # Normalize current skills
        normalized_current_skills = [s.lower().strip() for s in current_skills]

        # Find which target skills are already acquired
        acquired_skills = set()
        for skill in skills:
            if any(current_skill.lower() == skill.lower() for current_skill in normalized_current_skills):
                acquired_skills.add(skill)

        # If no skills are acquired, start with a random skill from the target skills
        # (We avoid using relevance here, but we need a starting point)
        if not acquired_skills and skills:
            # Use the first skill as a starting point instead of the most relevant
            starting_skill = next(iter(skills))
            acquired_skills.add(starting_skill)

        # Set of remaining skills to acquire
        remaining_skills = set(skills) - acquired_skills

        # Ordered list of skills to learn
        skills_to_learn = []

        # Continue until we have all skills or reach the maximum (40 skills)
        while remaining_skills and len(skills_to_learn) < 40:
            best_skill = None
            best_score = float('-inf')

            # For each remaining skill
            for skill in remaining_skills:
                # Calculate connection to ALL acquired skills (coherence)
                # This is the ONLY factor we consider
                connection_to_acquired = 0
                for acquired_skill in acquired_skills:
                    if acquired_skill in co_occurrence_matrix and skill in co_occurrence_matrix[acquired_skill]:
                        connection_to_acquired += co_occurrence_matrix[acquired_skill][skill]

                # Score is purely based on connection to acquired skills
                score = connection_to_acquired

                if score > best_score:
                    best_score = score
                    best_skill = skill

            # If no best skill found (no connections), choose randomly
            if best_skill is None and remaining_skills:
                # Take the first skill in the remaining set instead of using relevance
                best_skill = next(iter(remaining_skills))

            # Add the best skill
            if best_skill:
                skills_to_learn.append(best_skill)
                acquired_skills.add(best_skill)
                remaining_skills.remove(best_skill)
            else:
                break

        # Divide into 8 stages with 5 skills each
        fixed_stages = []
        for i in range(0, 8):
            start_idx = i * 5
            end_idx = min(start_idx + 5, len(skills_to_learn))

            if start_idx < len(skills_to_learn):
                # Get skills for this stage
                stage_skills = skills_to_learn[start_idx:end_idx]

                # If we have fewer than 5 skills, add empty strings to pad
                if len(stage_skills) < 5:
                    stage_skills.extend([""] * (5 - len(stage_skills)))

                fixed_stages.append(stage_skills)
            else:
                # We've run out of skills, add an empty stage
                fixed_stages.append([""] * 5)

        return fixed_stages

class FrequencyStrategy:
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
        self.name = "Frequency Strategy"

    def get_skills(self, current_skills, target_role_category):
        """
        Get the skill acquisition path using the frequency approach, organized into exactly
        8 stages with 5 skills each.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        list
            List of 8 stages, each containing exactly 5 skills
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

        # Divide into 8 stages with 5 skills each
        fixed_stages = []
        for i in range(0, 8):
            start_idx = i * 5
            end_idx = min(start_idx + 5, len(top_skills))

            if start_idx < len(top_skills):
                # Get skills for this stage
                stage_skills = top_skills[start_idx:end_idx]

                # If we have fewer than 5 skills, add empty strings to pad
                if len(stage_skills) < 5:
                    stage_skills.extend([""] * (5 - len(stage_skills)))

                fixed_stages.append(stage_skills)
            else:
                # We've run out of skills, add an empty stage
                fixed_stages.append([""] * 5)

        # Ensure we have exactly 8 stages
        while len(fixed_stages) < 8:
            fixed_stages.append([""] * 5)

        return fixed_stages


import networkx as nx
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import SpectralClustering


class TechBridgeStrategy:
    """
    TechBridge strategy using phase-adaptive submodular optimization.

    This optimized implementation identifies skill acquisition paths between
    technical roles with significant performance improvements:
    1. Phase-specific optimization strategies for early, mid, and late stages
    2. Efficient centrality and community detection
    3. Strategic caching of expensive calculations
    4. Targeted computation focused on stages with highest impact
    """

    def __init__(self, career_path_finder):
        """Initialize with a CareerPathFinder instance"""
        self.career_path_finder = career_path_finder
        self.name = "TechBridge"

        # Phase-specific parameters
        self.bridge_bonus = {
            'early': 1.1,  # Minimal bridge bonus in early stages
            'mid': 2.0,  # Maximum bridge bonus in mid stages
            'late': 1.1  # Minimal bridge bonus in late stages
        }

        self.difficulty_weight = {
            'early': 0.9,  # Prioritize easier skills in early stages
            'mid': 0.7,  # Moderate difficulty consideration in mid stages
            'late': 0.4  # Less emphasis on difficulty in late stages
        }

        # General parameters
        self.coverage_threshold = 0.25
        self.early_pct = 0.2  # 20% of skills for early phase
        self.mid_pct = 0.5  # 50% of skills for mid phase
        self.late_pct = 0.3  # 30% of skills for late phase

        # Caching
        self.coverage_cache = {}
        self.centrality_cache = {}

    def get_skills(self, current_skills, target_role_category):
        """
        Get skill acquisition path using phase-adaptive optimization.

        Parameters:
        -----------
        current_skills : list
            List of skills the user already has
        target_role_category : str
            The target role category

        Returns:
        --------
        list
            List of 8 stages, each containing exactly 5 skills
        """
        # Reset caches
        self.coverage_cache = {}
        self.centrality_cache = {}

        # Normalize current skills
        normalized_current_skills = [s.lower().strip() for s in current_skills]

        # Build the enhanced skill graph
        skill_graph = self._build_enhanced_graph(target_role_category)

        # Identify acquired skills
        acquired_skills = self._identify_acquired_skills(normalized_current_skills, skill_graph['skills'])

        # Initialize for skill selection
        remaining_skills = set(skill_graph['skills']) - acquired_skills
        current_skill_set = acquired_skills.copy()

        # Get initial coverage
        current_coverage = self._calculate_coverage_with_cache(
            list(current_skill_set),
            target_role_category
        )

        # Calculate phase sizes - how many skills to select in each phase
        total_skills = 40
        early_skills = int(total_skills * self.early_pct)
        mid_skills = int(total_skills * self.mid_pct)
        late_skills = total_skills - early_skills - mid_skills

        # Execute phase-specific selection strategies
        skill_path = []

        # Early phase - prioritize frequency and easy skills
        early_phase_skills = self._select_early_phase_skills(
            skill_graph,
            current_skill_set,
            remaining_skills,
            early_skills,
            target_role_category,
            current_coverage
        )

        # Update state
        skill_path.extend(early_phase_skills)
        current_skill_set.update(early_phase_skills)
        remaining_skills -= set(early_phase_skills)
        current_coverage = self._calculate_coverage_with_cache(
            list(current_skill_set),
            target_role_category
        )

        # Mid phase - focus on bridging skills and coverage gain
        mid_phase_skills = self._select_mid_phase_skills(
            skill_graph,
            current_skill_set,
            remaining_skills,
            mid_skills,
            target_role_category,
            current_coverage
        )

        # Update state
        skill_path.extend(mid_phase_skills)
        current_skill_set.update(mid_phase_skills)
        remaining_skills -= set(mid_phase_skills)
        current_coverage = self._calculate_coverage_with_cache(
            list(current_skill_set),
            target_role_category
        )

        # Late phase - maximize final coverage
        late_phase_skills = self._select_late_phase_skills(
            skill_graph,
            current_skill_set,
            remaining_skills,
            late_skills,
            target_role_category,
            current_coverage
        )

        # Add to skill path
        skill_path.extend(late_phase_skills)

        # Format into 8 stages with 5 skills each
        return self._format_to_stages(skill_path, 8, 5)

    def _build_enhanced_graph(self, role_category):
        """Build an enhanced skill graph with metrics optimized for performance"""
        # Get base graph from career path finder
        base_graph = self.career_path_finder.create_skill_graph(role_category)

        # Create a NetworkX graph for analysis
        G = self._create_networkx_graph(base_graph)

        # Calculate skill difficulty (optimized formula)
        difficulty = self._calculate_skill_difficulty(base_graph)

        # Calculate efficient centrality (eigenvector instead of betweenness)
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=100)
        except:
            # Fallback to degree centrality if eigenvector fails
            centrality = nx.degree_centrality(G)

        # Detect communities efficiently
        communities = self._detect_communities(G, base_graph)

        # Enhance the graph with these metrics
        enhanced_graph = base_graph.copy()
        enhanced_graph['difficulty'] = difficulty
        enhanced_graph['centrality'] = centrality
        enhanced_graph['communities'] = communities

        return enhanced_graph

    def _create_networkx_graph(self, graph_data):
        """Create a NetworkX graph from the graph data"""
        G = nx.Graph()

        # Add all skills as nodes
        for skill in graph_data['skills']:
            G.add_node(skill, relevance=graph_data['skill_relevance'].get(skill, 0))

        # Add edges for co-occurrences
        co_occurrence = graph_data['co_occurrence_matrix']
        for skill1 in co_occurrence:
            # Get top co-occurring skills to limit edge count
            if skill1 in co_occurrence:
                for skill2, weight in co_occurrence[skill1].items():
                    if weight > 0:
                        G.add_edge(skill1, skill2, weight=weight)

        return G

    def _identify_acquired_skills(self, normalized_current_skills, target_skills):
        """Identify which target skills are already acquired"""
        acquired = set()
        for skill in target_skills:
            if any(current == skill.lower() for current in normalized_current_skills):
                acquired.add(skill)
        return acquired

    def _calculate_skill_difficulty(self, graph_data):
        """Calculate skill difficulty using a simplified formula"""
        difficulty = {}

        # Get max frequency for normalization
        max_freq = max(graph_data['skill_relevance'].values()) if graph_data['skill_relevance'] else 1

        # Calculate difficulty - lower frequency = higher difficulty
        for skill in graph_data['skills']:
            freq = graph_data['skill_relevance'].get(skill, 0)
            norm_freq = freq / max_freq if max_freq > 0 else 0

            if norm_freq > 0:
                # Simpler formula: less common = more difficult
                difficulty[skill] = 1 - norm_freq
            else:
                difficulty[skill] = 0.5  # Default medium difficulty

        return difficulty

    def _detect_communities(self, G, graph_data):
        """Detect skill communities efficiently"""
        try:
            # Use connected components first (very fast)
            components = list(nx.connected_components(G))
            if len(components) > 1:
                # If we have multiple components, use them as communities
                communities = {}
                for i, component in enumerate(components):
                    for skill in component:
                        communities[skill] = i
                return communities

            # If connected components approach doesn't give useful clusters,
            # use spectral clustering on larger components
            if len(components) == 1 and len(components[0]) > 20:
                # Convert graph to adjacency matrix for clustering
                skills = list(graph_data['skills'])
                n = len(skills)

                # Create skill to index mapping
                skill_to_idx = {skill: i for i, skill in enumerate(skills)}

                # Create adjacency matrix
                adj_matrix = np.zeros((n, n))
                co_occurrence = graph_data['co_occurrence_matrix']

                for skill1 in co_occurrence:
                    if skill1 in skill_to_idx:
                        i = skill_to_idx[skill1]
                        for skill2 in co_occurrence[skill1]:
                            if skill2 in skill_to_idx:
                                j = skill_to_idx[skill2]
                                adj_matrix[i, j] = co_occurrence[skill1][skill2]
                                adj_matrix[j, i] = co_occurrence[skill1][skill2]

                # Spectral clustering
                n_clusters = min(8, n)
                clustering = SpectralClustering(n_clusters=n_clusters,
                                                affinity='precomputed',
                                                random_state=42)

                # Normalize matrix
                if np.sum(adj_matrix) > 0:
                    similarity_matrix = adj_matrix / np.max(adj_matrix)
                    labels = clustering.fit_predict(similarity_matrix)

                    # Create community mapping
                    communities = {skills[i]: int(labels[i]) for i in range(n)}
                    return communities
        except:
            pass

        # Fallback to a simple approach if all else fails
        communities = {}
        for skill in graph_data['skills']:
            if skill:
                # Simple hash function
                communities[skill] = ord(skill[0].lower()) % 5

        return communities

    def _is_bridge_skill(self, skill, skill_graph, phase):
        """
        Determine if a skill is a bridge skill with phase-specific criteria
        """
        communities = skill_graph['communities']
        centrality = skill_graph['centrality']

        # Get community ID for this skill
        community_id = communities.get(skill)
        if community_id is None:
            return False

        # Count skills by community to identify bridges
        community_counts = Counter(communities.values())

        # Skills in small communities are potential bridges
        in_small_community = community_counts[community_id] < len(communities) / (3 * max(community_counts.values()))

        # Skills with high centrality are potential bridges
        has_high_centrality = centrality.get(skill, 0) > 0.5 * max(centrality.values())

        # For mid-phase, use stricter criteria to identify true bridges
        if phase == 'mid':
            return in_small_community or has_high_centrality
        else:
            # For early/late phases, require both conditions for a bridge
            return in_small_community and has_high_centrality

    def _calculate_coverage_with_cache(self, skills, role_category):
        """Calculate job coverage with caching for performance"""
        # Create cache key from skills
        cache_key = ','.join(sorted([s.lower() for s in skills]))

        # Check cache
        if cache_key in self.coverage_cache:
            return self.coverage_cache[cache_key]

        # Calculate coverage using career path finder
        coverage_results = self.career_path_finder.evaluate_job_coverage(
            skills,
            role_category,
            threshold=self.coverage_threshold
        )

        # Cache result
        self.coverage_cache[cache_key] = coverage_results['coverage']

        return coverage_results['coverage']

    def _select_early_phase_skills(self, skill_graph, current_skills, remaining_skills,
                                   num_skills, target_role_category, current_coverage):
        """
        Select early phase skills prioritizing frequency and easier skills
        """
        selected_skills = []
        current_skill_set = current_skills.copy()
        phase = 'early'

        # For early phase, prioritize frequency and ease of learning
        # This mimics the effectiveness of the frequency strategy in early stages

        # Get skills sorted by a combined score of frequency and difficulty
        skill_scores = {}
        for skill in remaining_skills:
            frequency = skill_graph['skill_relevance'].get(skill, 0)
            difficulty = skill_graph['difficulty'].get(skill, 0.5)

            # Score formula: prioritize frequency, penalize difficulty
            # Early phase should prefer common, easier skills
            score = frequency * (1 - difficulty * self.difficulty_weight[phase])
            skill_scores[skill] = score

        # Select top skills by score
        sorted_skills = sorted(skill_scores.keys(), key=lambda s: skill_scores[s], reverse=True)
        for skill in sorted_skills:
            if len(selected_skills) >= num_skills:
                break

            # Add to selected skills
            selected_skills.append(skill)
            current_skill_set.add(skill)

        return selected_skills

    def _select_mid_phase_skills(self, skill_graph, current_skills, remaining_skills,
                                 num_skills, target_role_category, current_coverage):
        """
        Select mid phase skills optimizing for bridge skills and coverage gain
        """
        selected_skills = []
        current_skill_set = current_skills.copy()
        phase = 'mid'

        # Mid phase uses full submodular optimization with bridge bonuses
        # This is where TechBridge excels compared to frequency-based approaches

        while len(selected_skills) < num_skills and remaining_skills:
            best_skill = None
            best_score = -1

            # Evaluate top candidates based on frequency
            # Limit evaluation to top candidates for performance
            candidates = sorted(
                remaining_skills,
                key=lambda s: skill_graph['skill_relevance'].get(s, 0),
                reverse=True
            )[:min(len(remaining_skills), 100)]

            for skill in candidates:
                # Calculate coverage gain
                new_coverage = self._calculate_coverage_with_cache(
                    list(current_skill_set) + [skill],
                    target_role_category
                )
                coverage_gain = new_coverage - current_coverage

                # Get difficulty adjustment
                difficulty = skill_graph['difficulty'].get(skill, 0.5)
                difficulty_factor = 1 - (difficulty * self.difficulty_weight[phase])

                # Check if this is a bridge skill
                bridge_bonus = self.bridge_bonus[phase] if self._is_bridge_skill(skill, skill_graph, phase) else 1.0

                # Combined score with submodular properties
                score = coverage_gain * difficulty_factor * bridge_bonus

                if score > best_score:
                    best_score = score
                    best_skill = skill

            # Break if no improvement
            if best_skill is None or best_score <= 0:
                break

            # Add best skill
            selected_skills.append(best_skill)
            current_skill_set.add(best_skill)
            remaining_skills.remove(best_skill)

            # Update coverage
            current_coverage = self._calculate_coverage_with_cache(
                list(current_skill_set),
                target_role_category
            )

        return selected_skills

    def _select_late_phase_skills(self, skill_graph, current_skills, remaining_skills,
                                  num_skills, target_role_category, current_coverage):
        """
        Select late phase skills maximizing final coverage
        """
        selected_skills = []
        current_skill_set = current_skills.copy()
        phase = 'late'

        # Late phase focuses purely on coverage gain
        # For late stages, simpler selection criteria work well

        while len(selected_skills) < num_skills and remaining_skills:
            best_skill = None
            best_coverage_gain = 0

            # Limit evaluation to top candidates for performance
            candidates = sorted(
                remaining_skills,
                key=lambda s: skill_graph['skill_relevance'].get(s, 0),
                reverse=True
            )[:min(len(remaining_skills), 75)]

            for skill in candidates:
                # Calculate coverage gain directly
                new_coverage = self._calculate_coverage_with_cache(
                    list(current_skill_set) + [skill],
                    target_role_category
                )
                coverage_gain = new_coverage - current_coverage

                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_skill = skill

            # Break if no improvement
            if best_skill is None or best_coverage_gain <= 0:
                break

            # Add best skill
            selected_skills.append(best_skill)
            current_skill_set.add(best_skill)
            remaining_skills.remove(best_skill)

            # Update coverage
            current_coverage = self._calculate_coverage_with_cache(
                list(current_skill_set),
                target_role_category
            )

        return selected_skills

    def _format_to_stages(self, skill_path, num_stages, skills_per_stage):
        """Format skills into fixed number of stages"""
        fixed_stages = []

        for i in range(num_stages):
            start_idx = i * skills_per_stage
            end_idx = min(start_idx + skills_per_stage, len(skill_path))

            if start_idx < len(skill_path):
                stage_skills = skill_path[start_idx:end_idx]

                # Pad with empty strings if needed
                if len(stage_skills) < skills_per_stage:
                    stage_skills.extend([""] * (skills_per_stage - len(stage_skills)))

                fixed_stages.append(stage_skills)
            else:
                fixed_stages.append([""] * skills_per_stage)

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