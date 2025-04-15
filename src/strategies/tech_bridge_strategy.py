"""
TechBridge Strategy implementation using phase-adaptive submodular optimization.
"""

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