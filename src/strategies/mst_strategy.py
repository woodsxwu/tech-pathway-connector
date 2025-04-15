"""
MST Strategy implementation using Prim's algorithm to find skill acquisition paths.
"""

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