"""
Coherent MST Strategy implementation that prioritizes skills well-connected to ALL acquired skills.
"""

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