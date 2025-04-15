"""
Frequency Strategy implementation that recommends skills based on their frequency in job postings.
"""

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