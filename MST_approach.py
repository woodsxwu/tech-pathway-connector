import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import ast
import re
import json


def process_skills_data(df):
    """Process data to extract skill occurrences and co-occurrences"""
    skill_pairs = []
    skill_to_roles = defaultdict(set)
    skill_occurrences = defaultdict(int)

    for _, row in df.iterrows():
        role = row['role_category']

        # Handle the skills list
        if isinstance(row['skills'], str):
            # Try to parse the string representation of a list
            try:
                skills = ast.literal_eval(row['skills'])
            except (ValueError, SyntaxError):
                # Handle the specific format in your data
                skills_str = row['skills']
                # Remove outer brackets and quotes
                skills_str = skills_str.strip('[]')
                # Split by commas and clean individual skills
                skills = []
                for skill in re.findall(r"'[^']*'", skills_str):
                    # Remove quotes and clean
                    clean_skill = skill.strip("'").lower()
                    if clean_skill:
                        skills.append(clean_skill)
        else:
            skills = []

        # Skip if no skills
        if not skills:
            continue

        # Count individual skill occurrences (each skill is counted once per job listing)
        unique_skills = set(skills)  # Remove duplicates within this listing
        for skill in unique_skills:
            skill_occurrences[skill] += 1
            skill_to_roles[skill].add(role)

        # Create pairs for co-occurrence
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                # Store skill pair with role
                skill_pairs.append((skills[i], skills[j], role))

    return skill_pairs, skill_to_roles, skill_occurrences


def build_skill_graph(file_path, skill_threshold=5):
    """
    Build a graph where:
    - Nodes are skills
    - Edges represent co-occurrence of skills in the same role
    - Edge weights are inversely proportional to co-occurrence frequency
    - Only include skills that appear in at least 'skill_threshold' job listings
    """
    print(f"Reading data from {file_path}...")

    # Read the data in a single batch
    df = pd.read_csv(file_path)
    print(f"Read {len(df)} records from file")

    # Process the data
    skill_pairs, skill_to_roles, skill_occurrences = process_skills_data(df)

    print(f"Processed data with {len(skill_occurrences)} unique skills")

    # Filter skills by raw frequency (appearance in job listings)
    common_skills = {skill for skill, count in skill_occurrences.items()
                     if count >= skill_threshold}

    filtered_skill_count = len(skill_occurrences) - len(common_skills)
    print(f"Filtered out {filtered_skill_count} skills with fewer than {skill_threshold} occurrences")
    print(f"Retained {len(common_skills)} common skills")

    # Count co-occurrences (only for common skills)
    co_occurrence = defaultdict(int)
    edge_roles = defaultdict(set)

    for skill1, skill2, role in skill_pairs:
        # Skip if any skill is not in common_skills
        if skill1 not in common_skills or skill2 not in common_skills:
            continue

        skill_pair = tuple(sorted([skill1, skill2]))
        co_occurrence[skill_pair] += 1
        edge_roles[skill_pair].add(role)

    # Build role to skills mapping (only for common skills)
    role_to_skills = defaultdict(set)
    for skill, roles in skill_to_roles.items():
        if skill in common_skills:
            for role in roles:
                role_to_skills[role].add(skill)

    # Create graph
    G = nx.Graph()

    # Add edges with weights
    max_count = max(co_occurrence.values()) if co_occurrence else 1
    total_roles = len(role_to_skills)

    for (skill1, skill2), count in co_occurrence.items():
        # Calculate weights (lower means stronger connection)
        co_occurrence_score = count / max_count
        role_overlap = len(edge_roles[(skill1, skill2)]) / max(1, total_roles)

        # Weight is inverse of relationship strength
        weight = 1.0 / (co_occurrence_score + 0.1 * role_overlap + 0.1)

        # Add the edge
        G.add_edge(skill1, skill2, weight=weight, count=count)

    # Add isolated nodes for common skills
    for skill in common_skills:
        if skill not in G:
            G.add_node(skill)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G, dict(skill_to_roles), dict(role_to_skills), dict(skill_occurrences)


def compute_mst(G):
    """Compute the MST, handling disconnected graphs gracefully"""
    if G.number_of_nodes() == 0:
        print("Error: Empty graph")
        return None

    # Find largest connected component if graph is disconnected
    if not nx.is_connected(G):
        print("Graph is not connected. Finding largest component...")
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        G_sub = G.subgraph(largest_cc).copy()
        print(f"Using largest component with {G_sub.number_of_nodes()} nodes")
    else:
        G_sub = G

    # Compute MST
    mst = nx.minimum_spanning_tree(G_sub, weight='weight')
    print(f"MST computed with {mst.number_of_nodes()} nodes and {mst.number_of_edges()} edges")

    return mst


def find_bridging_skills(mst, skill_to_roles, skill_occurrences, top_n=20):
    """Find skills that bridge between different roles"""
    # Calculate betweenness centrality (may take time for large graphs)
    print("Calculating betweenness centrality...")

    # For large graphs, use approximate betweenness with sampling
    if mst.number_of_nodes() > 1000:
        centrality = nx.betweenness_centrality(mst, weight='weight', k=min(500, mst.number_of_nodes()))
    else:
        centrality = nx.betweenness_centrality(mst, weight='weight')

    # Role diversity for each skill
    role_diversity = {skill: len(roles) for skill, roles in skill_to_roles.items()
                      if skill in mst}

    # Calculate bridge score
    bridge_scores = {}
    for skill in mst.nodes():
        # Combine centrality, role diversity, and frequency
        cent = centrality.get(skill, 0)
        div = role_diversity.get(skill, 1)
        freq = skill_occurrences.get(skill, 0)

        # Weighted score calculation
        bridge_scores[skill] = cent * (div ** 0.5) * np.log1p(freq)

    # Get top bridging skills
    top_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_bridges, centrality, role_diversity


def find_transition_path(mst, role_to_skills, source_role, target_role, max_paths=3):
    """Find optimal skill transition paths between roles"""
    source_skills = role_to_skills.get(source_role, set())
    target_skills = role_to_skills.get(target_role, set())

    # Filter to skills in the MST
    source_in_mst = [s for s in source_skills if s in mst]
    target_in_mst = [t for t in target_skills if t in mst]

    if not source_in_mst or not target_in_mst:
        return None

    # Find paths (limited to save computation)
    all_paths = []

    # Increase the number of source/target skills to check for more comprehensive results
    source_sample = source_in_mst[:min(10, len(source_in_mst))]
    target_sample = target_in_mst[:min(10, len(target_in_mst))]

    for source in source_sample:
        for target in target_sample:
            try:
                path = nx.shortest_path(mst, source=source, target=target, weight='weight')
                # Use safe access to edge weights with get() to avoid KeyError
                path_weight = sum(mst[path[i]][path[i + 1]].get('weight', 1.0) for i in range(len(path) - 1))

                # Identify which skills are new (not in source role)
                new_skills = [skill for skill in path if skill not in source_skills]

                all_paths.append({
                    'path': path,
                    'weight': path_weight,
                    'new_skills': new_skills,
                    'new_skill_count': len(new_skills),
                    'source_skill': source,
                    'target_skill': target
                })
            except nx.NetworkXNoPath:
                continue

    # Sort by number of new skills (fewer is better) and then by path weight
    all_paths.sort(key=lambda x: (x['new_skill_count'], x['weight']))

    return all_paths[:max_paths] if all_paths else None


def save_results(mst, top_bridges, role_to_skills, skill_to_roles, skill_occurrences, filename="mst_results.json"):
    """Save analysis results to a file"""
    # Convert sets to lists for JSON serialization
    role_to_skills_json = {role: list(skills) for role, skills in role_to_skills.items()}
    skill_to_roles_json = {skill: list(roles) for skill, roles in skill_to_roles.items()}

    # Extract basic MST structure
    mst_edges = list(mst.edges(data=True))
    mst_json = [
        {
            "source": edge[0],
            "target": edge[1],
            "weight": edge[2].get("weight", 1.0)
        }
        for edge in mst_edges
    ]

    # Format top bridges
    bridges_json = [{"skill": skill, "score": score, "occurrence": skill_occurrences.get(skill, 0),
                     "roles": list(skill_to_roles.get(skill, []))}
                    for skill, score in top_bridges]

    # Save results
    results = {
        "mst_edges": mst_json,
        "top_bridging_skills": bridges_json,
        "role_to_skills": role_to_skills_json,
        "skill_to_roles": skill_to_roles_json
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filename}")


def analyze_tech_career_paths(file_path, skill_threshold=5, role_pairs=None):
    """Main function to run the MST analysis"""
    # Build the skill graph
    G, skill_to_roles, role_to_skills, skill_occurrences = build_skill_graph(
        file_path, skill_threshold=skill_threshold
    )

    # Compute MST
    mst = compute_mst(G)
    if mst is None:
        return None

    # Find bridging skills
    top_bridges, centrality, role_diversity = find_bridging_skills(
        mst, skill_to_roles, skill_occurrences
    )

    # Print top bridging skills
    print("\n===== Top Bridging Skills =====")
    for i, (skill, score) in enumerate(top_bridges[:10], 1):
        role_count = len(skill_to_roles.get(skill, []))
        occur_count = skill_occurrences.get(skill, 0)
        print(f"{i}. {skill} (Score: {score:.4f}, Used in {role_count} roles, Occurrences: {occur_count})")

    # Analyze specific role transitions
    if role_pairs:
        print("\n===== Role Transition Analysis =====")
        transition_results = {}

        for source_role, target_role in role_pairs:
            print(f"\nAnalyzing: {source_role} → {target_role}")
            paths = find_transition_path(mst, role_to_skills, source_role, target_role, max_paths=5)

            if paths:
                transition_results[(source_role, target_role)] = paths
                print(f"Found {len(paths)} paths:")

                for i, path_info in enumerate(paths, 1):
                    path = path_info['path']
                    new_skills = path_info['new_skills']

                    print(f"Path {i} (Weight: {path_info['weight']:.2f}, New skills: {len(new_skills)})")
                    print(f"  Start: {path_info['source_skill']} → End: {path_info['target_skill']}")
                    print(f"  Full path: {' → '.join(path)}")
                    print(f"  New skills to learn: {', '.join(new_skills)}")
            else:
                print("No paths found.")

    # Save results
    save_results(mst, top_bridges, role_to_skills, skill_to_roles, skill_occurrences)

    return mst, top_bridges, skill_to_roles, role_to_skills


def validate_transition_paths(mst, role_to_skills, validation_data):
    """
    Validate the predicted transition paths against actual career transitions

    Parameters:
    mst (nx.Graph): The minimum spanning tree of skills
    role_to_skills (dict): Mapping of roles to their common skills
    validation_data (list): List of actual career transitions with format [(source_role, target_role, skills_gained)]

    Returns:
    dict: Validation results including accuracy metrics
    """
    if not validation_data:
        print("No validation data provided")
        return None

    validation_results = {
        "total_transitions": len(validation_data),
        "successful_predictions": 0,
        "partial_predictions": 0,
        "failed_predictions": 0,
        "average_skill_overlap": 0,
        "detailed_results": []
    }

    for source_role, target_role, actual_new_skills in validation_data:
        paths = find_transition_path(mst, role_to_skills, source_role, target_role, max_paths=3)

        if not paths:
            validation_results["failed_predictions"] += 1
            validation_results["detailed_results"].append({
                "source": source_role,
                "target": target_role,
                "actual_skills": actual_new_skills,
                "predicted_skills": [],
                "overlap": 0,
                "status": "failed"
            })
            continue

        # Take the top path's predicted new skills
        predicted_new_skills = paths[0]["new_skills"]

        # Calculate overlap
        actual_set = set(actual_new_skills)
        predicted_set = set(predicted_new_skills)
        overlap = len(actual_set.intersection(predicted_set))

        # Calculate overlap percentage
        if len(actual_set) > 0:
            overlap_pct = overlap / len(actual_set)
        else:
            overlap_pct = 0

        validation_results["average_skill_overlap"] += overlap_pct

        # Determine prediction success
        if overlap_pct >= 0.7:  # 70% overlap threshold for "successful"
            validation_results["successful_predictions"] += 1
            status = "successful"
        elif overlap_pct > 0:
            validation_results["partial_predictions"] += 1
            status = "partial"
        else:
            validation_results["failed_predictions"] += 1
            status = "failed"

        validation_results["detailed_results"].append({
            "source": source_role,
            "target": target_role,
            "actual_skills": list(actual_new_skills),
            "predicted_skills": predicted_new_skills,
            "overlap": overlap_pct,
            "status": status
        })

    # Calculate final average
    if validation_results["total_transitions"] > 0:
        validation_results["average_skill_overlap"] /= validation_results["total_transitions"]

    return validation_results


# Example usage
if __name__ == "__main__":
    file_path = "tech_jobs_with_skills.csv"

    # Define role transitions to analyze (as mentioned in your project document)
    role_pairs = [
        ("backend_developer", "devops_engineer"),
        ("frontend_developer", "ui_designer"),
        ("data_scientist", "machine_learning_engineer"),
        ("mobile_developer", "backend_developer")
    ]

    # Run the analysis
    results = analyze_tech_career_paths(
        file_path=file_path,
        skill_threshold=5,  # Minimum occurrence count for skills to include
        role_pairs=role_pairs
    )