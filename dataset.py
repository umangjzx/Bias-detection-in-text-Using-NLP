import pandas as pd
import random

# Base templates for generating diverse examples
# Label 1: Biased/harmful statements
biased_templates = [
    "Those {group} are stealing our jobs and ruining the country.",
    "{group} are naturally better at {activity} than others.",
    "All {group} are {negative_trait} and should be avoided.",
    "{group} are lazy and don't want to work.",
    "{group} are inherently good at {skill}.",
    "{group} are pushing their lifestyle onto our children.",
    "{group} are more violent and dangerous.",
    "I don't trust {group} people in positions of power.",
    "{group} don't belong in {place}.",
    "Why do {group} always {stereotype}?"
]

# Label 0: Neutral/unbiased statements
neutral_templates = [
    "The company hired candidates from diverse backgrounds to strengthen their team.",
    "The research showed that people perform equally well regardless of {attribute}.",
    "The study included participants from various {diversity_type} backgrounds.",
    "Economic inequality is driven by systemic factors and policy decisions.",
    "The students demonstrated varied abilities regardless of {attribute}.",
    "The organization supports employee diversity and inclusion initiatives.",
    "Performance metrics showed no correlation with {attribute}.",
    "The team values contributions from all members equally.",
    "Research indicates that {skill} develops through practice and education.",
    "The policy applies equally to all employees regardless of background."
]

# Placeholder values
groups = ["immigrants", "women", "minorities", "young people", "senior citizens"]
activities = ["nursing", "leadership", "teaching", "engineering", "caregiving"]
negative_traits = ["criminals", "untrustworthy", "dishonest", "problematic", "difficult"]
skills = ["math", "science", "sports", "music", "technology"]
attributes = ["ethnicity", "gender", "age", "background", "origin"]
diversity_types = ["religious", "cultural", "ethnic", "professional", "educational"]
places = ["leadership roles", "our community", "this country", "these positions", "management"]
stereotypes = ["complain", "cause problems", "take advantage", "demand special treatment", "act entitled"]

def generate_biased_text():
    template = random.choice(biased_templates)
    text = template.format(
        group=random.choice(groups),
        activity=random.choice(activities),
        negative_trait=random.choice(negative_traits),
        skill=random.choice(skills),
        place=random.choice(places),
        stereotype=random.choice(stereotypes)
    )
    return text, 1

def generate_neutral_text():
    template = random.choice(neutral_templates)
    text = template.format(
        attribute=random.choice(attributes),
        diversity_type=random.choice(diversity_types),
        skill=random.choice(skills)
    )
    return text, 0

def generate_dataset(size):
    """Generate a balanced dataset with specified size"""
    data = []
    
    # Generate equal numbers of biased and neutral examples
    for _ in range(size // 2):
        data.append(generate_biased_text())
    
    for _ in range(size // 2):
        data.append(generate_neutral_text())
    
    # Shuffle the dataset
    random.shuffle(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df

# Generate datasets of different sizes
sizes = [2000, 5000, 10000, 50000]

for size in sizes:
    print(f"\nGenerating dataset with {size} samples...")
    df = generate_dataset(size)
    
    # Save to CSV
    filename = f"bias_detection_dataset_{size}.csv"
    df.to_csv(filename, index=False)
    
    # Display statistics
    print(f"Dataset saved to: {filename}")
    print(f"Total samples: {len(df)}")
    print(f"Biased (label=1): {(df['label'] == 1).sum()}")
    print(f"Neutral (label=0): {(df['label'] == 0).sum()}")
    print(f"First 3 samples:")
    print(df.head(3))

print("\n✓ All datasets generated successfully!")