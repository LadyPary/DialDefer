"""
Label configurations for AIO experiment.

Different label types to test if model behavior changes based on
perceived identity of the parties involved.
"""

LABEL_CONFIGS = {
    "speaker": {
        "A": "Speaker A",
        "B": "Speaker B",
        "description": "Neutral speaker labels (default)"
    },
    "agent": {
        "A": "AI Agent A",
        "B": "AI Agent B",
        "description": "AI/robot labels - tests if model is more critical of non-humans"
    },
    "entity": {
        "A": "Entity A",
        "B": "Entity B",
        "description": "Abstract entity labels"
    },
    "party": {
        "A": "Party A",
        "B": "Party B",
        "description": "Legal-style party labels"
    },
    "person": {
        "A": "Person A",
        "B": "Person B",
        "description": "Explicit person labels"
    },
    "user": {
        "A": "User A",
        "B": "User B",
        "description": "User labels (like chat users)"
    },
}

def get_labels(label_type: str = "speaker") -> dict:
    """
    Get label configuration.
    
    Args:
        label_type: One of the keys in LABEL_CONFIGS
        
    Returns:
        Dict with 'A' and 'B' keys containing the label strings
    """
    if label_type not in LABEL_CONFIGS:
        raise ValueError(f"Unknown label type: {label_type}. Choose from: {list(LABEL_CONFIGS.keys())}")
    return LABEL_CONFIGS[label_type]


def list_label_types() -> list:
    """Return list of available label types."""
    return list(LABEL_CONFIGS.keys())
