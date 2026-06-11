def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip().lower() == reference.strip().lower()