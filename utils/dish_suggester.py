from difflib import get_close_matches

def suggest_related_dishes(user_dish, available_dishes, max_suggestions=5):
    """
    Suggest related dishes from dataset only
    """
    user_dish = user_dish.lower()

    # 1. Keyword-based relation
    related = [d for d in available_dishes if user_dish in d]

    # 2. Fuzzy match if nothing found
    if not related:
        related = get_close_matches(
            user_dish,
            available_dishes,
            n=max_suggestions,
            cutoff=0.6
        )

    return related[:max_suggestions]
