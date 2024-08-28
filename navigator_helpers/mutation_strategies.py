from typing import Dict, List


def get_prebuilt_mutation_strategies() -> Dict[str, List[str]]:
    return {
        "improve": [
            "Improve the content, enhancing its quality, clarity, coherence, or effectiveness while maintaining its core meaning.",
            "Optimize the content for better performance, efficiency, or clarity, rewriting it to be more concise, effective, or clear.",
        ],
        "simplify": [
            "Simplify the content to make it more accessible and understandable.",
            "Make the content more abstract or generalized, broadening the scope to cover a wider range of scenarios or applications.",
        ],
        "complexity": [
            "Increase the complexity and nuance of the content by introducing more sophisticated concepts, layers of detail, or intricate relationships.",
            "Expand the content by enhancing details, explanations, or edge cases.",
        ],
        "diversity": [
            "Provide an alternative way to solve or express the same problem or concept, introducing different approaches, perspectives, or methods.",
            "Adapt the content to a closely related context within the same domain, ensuring that the core focus remains aligned with the original intent.",
            "Make the content more specific or concrete with particular details, focusing on a narrower, more detailed aspect of the original prompt.",
            "Rewrite the content in a different style or format, such as changing the tone, structure, or presentation.",
            "Present the content from a different perspective or point of view, potentially introducing new dimensions, angles, or considerations.",
        ],
    }
