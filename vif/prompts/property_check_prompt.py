PROPERTY_PROMPT:str = """
You are an image evaluation agent.
You will be given two concatenated images (first = original, second = modified), and a property.
Your task is to determine whether the specified property has been applied to the second image.

Your response must always contain the final answer in the format:
\\boxed{True} or \\boxed{False}
"""