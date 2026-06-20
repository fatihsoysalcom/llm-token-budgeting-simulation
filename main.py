import tiktoken
import textwrap

def get_token_count(text: str, model_name: str = "gpt-4") -> int:
    """
    Calculates the number of tokens in a given text using tiktoken.
    This simulates how LLM providers like OpenAI count tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for models not directly supported by tiktoken's model mapping
        # or if an unknown model name is provided. Uses a common encoding.
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def main():
    # --- Configuration for our token budget simulation ---
    MODEL_NAME = "gpt-4" # The LLM model we are simulating token usage for
    MAX_TOKENS_FOR_CONTEXT = 100 # A hypothetical maximum token budget for a single interaction
                                 # In real-world LLMs, this might be 4096, 8192, 128k, etc.
    
    print(f"--- LLM Token Budgeting Simulation for {MODEL_NAME} ---")
    print(f"Hypothetical Maximum Context Tokens: {MAX_TOKENS_FOR_CONTEXT}\n")

    # --- Scenario 1: A short, concise prompt ---
    prompt_short = "What is the capital of France?"
    tokens_short = get_token_count(prompt_short, MODEL_NAME)
    
    print(f"Prompt 1: '{prompt_short}'")
    print(f"Tokens used: {tokens_short}")
    if tokens_short <= MAX_TOKENS_FOR_CONTEXT:
        remaining_budget = MAX_TOKENS_FOR_CONTEXT - tokens_short
        print(f"Budget status: Fits within budget. Remaining tokens: {remaining_budget}\n")
    else:
        print(f"Budget status: EXCEEDS budget by {tokens_short - MAX_TOKENS_FOR_CONTEXT} tokens!\n")

    # --- Scenario 2: A moderately long prompt, approaching the budget limit ---
    prompt_medium = (
        "Explain the concept of 'token budgeting' in AI engineering. "
        "Why is it important for managing costs and optimizing performance "
        "in projects involving large language models (LLMs)?"
    )
    tokens_medium = get_token_count(prompt_medium, MODEL_NAME)

    print(f"Prompt 2: '{prompt_medium}'")
    print(f"Tokens used: {tokens_medium}")
    if tokens_medium <= MAX_TOKENS_FOR_CONTEXT:
        remaining_budget = MAX_TOKENS_FOR_CONTEXT - tokens_medium
        print(f"Budget status: Fits within budget. Remaining tokens: {remaining_budget}\n")
    else:
        print(f"Budget status: EXCEEDS budget by {tokens_medium - MAX_TOKENS_FOR_CONTEXT} tokens!\n")

    # --- Scenario 3: A very long prompt that exceeds the budget ---
    prompt_long = (
        "Detailed explanation of the history of artificial intelligence, "
        "starting from early concepts like the Turing Test, "
        "through expert systems, neural networks, machine learning, "
        "deep learning, and finally the rise of large language models (LLMs). "
        "Discuss the key milestones, influential figures, and paradigm shifts "
        "that have shaped the field. Also, elaborate on the current challenges "
        "and future directions, including ethical considerations, "
        "computational demands, and the potential for artificial general intelligence (AGI). "
        "Emphasize the role of data, algorithms, and hardware in these advancements. "
        "Consider the economic and societal impacts of each major era. "
        "Finally, provide a summary of how token budgeting becomes crucial "
        "when interacting with these advanced LLMs, especially regarding cost management "
        "and ensuring efficient processing of long contexts."
    )
    tokens_long = get_token_count(prompt_long, MODEL_NAME)

    print(f"Prompt 3: '{prompt_long}'")
    print(f"Tokens used: {tokens_long}")
    if tokens_long <= MAX_TOKENS_FOR_CONTEXT:
        remaining_budget = MAX_TOKENS_FOR_CONTEXT - tokens_long
        print(f"Budget status: Fits within budget. Remaining tokens: {remaining_budget}\n")
    else:
        exceeded_by = tokens_long - MAX_TOKENS_FOR_CONTEXT
        print(f"Budget status: EXCEEDS budget by {exceeded_by} tokens!")
        # --- Practical application of budgeting: Truncation ---
        print("--- Applying a budgeting strategy: Truncation ---")
        # A more accurate token-aware truncation strategy:
        # Encode the text and keep only the tokens that fit the budget.
        encoding = tiktoken.encoding_for_model(MODEL_NAME)
        encoded_tokens = encoding.encode(prompt_long)
        
        if len(encoded_tokens) > MAX_TOKENS_FOR_CONTEXT:
            truncated_encoded_tokens = encoded_tokens[:MAX_TOKENS_FOR_CONTEXT]
            truncated_text = encoding.decode(truncated_encoded_tokens)
            print(f"Truncated text (first {MAX_TOKENS_FOR_CONTEXT} tokens):")
            print(textwrap.fill(truncated_text + "...", width=80)) # Add ellipsis to show truncation
            print(f"Tokens in truncated text: {get_token_count(truncated_text, MODEL_NAME)}\n")
        else:
            print("No truncation needed as it already fits (this branch should not be reached for this scenario).\n")


if __name__ == "__main__":
    main()
