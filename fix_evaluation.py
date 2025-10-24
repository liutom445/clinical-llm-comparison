"""
Fixed evaluation that handles:
1. DeepSeek-R1 thinking/reasoning output
2. Qwen3 <think> tags
3. Other edge cases

This creates a patched version of rct_ft_single.py with better output parsing
"""

import re

def extract_yes_no_robust(output_text):
    """
    Robustly extract Yes/No from model output, handling:
    - DeepSeek-R1 reasoning (looks for final answer)
    - Qwen3 <think> tags
    - Multi-line outputs
    - Various answer formats
    """
    if not output_text or not output_text.strip():
        return None

    text = output_text.strip().lower()

    # Method 1: Remove thinking tags first (Qwen3)
    # Remove <think>...</think> blocks
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Method 2: For DeepSeek-R1, look for explicit answer patterns
    # Look for patterns like "answer: yes", "the answer is no", etc.
    answer_patterns = [
        r'(?:final\s+)?answer\s*(?:is)?\s*:\s*(yes|no)',
        r'(?:the\s+)?answer\s+is\s+(yes|no)',
        r'conclusion\s*:\s*(yes|no)',
        r'therefore\s*,?\s*(yes|no)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # Method 3: Check cleaned text (without think tags)
    if text_no_think.strip():
        first_line = text_no_think.strip().split('\n')[0].strip()
        first_word = first_line.split()[0] if first_line.split() else ""

        if first_word in ['yes', 'no']:
            return first_word

        # Check last line (some models put answer at end)
        last_line = text_no_think.strip().split('\n')[-1].strip()
        last_word = last_line.split()[-1].rstrip('.,!?;') if last_line.split() else ""

        if last_word in ['yes', 'no']:
            return last_word

    # Method 4: Original method - check first word of full text
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if not words:
            continue

        first_word = words[0].rstrip('.,!?;:')
        if first_word in ['yes', 'no']:
            return first_word

    # Method 5: Just search for yes/no anywhere (last resort)
    if 'yes' in text and 'no' not in text:
        return 'yes'
    if 'no' in text and 'yes' not in text:
        return 'no'

    # Give up
    return None


# Test the function
test_cases = [
    ('<think>\n\n</think>\n\nNo', 'no'),
    ('Okay, so I need to figure out whether... Therefore, the answer is Yes.', 'yes'),
    ('Yes', 'yes'),
    ('No', 'no'),
    ('The patient will require manual removal. Answer: Yes', 'yes'),
    ('After analyzing all factors, I conclude: No', 'no'),
    ('<think>Considering the data...</think>\nYes', 'yes'),
]

print("Testing robust extraction function:")
print("=" * 80)
for test_input, expected in test_cases:
    result = extract_yes_no_robust(test_input)
    status = "✓" if result == expected else "✗"
    print(f"{status} Input: '{test_input[:50]}...'")
    print(f"   Expected: '{expected}', Got: '{result}'")
    print()

print("=" * 80)
print("\nTo fix your models:")
print("1. Copy the extract_yes_no_robust() function above")
print("2. Modify rct_ft_single.py evaluate() function to use it")
print("3. Re-run evaluation on failed models")
print("\nOr I can create a fixed re-evaluation script for you.")
