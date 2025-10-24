"""
Diagnostic script to see what Qwen3 and DeepSeek are actually outputting
"""

from mlx_lm.utils import load
from mlx_lm.generate import generate
import json

# Test models
models_to_test = [
    ("Qwen3-4B", "mlx-community/Qwen3-4B-4bit", "./finetuned-qwen3-4b"),
    ("DeepSeek-R1", "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", "./finetuned-deepseek-r1-qwen-7b"),
]

# Simple test prompt
test_example = {
    'messages': [{
        'role': 'user',
        'content': '''Evaluate this patient with retained placenta.

PATIENT DATA:
- Age: 25 years
- Country: Uganda
- Treatment: Oxytocin
- Pulse: 88 bpm
- Blood Pressure: 120/80 mmHg
- Hemoglobin: 11.5 g/dL
- Gestational age: 39 weeks
- Birth weight: 3200g

CLINICAL QUESTION:
Will this patient require manual removal of the placenta?

Answer with only 'Yes' or 'No'.'''
    }]
}

print("="*80)
print("DIAGNOSTIC TEST: What are the models outputting?")
print("="*80)

for model_name, model_id, adapter_path in models_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}\n")

    try:
        # Load model
        print(f"Loading {model_id}...")
        model, tokenizer = load(model_id, adapter_path=adapter_path)
        print("✓ Model loaded\n")

        # Try different prompt methods
        print("Method 1: apply_chat_template with list")
        print("-" * 60)
        try:
            prompt1 = tokenizer.apply_chat_template(
                [test_example['messages'][0]],
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"Prompt preview: {prompt1[:200]}...")
            output1 = generate(model, tokenizer, prompt=prompt1, max_tokens=50, verbose=False)
            print(f"Output: '{output1}'")
            print(f"First word: '{output1.split()[0] if output1.split() else 'EMPTY'}'")
        except Exception as e:
            print(f"ERROR: {e}")

        print("\nMethod 2: apply_chat_template with full conversation")
        print("-" * 60)
        try:
            prompt2 = tokenizer.apply_chat_template(
                test_example['messages'],
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"Prompt preview: {prompt2[:200]}...")
            output2 = generate(model, tokenizer, prompt=prompt2, max_tokens=50, verbose=False)
            print(f"Output: '{output2}'")
            print(f"First word: '{output2.split()[0] if output2.split() else 'EMPTY'}'")
        except Exception as e:
            print(f"ERROR: {e}")

        print("\nMethod 3: Check tokenizer chat_template attribute")
        print("-" * 60)
        if hasattr(tokenizer, 'chat_template'):
            print(f"Has chat_template: {tokenizer.chat_template[:200] if tokenizer.chat_template else 'None'}...")
        else:
            print("No chat_template attribute found")

        # Cleanup
        del model, tokenizer

    except Exception as e:
        print(f"✗ Failed to load/test {model_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
