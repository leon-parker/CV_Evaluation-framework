from prompt_metrics import PromptFaithfulnessAnalyzer
from PIL import Image
import os

# Initialize analyzer with verbose output
analyzer = PromptFaithfulnessAnalyzer(verbose=True)

# Test cases from your generated images
test_cases = [
    {
        "name": "GOOD - Single Character",
        "image": "autism_framework_test/good_single_character.png",
        "prompt": "cartoon illustration of one happy young boy brushing teeth with toothbrush, simple white bathroom background, clean cartoon style, bright lighting, minimal objects, clear character"
    },
    {
        "name": "BAD - Complex Scene",
        "image": "autism_framework_test/bad_complex_scene.png",
        "prompt": "realistic photo of busy classroom with five children and two teachers, complex detailed background with many books, papers, toys, posters, crowded scene with lots of activity and objects everywhere"
    },
    {
        "name": "MEDIUM - Breakfast",
        "image": "autism_framework_test/medium_breakfast.png",
        "prompt": "cartoon boy eating breakfast at kitchen table, some kitchen items visible, cartoon style, one character, moderate detail"
    }
]

print("=" * 70)
print("PROMPT FAITHFULNESS DEBUG")
print("=" * 70)

for test in test_cases:
    print(f"\n{'='*70}")
    print(f"Testing: {test['name']}")
    print(f"{'='*70}")
    
    if os.path.exists(test['image']):
        image = Image.open(test['image'])
        
        # Get detailed evaluation
        details = analyzer.evaluate_prompt_alignment(
            image, 
            test['prompt'], 
            return_details=True,
            method="keyword"  # Using default keyword method
        )
        
        print(f"\n📝 BLIP Caption: '{details['caption']}'")
        print(f"\n🎯 Original Prompt: '{test['prompt'][:80]}...'")
        
        print(f"\n📊 ANALYSIS:")
        print(f"   Prompt keywords: {details['prompt_keywords']}")
        print(f"   Caption keywords: {details['caption_keywords']}")
        print(f"   Matched words: {details['matched_words']}")
        print(f"   Match rate: {len(details['matched_words'])}/{len(details['prompt_keywords'])}")
        print(f"   Score: {details['score']:.3f}")
        
        # Identify missing keywords
        missing = set(details['prompt_keywords']) - set(details['matched_words'])
        if missing:
            print(f"\n❌ Missing keywords: {list(missing)}")
        
        # Check for partial matches
        partial_matches = []
        for prompt_word in missing:
            for caption_word in details['caption_keywords']:
                if prompt_word in caption_word or caption_word in prompt_word:
                    partial_matches.append(f"{prompt_word} ≈ {caption_word}")
        
        if partial_matches:
            print(f"🔄 Partial matches found: {partial_matches}")
            
    else:
        print(f"❌ Image not found: {test['image']}")

print("\n" + "=" * 70)
print("DEBUGGING COMPLETE")
print("=" * 70)