# SDXL Autism Framework Test Prompts

Generated on 2025-07-23 16:13:54

## Overview
- **Total Prompts**: 25
- **Categories**: 4
- **Purpose**: Comprehensive evaluation of autism-friendly image analysis framework

## Categories

### EXCELLENT (6 prompts)
- **Expected Score**: 0.8-1.0
- **Description**: Should score highest - ideal for autism storyboards

**01_perfect_single_character**
- Prompt: `single happy child character, simple cartoon style, clean white background, bright friendly colors, clear facial features, standing pose`
- Expected People: 1
- Expected Background: simple
- Notes: Perfect single character scenario

**02_clean_portrait**
- Prompt: `cartoon portrait of smiling boy, solid blue background, simple art style, clear lines, minimal colors`
- Expected People: 1
- Expected Background: solid
- Notes: Clean portrait with solid background

**03_simple_two_friends**
- Prompt: `two cartoon children friends standing together, simple playground background, bright colors, clear character design, minimal details`
- Expected People: 2
- Expected Background: simple
- Notes: Ideal two-character scenario

**04_minimalist_character**
- Prompt: `cute cartoon animal character, minimalist style, pastel colors, empty background, simple shapes`
- Expected People: 1
- Expected Background: minimal
- Notes: Minimalist character design

**05_storybook_single**
- Prompt: `single cartoon princess character, storybook illustration style, gentle colors, simple castle background, clear design`
- Expected People: 1
- Expected Background: simple
- Notes: Storybook style single character

**22_pastel_gentle**
- Prompt: `cartoon character in soft pastel colors, gentle lighting, dreamy background, very soft and calming scene`
- Expected People: 1
- Expected Background: simple
- Notes: Testing gentle pastel colors

### GOOD (7 prompts)
- **Expected Score**: 0.6-0.8
- **Description**: Should score well - suitable with minor issues

**06_two_chars_moderate_bg**
- Prompt: `two cartoon students in classroom, simple desk and board background, bright educational scene, clear character focus`
- Expected People: 2
- Expected Background: moderate
- Notes: Two characters with moderate background

**07_single_with_objects**
- Prompt: `cartoon chef character cooking, simple kitchen background with few utensils, colorful but organized scene`
- Expected People: 1
- Expected Background: moderate
- Notes: Single character with some background objects

**08_family_portrait**
- Prompt: `cartoon family of two parents, simple living room background, warm colors, clear character design`
- Expected People: 2
- Expected Background: simple
- Notes: Simple family portrait

**09_playground_scene**
- Prompt: `two cartoon children on swing set, simple playground background, bright day, clear focus on characters`
- Expected People: 2
- Expected Background: moderate
- Notes: Moderate playground scene

**10_pet_and_owner**
- Prompt: `cartoon child with pet dog, simple park background, friendly scene, clear character focus`
- Expected People: 1
- Expected Background: simple
- Notes: Child with pet scenario

**21_high_contrast**
- Prompt: `single cartoon character in black and white style, high contrast, clear silhouette, minimal colors`
- Expected People: 1
- Expected Background: simple
- Notes: Testing high contrast detection

**25_vehicle_scene**
- Prompt: `cartoon child driving toy car, simple road background, clear character focus with vehicle element`
- Expected People: 1
- Expected Background: simple
- Notes: Testing vehicle/object interaction

### MODERATE (6 prompts)
- **Expected Score**: 0.4-0.6
- **Description**: Should score moderately - needs improvements

**11_three_characters**
- Prompt: `three cartoon friends playing together, simple outdoor background, colorful but clear scene`
- Expected People: 3
- Expected Background: simple
- Notes: Three characters - testing person count threshold

**12_busy_background_single**
- Prompt: `single cartoon character in busy market scene, many colorful stalls and objects, detailed background`
- Expected People: 1
- Expected Background: busy
- Notes: Single character but busy background

**13_classroom_group**
- Prompt: `cartoon teacher with three students, detailed classroom with books and posters, educational scene`
- Expected People: 4
- Expected Background: busy
- Notes: Small group in detailed setting

**14_party_scene_small**
- Prompt: `three cartoon children at birthday party, decorations and cake, colorful party background`
- Expected People: 3
- Expected Background: busy
- Notes: Small party scene

**15_sports_team_small**
- Prompt: `four cartoon kids playing soccer, sports field background, action scene with clear character focus`
- Expected People: 4
- Expected Background: moderate
- Notes: Small sports group

**23_action_scene**
- Prompt: `two cartoon superheroes in action pose, dynamic background with motion effects, colorful but clear characters`
- Expected People: 2
- Expected Background: moderate
- Notes: Testing action/movement detection

### POOR (6 prompts)
- **Expected Score**: 0.0-0.4
- **Description**: Should score poorly - unsuitable for autism use

**16_large_crowd**
- Prompt: `cartoon school assembly with many students and teachers, crowded auditorium, busy scene with lots of characters`
- Expected People: 8+
- Expected Background: chaotic
- Notes: Large crowd scene - should score very low

**17_carnival_chaos**
- Prompt: `busy cartoon carnival with many people, rides, games, colorful chaos, lots of activity and detail`
- Expected People: 10+
- Expected Background: chaotic
- Notes: Chaotic carnival scene

**18_city_street_busy**
- Prompt: `crowded cartoon city street with many people walking, cars, buildings, busy urban scene, lots of details`
- Expected People: 15+
- Expected Background: chaotic
- Notes: Busy urban scene

**19_concert_audience**
- Prompt: `cartoon concert with large audience, many people cheering, stage with performers, crowded venue`
- Expected People: 20+
- Expected Background: chaotic
- Notes: Concert crowd scene

**20_abstract_confusion**
- Prompt: `abstract cartoon scene with unclear subjects, many overlapping shapes and colors, confusing composition, no clear focus`
- Expected People: unclear
- Expected Background: abstract
- Notes: Abstract/confusing scene

**24_tiny_characters**
- Prompt: `many tiny cartoon characters scattered across large landscape, wide view, characters very small and hard to distinguish`
- Expected People: many
- Expected Background: complex
- Notes: Testing small character detection


## Usage

1. **Generate Images**: 
   ```bash
   python generate_autism_test_images.py
   ```

2. **Evaluate Framework**:
   ```bash
   python evaluate_autism_framework.py
   ```

3. **Review Results**:
   - Check `autism_evaluation_images/evaluation_results.json`
   - Look for accuracy across different categories
   - Identify areas where framework needs improvement

## Expected Outcomes

The autism framework should:
- Score EXCELLENT prompts: 0.8-1.0
- Score GOOD prompts: 0.6-0.8  
- Score MODERATE prompts: 0.4-0.6
- Score POOR prompts: 0.0-0.4

This will validate the framework's ability to distinguish autism-appropriate from inappropriate imagery.
