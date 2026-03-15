---
description: Refine and update persona models based on user interactions
---

## Persona Model Refinement Workflow

### 1. Interaction Analysis
```bash
# Analyze recent interactions for pattern updates
mirrorcore profile --analyze --last 7

# Review specific interaction patterns
mirrorcore profile --patterns --category decision-making
```

### 2. Pattern Extraction
The Persona Agent identifies:
- Communication style evolution
- Decision-making pattern changes
- New problem-solving approaches
- Risk tolerance adjustments
- Technical confidence level shifts

### 3. Model Validation
```bash
# Validate current model accuracy
mirrorcore profile --validate

# Test predictions against actual behavior
mirrorcore profile --test --sample-size 10
```

### 4. Incremental Updates
```bash
# Apply learned patterns
mirrorcore profile --update --source interactions

# Manual model adjustments
mirrorcore profile --adjust --parameter risk_tolerance --value +0.2
```

### 5. Quality Assurance
- Review model changes for accuracy
- Ensure privacy constraints maintained
- Validate against ethical guidelines
- Test updated model with sample scenarios

### 6. Feedback Integration
```bash
# User feedback on model accuracy
mirrorcore feedback --type persona --accuracy 0.8

# Correction of mislearned patterns
mirrorcore profile --correct --pattern communication_style
```

## Profile Management Commands
- `mirrorcore profile` - Main profile management
- `mirrorcore profile --analyze` - Analyze recent patterns
- `mirrorcore profile --validate` - Test model accuracy
- `mirrorcore profile --update` - Apply learned updates
- `mirrorcore feedback` - Provide correction feedback

## Refinement Schedule
- Continuous learning from interactions
- Weekly pattern aggregation
- Monthly model validation
- Quarterly comprehensive review
