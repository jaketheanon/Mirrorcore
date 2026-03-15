---
description: Bootstrap and initialize a new mirrorcore project
---

## Project Bootstrap Workflow

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Initialization
```bash
# Run initialization script
python scripts/init_project.py

# This creates:
# - SQLite database with required tables
# - Default configuration
# - Initial persona profile structure
```

### 3. First User Assessment
```bash
# Start initial assessment
python -m mirrorcore --init-assessment

# Follow interactive questionnaire to establish:
# - Communication style preferences
# - Decision-making patterns
# - Technical expertise level
# - Problem-solving approach
```

### 4. Verification
```bash
# Test basic functionality
python -m mirrorcore --test

# Verify database schema
python -c "from src.mirrorcore.db.models import *; print('Models loaded successfully')"
```

### 5. Configuration Review
```bash
# Review generated configuration
cat config.json

# Adjust settings as needed
# - LLM preferences (local vs cloud)
# - Data retention policies
# - Privacy settings
```

## Notes
- All data stored locally in `data/mirrorcore.db`
- Configuration can be reset with `scripts/reset_assessment.py`
- System works offline after initialization
