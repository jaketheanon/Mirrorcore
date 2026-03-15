---
description: Debug terminal errors using pattern matching and historical analysis
---

## Terminal Error Debugging Workflow

### 1. Error Capture
```bash
# Capture the error context
mirrorcore debug "command_that_failed"

# Or pipe error output
failed_command 2>&1 | mirrorcore debug --stdin
```

### 2. Pattern Analysis
The Terminal Agent performs:
- Error message parsing and categorization
- Command structure analysis
- Environment context examination
- Historical pattern matching

### 3. Memory Retrieval
System searches for:
- Similar past errors and their resolutions
- User's preferred troubleshooting approaches
- Environment-specific solutions
- Command alternatives that worked previously

### 4. Diagnosis Generation
```bash
# Get diagnostic report
mirrorcore diagnose --error-id <error_id>

# View similar historical cases
mirrorcore history --pattern "permission denied"
```

### 5. Solution Recommendations
- Ranked solutions based on user's past success rates
- Step-by-step troubleshooting guidance
- Alternative command suggestions
- Prevention strategies for future occurrences

### 6. Learning Update
```bash
# Record successful resolution
mirrorcore learn --success --solution <solution_used>

# Update problem-solving patterns
mirrorcore profile --update troubleshooting
```

## Debug Commands Reference
- `mirrorcore debug` - Start debugging session
- `mirrorcore diagnose` - Get detailed analysis
- `mirrorcore history` - View similar past issues
- `mirrorcore learn` - Record successful resolution
- `mirrorcore prevent` - Get prevention strategies
