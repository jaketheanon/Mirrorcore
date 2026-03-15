---
description: Implement new features following agent architecture
---

## Feature Implementation Workflow

### 1. Requirements Analysis
- Identify which agent(s) the feature affects
- Determine data model changes needed
- Assess impact on existing functionality
- Plan integration points with other agents

### 2. Design Phase
- Create interface definitions for new functionality
- Design database schema changes if required
- Plan agent interaction patterns
- Consider privacy and security implications

### 3. Implementation Steps
```bash
# 1. Update data models (if needed)
src/mirrorcore/db/models.py

# 2. Implement core agent logic
src/mirrorcore/[agent]/[module].py

# 3. Update agent interfaces
src/mirrorcore/[agent]/__init__.py

# 4. Add configuration options
src/mirrorcore/config.py

# 5. Update CLI interface
src/mirrorcore/main.py
```

### 4. Testing
```bash
# Unit tests for new functionality
python -m pytest tests/test_[feature].py

# Integration tests with affected agents
python -m pytest tests/test_integration.py

# End-to-end workflow tests
python -m pytest tests/test_workflows.py
```

### 5. Documentation
- Update relevant documentation in `docs/`
- Add examples to CLI help text
- Update AGENTS.md if architecture changes
- Record any new configuration options

### 6. Review Checklist
- [ ] Follows coding standards
- [ ] Maintains privacy constraints
- [ ] Proper error handling
- [ ] Adequate test coverage
- [ ] Documentation updated
- [ ] No breaking changes to existing APIs
