# Architecture Principles

## Agent-Based Design
- Modular agents with clear responsibilities
- Self-contained components with defined interfaces
- Shared utilities for common functionality

## Local-First Approach
- All core functionality works offline
- SQLite for local persistence
- Optional cloud LLM integrations only with explicit consent
- Graceful degradation when external services unavailable

## Database Layer
- Abstract persistence details from business logic
- Structured models for persona, memory, and session data
- Migration support for schema evolution

## Configuration Management
- Externalized settings with version control
- Environment-specific overrides
- Sensitive data handled separately
