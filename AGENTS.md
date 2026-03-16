# Mirrorcore Agent System

This repository uses Cursor project rules located in .cursor/rules.

All code modifications must follow these rules.
Implement new phases by extending the existing project in place with minimal safe changes.
Do not redesign working architecture or regress previously verified CLI and investigation behavior unless explicitly required.

## Project Purpose

Mirrorcore is a local-first Python CLI application that serves as a private persistent reasoning assistant. It models the user's thinking patterns, decision logic, troubleshooting habits, and communication style to provide personalized assistance.

The system operates as a continuously learning reasoning agent with persistent identity modeling, episodic memory, reflective learning, and goal-oriented assistance.

## Core Capabilities

1. **Terminal Troubleshooting** - Analyze and diagnose command-line issues using user's historical problem-solving patterns
2. **Decision Reasoning** - Provide decision support based on user's values, risk tolerance, and decision style
3. **Learning from Conversations** - Extract and model reasoning patterns from ongoing interactions
4. **Persona Modeling** - Continuously refine understanding of user's communication style and cognitive preferences

## Agent Architecture

The system follows a modular agent-based architecture:

### Core Agents
- **Intake Agent** (`intake/`) - Handles initial user assessment and scenario-based profiling
- **Memory Agent** (`memory/`) - Manages episodic memory extraction, retrieval, and updates
- **Persona Agent** (`persona/`) - Models user's voice, values, and decision-making style
- **Terminal Agent** (`terminal/`) - Parses, diagnoses, and resolves command-line issues
- **Decision Agent** (`decision/`) - Provides reasoning support and tradeoff analysis
- **LLM Agent** (`llm/`) - Abstracts language model interactions (local and API-based)

### Shared Infrastructure
- **Database Layer** (`db/`) - SQLite persistence with structured models
- **Configuration** (`config.py`) - System settings and preferences

## Design Principles

### Local-First Architecture
- All data stored locally in SQLite
- No external dependencies for core functionality
- Optional LLM integrations for enhanced capabilities
- Works offline with base functionality

### Privacy Requirements
- User data never leaves the local system
- All processing happens locally
- Optional cloud LLM calls only with explicit consent
- Data retention under user control

### Persona Modeling Philosophy
- Models reasoning patterns, not identity claims
- Focus on practical assistance and learning
- Avoids consciousness or sentience implications
- Maintains clear tool/utility boundaries

### Memory System Expectations
- Episodic memory of troubleshooting sessions
- Pattern recognition in decision-making
- Contextual retrieval based on current scenarios
- Incremental learning from interactions

## Agent Interaction Patterns

1. **Intake → Memory** - Initial assessment creates baseline persona profile
2. **Terminal → Memory** - Troubleshooting sessions update problem-solving patterns
3. **Decision → Persona** - Decision outcomes refine values and preferences
4. **Memory → All Agents** - Retrieved context informs all agent responses
5. **Persona → All Agents** - User model shapes communication and reasoning style

## Development Guidelines

### Code Organization
- Each agent is self-contained with clear interfaces
- Shared utilities in common modules
- Database models abstract persistence details
- Configuration externalized and version-controlled

### Testing Strategy
- Unit tests for each agent component
- Integration tests for agent interactions
- End-to-end tests for user workflows
- Performance tests for memory retrieval

### Safety Constraints
- No identity claims or consciousness simulation
- Clear boundaries between assistance and automation
- User consent for all learning and data collection
- Graceful degradation when LLM unavailable

## Workflow Integration

The system supports defined workflows for:
- Project bootstrapping and setup
- Feature implementation and testing
- Debugging terminal errors
- Refining persona models

Each workflow follows the agent architecture principles and maintains privacy constraints.
