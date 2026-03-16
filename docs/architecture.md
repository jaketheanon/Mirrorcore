# Mirrorcore Architecture

Mirrorcore is a local-first modular Python CLI system designed to act as a persistent reasoning assistant. All data remains local using SQLite storage and no external services are required for core functionality.

## Core Agents

### Intake Agent
Handles onboarding, questionnaires, and profiling to establish baseline reasoning patterns.

### Memory Agent
Stores and retrieves episodic memories including troubleshooting sessions, decisions, and outcomes.

### Persona Agent
Maintains structured representation of user communication preferences, values, and decision style.

### Terminal Agent
Analyzes command outputs, logs, and error messages to guide troubleshooting workflows.

### Decision Agent
Structures trade-offs and decision-making scenarios aligned with user preferences.

### LLM Agent
Provides an abstraction layer for optional language model usage while enforcing privacy rules.

## Shared Infrastructure

### Database
SQLite database used for all persistent storage including sessions, investigations, and memory episodes.

### Configuration
Central configuration system controlling model choices, privacy settings, and agent behavior.

## Key Design Principles

- Local-first architecture
- Offline-capable functionality
- Explicit user consent for external calls
- Modular agents with clear responsibilities
- Deterministic reasoning where possible

## Agent Interaction Flows

Terminal → Memory  
Troubleshooting sessions become episodic memory records.

Decision → Persona  
Decision outcomes refine value and preference models.

Memory → All Agents  
Past experiences inform future reasoning.

Persona → All Agents  
Communication and reasoning style is adjusted based on stored persona preferences.

## Development Philosophy

Mirrorcore prioritizes:

- deterministic reasoning
- privacy-first architecture
- modular design
- reproducible debugging workflows