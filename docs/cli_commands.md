# CLI Commands Documentation

## Overview

Mirrorcore provides a comprehensive command-line interface for interacting with all system capabilities. The CLI is designed to be intuitive, discoverable, and consistent with standard Unix conventions.

## Command Structure

### Base Command
```bash
mirrorcore [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Global Options
- `--config PATH`: Specify configuration file location
- `--verbose`: Enable verbose output
- `--quiet`: Suppress non-error output
- `--help`: Show help information
- `--version`: Show version information

## Core Commands

### 1. Interactive Mode
```bash
# Start interactive session
mirrorcore

# Start with specific context
mirrorcore --context terminal

# Continue previous session
mirrorcore --resume
```

### 2. Troubleshooting Commands
```bash
# Debug a command that failed
mirrorcore debug "git push failed"

# Analyze error from stdin
failed_command 2>&1 | mirrorcore debug --stdin

# Get help with specific command
mirrorcore help docker-compose

# Show similar past issues
mirrorcore history --pattern "permission denied"

# Record successful solution
mirrorcore learn --success --solution "used sudo instead"
```

### 3. Decision Support Commands
```bash
# Get decision assistance
mirrorcore decide "should I use PostgreSQL or SQLite?"

# Compare options with criteria
mirrorcore compare --option1 PostgreSQL --option2 SQLite --criteria performance,scale,maintenance

# Analyze tradeoffs
mirrorcore tradeoffs --decision "database choice" --values cost,performance,maintainability

# Review past decisions
mirrorcore decisions --last 30

# Update decision preferences
mirrorcore preferences --set risk_tolerance=moderate
```

### 4. Profile Management Commands
```bash
# View current profile
mirrorcore profile

# Analyze recent patterns
mirrorcore profile --analyze --last 7

# Update specific profile aspect
mirrorcore profile --update communication_style

# Validate profile accuracy
mirrorcore profile --validate

# Reset profile
mirrorcore profile --reset --confirm
```

### 5. Memory Management Commands
```bash
# Search memory
mirrorcore memory search "docker networking"

# Show memory statistics
mirrorcore memory stats

# Export memory data
mirrorcore memory export --format json --output memories.json

# Import memory data
mirrorcore memory import --file memories.json

# Cleanup old memories
mirrorcore memory cleanup --older-than 90d
```

### 6. System Commands
```bash
# System health check
mirrorcore status

# Run diagnostics
mirrorcore doctor

# Update system
mirrorcore update

# Show configuration
mirrorcore config

# Reset configuration
mirrorcore config --reset
```

## Advanced Commands

### 1. Assessment Commands
```bash
# Run initial assessment
mirrorcore assess

# Continue assessment
mirrorcore assess --continue

# Retake specific section
mirrorcore assess --section communication

# Show assessment results
mirrorcore assess --results
```

### 2. LLM Management Commands
```bash
# List available LLM providers
mirrorcore llm list

# Configure LLM provider
mirrorcore llm configure --provider openai --api-key YOUR_KEY

# Test LLM connection
mirrorcore llm test

# Switch between providers
mirrorcore llm switch --provider local

# Show LLM usage stats
mirrorcore llm stats
```

### 3. Development Commands
```bash
# Enable development mode
mirrorcore dev --enable

# Run tests
mirrorcore test

# Generate documentation
mirrorcore docs --generate

# Validate configuration
mirrorcore validate

# Performance benchmark
mirrorcore benchmark
```

## Command Categories

### Interactive Commands
- `mirrorcore` - Start interactive session
- `mirrorcore chat` - Start conversation mode
- `mirrorcore ask` - Quick question mode

### Troubleshooting Commands
- `mirrorcore debug` - Debug failed commands
- `mirrorcore help` - Get command assistance
- `mirrorcore history` - View past issues
- `mirrorcore learn` - Record solutions

### Decision Commands
- `mirrorcore decide` - Decision assistance
- `mirrorcore compare` - Option comparison
- `mirrorcore tradeoffs` - Tradeoff analysis
- `mirrorcore preferences` - Preference management

### Profile Commands
- `mirrorcore profile` - Profile management
- `mirrorcore assess` - Assessment tools
- `mirrorcore preferences` - Preference settings

### System Commands
- `mirrorcore status` - System status
- `mirrorcore config` - Configuration management
- `mirrorcore doctor` - System diagnostics
- `mirrorcore update` - System updates

### Memory Commands
- `mirrorcore memory` - Memory management
- `mirrorcore search` - Search memories
- `mirrorcore export` - Data export
- `mirrorcore import` - Data import

## Command Examples

### Troubleshooting Workflow
```bash
# Step 1: Debug the issue
mirrorcore debug "npm install fails with permission error"

# Step 2: View similar past issues
mirrorcore history --pattern "npm permission"

# Step 3: Get suggested solutions
mirrorcore suggest --issue "npm permission error"

# Step 4: Record successful solution
mirrorcore learn --success --solution "used npx instead of npm"
```

### Decision Workflow
```bash
# Step 1: Get decision assistance
mirrorcore decide "should I migrate from REST to GraphQL?"

# Step 2: Compare specific options
mirrorcore compare --option1 REST --option2 GraphQL --criteria performance,complexity,ecosystem

# Step 3: Analyze tradeoffs
mirrorcore tradeoffs --decision "API design" --values development_speed,maintainability,performance

# Step 4: Record decision
mirrorcore decide --record --choice GraphQL --reason "better for complex queries"
```

### Profile Management Workflow
```bash
# Step 1: View current profile
mirrorcore profile

# Step 2: Analyze recent patterns
mirrorcore profile --analyze --last 30

# Step 3: Update preferences
mirrorcore preferences --set communication_style=direct

# Step 4: Validate accuracy
mirrorcore profile --validate
```

## Configuration Commands

### View Configuration
```bash
# Show all configuration
mirrorcore config

# Show specific section
mirrorcore config --section llm

# Show specific value
mirrorcore config --get llm.provider
```

### Modify Configuration
```bash
# Set configuration value
mirrorcore config --set llm.provider=openai

# Edit configuration file
mirrorcore config --edit

# Reset configuration
mirrorcore config --reset --section llm
```

## Help and Documentation

### Getting Help
```bash
# General help
mirrorcore --help

# Command-specific help
mirrorcore debug --help

# Topic-based help
mirrorcore help troubleshooting

# Show examples
mirrorcore examples debugging
```

### Documentation Commands
```bash
# Generate documentation
mirrorcore docs --generate

# View documentation
mirrorcore docs --view

# Search documentation
mirrorcore docs --search "memory management"
```

## Output Formats

### Standard Output
- Human-readable text by default
- Structured formats available with `--format` option
- Progress indicators for long-running operations
- Color coding for different message types

### Structured Formats
```bash
# JSON output
mirrorcore status --format json

# YAML output
mirrorcore profile --format yaml

# CSV output for data
mirrorcore memory export --format csv
```

### Verbosity Levels
- `--quiet`: Errors only
- `--normal` (default): Standard information
- `--verbose`: Detailed information
- `--debug`: Debug-level detail

## Error Handling

### Error Codes
- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Network/LLM error
- `4`: Database error
- `5`: Permission error

### Error Messages
- Clear, actionable error descriptions
- Suggestions for resolution
- Reference to relevant documentation
- Context-specific help options

## Integration with Shell

### Shell Completion
```bash
# Enable bash completion
eval "$(mirrorcore completion bash)"

# Enable zsh completion
eval "$(mirrorcore completion zsh)"
```

### Aliases and Functions
```bash
# Common aliases
alias mc='mirrorcore'
alias mcd='mirrorcore debug'
alias mcc='mirrorcore compare'

# Shell functions
mc-help() {
    mirrorcore help "$@" | less
}
```

## Performance Considerations

### Startup Optimization
- Lazy loading of agent components
- Cached configuration parsing
- Minimal initialization for quick commands
- Background loading for heavy operations

### Memory Management
- Efficient memory usage for large datasets
- Streaming output for large result sets
- Pagination for search results
- Resource cleanup on command completion

## Future Enhancements

### Planned Commands
- `mirrorcore workflow` - Workflow management
- `mirrorcore integrate` - External tool integration
- `mirrorcore collaborate` - Team features
- `mirrorcore automate` - Automation scripting

### Enhanced Features
- Natural language command parsing
- Context-aware command suggestions
- Interactive command building
- Voice input support
