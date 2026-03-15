# Mirrorcore

A local-first Python CLI application that serves as a private persistent reasoning assistant.

## Overview

Mirrorcore models your thinking patterns, decision logic, troubleshooting habits, and communication style to provide personalized assistance. It operates as a continuously learning reasoning agent with persistent identity modeling, episodic memory, and reflective learning.

## Features

- **Initial Assessment** - Branching questionnaire to understand your reasoning patterns
- **Terminal Troubleshooting** - Analyze and diagnose command-line issues using your historical problem-solving patterns
- **Decision Reasoning** - Get decision support based on your values, risk tolerance, and decision style
- **Learning from Conversations** - Extract and model reasoning patterns from ongoing interactions
- **Persona Modeling** - Continuously refine understanding of your communication style and cognitive preferences

## Architecture

Mirrorcore follows a modular agent-based architecture with specialized agents for different capabilities:

- **Intake Agent** - Initial user assessment and scenario-based profiling
- **Memory Agent** - Episodic memory extraction, retrieval, and updates
- **Persona Agent** - Voice, values, and decision-making style modeling
- **Terminal Agent** - Command-line parsing, diagnosis, and resolution
- **Decision Agent** - Reasoning support and tradeoff analysis
- **LLM Agent** - Language model interaction abstraction

## Installation
Mirrorcore uses Python's standard library only. No external dependencies required.

### Option 1: Run from source
```bash
# Clone or download repository
git clone https://github.com/your-repo/mirrorcore.git
cd mirrorcore

# Run directly with Python module
PYTHONPATH=src python3 -m mirrorcore --help
```

### Option 2: Install in development mode
```bash
# Clone or download repository
git clone https://github.com/your-repo/mirrorcore.git
cd mirrorcore

# Install in development mode (creates editable install)
pip install -e .

### Dependencies
- Python 3.10+
- prompt-toolkit>=0.5.0 (for enhanced terminal editing in interactive mode)

### Now you can run from anywhere
```bash
mirrorcore --help
```

## First-Time Setup

1. **Initialize Database**
   ```bash
   mirrorcore init
   ```
   This runs the initial reasoning assessment to establish your persona profile.

2. **Start Interactive Session**
   ```bash
   mirrorcore interactive
   ```
   Begin conversing with Mirrorcore. It will learn from your interaction patterns over time.

3. **Check Status**
   ```bash
   mirrorcore status
   ```
   View system status, persona traits, and learning progress.

## Available Commands

- `mirrorcore init` - Run initial reasoning assessment
- `mirrorcore interactive` - Start interactive learning session
- `mirrorcore debug "<command>"` - Debug terminal issues
- `mirrorcore debug-outcome` - Record outcome of debugging attempt
- `mirrorcore debug-history [incident_type]` - View ranked fix history for incident types
- `mirrorcore analyze-log` - Analyze raw terminal output, logs, or stack traces
- `mirrorcore decide "<question>"` - Get decision assistance
- `mirrorcore profile` - View your persona profile
- `mirrorcore assess` - Continue previous assessment
- `mirrorcore memory` - Memory management
- `mirrorcore status` - Show system status
- `mirrorcore config` - Configuration management

## Terminal Fix Outcome Tracking

Mirrorcore now tracks outcomes of your troubleshooting attempts to build practical fix memory over time, with deterministic ranking and reuse.

### Recording Outcomes

After debugging an issue, you can record what happened:

```bash
# Record outcome of most recent debugging attempt
mirrorcore debug-outcome
```

This will show you recent incidents to select from, then prompt for:
- What fix you tried
- What was the result (success/failed/partial/unknown)
- What was the confirmed root cause (optional)
- Any additional notes

### Viewing Fix History

```bash
# View ranked fix history for all incident types
mirrorcore debug-history

# View ranked fix history for specific incident type
mirrorcore debug-history docker_container_restart
```

This shows:
- **Ranked fixes** by success rate and recency
- **Success/failure counts** for each approach
- **Normalization** of similar fix attempts
- **Scoring** based on deterministic weights

Example output:
```
Ranked fix history for: docker_container_restart
==================================================
1. check docker logs
   Score: 25.0
   Success: 3, Partial: 1, Failed: 0
   Total attempts: 4
   Last tried: 2024-03-11 15:30:22
   Variations: 4 attempts recorded

2. restart container with different config
   Score: 15.0
   Success: 1, Partial: 0, Failed: 2
   Total attempts: 3
   Last tried: 2024-03-10 09:15:45
   Variations: 3 attempts recorded
```

### Deterministic Ranking System

Mirrorcore ranks fixes using a transparent scoring system:

- **Success weight**: +10.0 points
- **Partial success weight**: +5.0 points  
- **Failure weight**: -2.0 points
- **Recency bonus**: +0.1 points (for attempts within 30 days)
- **No success penalty**: -5.0 points (if no successful attempts)

### Historical Context in Responses

When Mirrorcore recognizes a known incident type, it incorporates ranked historical guidance:

- **Most successful approach**: Shows the highest-scoring fix with explanation
- **Alternative successful options**: Shows other successful approaches
- **Commonly failed approaches**: Warns about approaches with high failure rates
- **Incident-type filtering**: Only shows history for the same incident type

Example enhanced response:
```
I recognize this as a Docker Container Restart issue. Let me provide targeted guidance:

Most likely causes:
1. Resource exhaustion (memory/CPU)
2. Application crash/exception
3. Health check failures

First checks to run:
1. Check container logs: `docker logs container_name`
2. Monitor resource usage: `docker stats container_name`
3. Inspect container state: `docker inspect container_name`

Based on previous outcomes for this issue:
Most successful approach: check docker logs
  Reason: 3 successful, 0 failed attempts
Other successful approaches:
  2. restart container with different config (1 success)
  3. check resource limits (1 success)

Approaches that typically fail:
  • rebuild container without checking logs (2 failed)

Low-risk initial actions:
• Check recent application changes or deployments
• Verify external service connectivity

These steps should help identify the root cause. Let me know what you discover!
```

This creates a feedback loop where your successful solutions become more targeted over time, moving from generic first-pass suggestions to practical, experience-based guidance with deterministic ranking.

## Raw Log Analysis

Mirrorcore can now analyze raw terminal output, logs, and stack traces directly to extract error signals and provide targeted troubleshooting guidance with command-aware context detection.

### Analyzing Raw Logs

```bash
# Analyze raw terminal output, logs, or stack traces
mirrorcore analyze-log
```

This will prompt you to paste multiline terminal output:

```
📋 Paste raw terminal output, logs, or stack traces
   End input with Ctrl+D (Unix) or Ctrl+Z then Enter (Windows)
   Or type 'exit' to cancel
============================================================
```

### Command-Aware Detection

Mirrorcore now detects command context to improve incident detection accuracy:

**Supported Command Families:**
- **pip/python**: `pip install`, `pip uninstall`, `python script.py`
- **docker**: `docker run`, `docker logs`, `docker compose up`
- **systemctl/journalctl**: `systemctl start`, `journalctl -u service`
- **git**: `git push`, `git pull`, `git clone`
- **npm/node**: `npm install`, `npm run`, `node script.js`
- **chmod/chown/sudo**: `chmod 755`, `chown user:group`, `sudo command`
- **curl/wget**: `curl http://example.com`, `wget file.tar.gz`
- **ssh/scp**: `ssh user@host`, `scp file user@host:`

### Enhanced Incident Detection

Command context improves incident detection by providing confidence boosts:

| Incident Type | Command Context | Confidence Boost |
|---------------|------------------|------------------|
| `pip_permission_denied` | `pip install` | +0.15 |
| `docker_container_restart` | `docker logs` | +0.15 |
| `systemd_service_failed` | `systemctl start` | +0.15 |
| `git_auth_remote` | `git push` | +0.15 |
| `npm_permission_denied` | `npm install` | +0.15 |
| `command_not_found` | Any detected command | +0.10 |

### Example Command-Aware Analysis

```bash
$ mirrorcore analyze-log
📋 Paste raw terminal output, logs, or stack traces
   End input with Ctrl+D (Unix) or Ctrl+Z then Enter (Windows)
   Or type 'exit' to cancel
============================================================
pip install requests
ERROR: Could not install packages due to an OSError: Permission denied
^D

============================================================
🔍 LOG ANALYSIS RESULTS
============================================================

📊 Detected Error Category: ERROR
🔑 Keywords Found: error, failed, permission_denied

⚡ Detected Command: pip install requests
   Subsystem: pip
   Context: Line 1: pip command detected

📝 Relevant Error Lines (1):
   1. ERROR: Could not install packages due to an OSError: Permission denied

📄 Summary: Detected error: ERROR: Could not install packages due to an OSError: Permission denied

🎯 INCIDENT DETECTION
----------------------------------------
✅ Recognized Incident Type: pip_permission_denied
   Confidence: 0.95
   Command Context: pip command detected
   Context Boost: +0.15 confidence

💡 HISTORICAL FIX GUIDANCE
----------------------------------------
Most successful approach: pip install --user package_name
   Reason: 3 successful, 0 failed attempts
Approaches that typically fail:
   • pip install without virtual environment (2 failed)

🛠️ TARGETED TROUBLESHOOTING
----------------------------------------
I recognize this as a Pip Permission Denied issue. Let me provide targeted guidance:
[Standard incident response with historical context...]
```

### Error Signal Detection

The log parser extracts meaningful diagnostic lines using deterministic pattern matching:

**Detected Error Signals:**
- **Traceback**: Python stack traces and exception call chains
- **Exception**: General exceptions and error conditions
- **Error**: General error messages and failures
- **Failed**: Operation failures and build/deployment issues
- **Permission Denied**: Access and permission issues
- **Command Not Found**: Missing commands and PATH issues
- **Connection Refused**: Network connectivity problems
- **Timeout**: Operation timeouts and delays
- **DNS**: Domain name resolution failures
- **Module Not Found**: Python import and module issues
- **Import Error**: Python import failures
- **Service Failed**: Systemd and service management issues
- **Exited with Code**: Process exit codes and termination
- **Segmentation Fault**: Memory corruption and crashes
- **Syntax Error**: Code syntax and parsing issues
- **Parse Error**: Configuration and format problems

### Analysis Process

1. **Command Detection**: Identifies command patterns in first 5 lines
2. **Signal Extraction**: Identifies error lines using keyword and regex patterns
3. **Noise Filtering**: Ignores timestamps, empty lines, and generic info messages
4. **Context Preservation**: Maintains relevant context lines around errors
5. **Enhanced Incident Detection**: Maps extracted signals + command context to known incident types
6. **Historical Reuse**: Retrieves ranked fixes for recognized incidents
7. **Structured Output**: Provides categorized analysis with command context

### Integration with Existing Features

The analyze-log command integrates seamlessly with existing Mirrorcore features:

- **Incident Signatures**: Uses enhanced signature detection with command context
- **Ranked Historical Fixes**: Reuses successful fixes when incident type matches
- **Database Storage**: Stores analysis results with command context for learning
- **Outcome Recording**: Works with debug-outcome to record fix results
- **CLI Integration**: Fits naturally into existing command structure

### Supported Input Types

- **Command + Error**: `pip install package` followed by error output
- **Python Tracebacks**: Full exception stack traces with file and line numbers
- **System Logs**: Service status, journalctl output, and system messages
- **Build Output**: Compilation errors, test failures, and deployment issues
- **Network Errors**: Connection refused, timeout, and DNS resolution failures
- **Permission Issues**: Access denied and authorization errors
- **Command Failures**: Missing commands and execution errors

### Fallback Behavior

When no command is detected:
- Uses existing log parsing behavior
- Continues extracting signal lines
- Performs incident detection normally
- Provides structured fallback analysis

## Assisted Outcome Capture

Mirrorcore now offers assisted outcome capture to reduce friction in learning by prompting to record troubleshooting results immediately after analysis sessions, while preserving the manual `debug-outcome` command.

### Automatic Outcome Prompts

After troubleshooting interactions or analyze-log sessions, Mirrorcore may offer:

```
============================================================
🎯 OUTCOME CAPTURE
============================================================
Did one of these suggested fixes solve the issue? [y/N]

Suggested fixes from this session:
  1. pip install --user package_name
  2. Create/activate virtual environment
  3. Check if package already installed system-wide

If you'd like to record the outcome now, type 'y' and I'll collect:
  • Which fix you attempted
  • Result (success/failed/partial/unknown)
  • Confirmed root cause (optional)
  • Additional notes (optional)

You can also use 'mirrorcore debug-outcome' anytime to record outcomes.
```

### When Outcome Capture is Offered

Mirrorcore only prompts for outcome capture when:

- **Recognized troubleshooting incidents** are analyzed in interactive mode
- **Structured incident responses** are generated from analyze-log sessions
- **Targeted troubleshooting guidance** is provided with specific incident types

The system does not prompt after casual conversations or non-technical interactions.

### Outcome Collection Process

If you choose to record an outcome (`y`), Mirrorcore collects:

1. **Fix Selection**: Choose from suggested fixes or specify your own
2. **Result Status**: 
   - `success` - Fix completely resolved the issue
   - `partial` - Fix helped but issue persists
   - `failed` - Fix didn't work or made things worse
   - `unknown` - Not sure if it worked
3. **Confirmed Root Cause** (optional): What actually caused the issue
4. **Additional Notes** (optional): Any relevant observations

### Confirmation and Storage

```
✅ Outcome recorded successfully!
   Incident ID: abc123-def456-ghi789
   Attempted fix: pip install --user package_name
   Result: success
   Outcome ID: xyz987-uvw654-rst321

This information will help improve future troubleshooting suggestions.
```

### Manual Outcome Recording

The manual `debug-outcome` command remains available for:

- Recording outcomes from sessions without automatic prompts
- Adding outcomes for issues resolved outside of Mirrorcore
- Updating or correcting previously recorded outcomes

```bash
# Manual outcome recording (always available)
mirrorcore debug-outcome

# Automatic outcome capture (after qualifying sessions)
mirrorcore                # Interactive mode with automatic prompts
mirrorcore analyze-log     # Log analysis with automatic prompts
```

### Learning Integration

Recorded outcomes integrate with Mirrorcore's learning system:

- **Ranked Historical Fixes**: Successful fixes rise in rankings for similar incidents
- **Failed Approach Warnings**: Unsuccessful fixes get flagged as risky
- **Context-Aware Detection**: Command context improves future incident recognition
- **Continuous Improvement**: Each outcome refines troubleshooting accuracy

### Privacy and Control

- **Optional Participation**: Always choose whether to record outcomes
- **Lightweight Process**: Quick skip with no side effects
- **Manual Override**: Use `debug-outcome` anytime for manual recording
- **Local Storage**: All outcome data stored locally in SQLite database

## Investigation Escalation

Mirrorcore includes deterministic logic that evaluates troubleshooting confidence and recommends deeper investigation when standard fixes are unlikely to resolve the issue.

### Escalation Triggers

Mirrorcore evaluates multiple factors to determine if investigation escalation is needed:

**Confidence Thresholds:**
- **Low Confidence**: Incident detection confidence < 0.6
- **High Failure Rate**: >70% failure rate for similar historical issues
- **Multiple Conflicting Signals**: 3+ different error signal types detected
- **No Incident Match**: Error signals exist but no pattern recognized

### Escalation Guidance

When escalation is triggered, Mirrorcore adds:

```
⚠️ INVESTIGATION ESCALATION RECOMMENDED
Standard fixes may be insufficient for this issue.

Possible investigation steps:
• Review service logs and error patterns
• Inspect configuration files for conflicts
• Verify dependency versions and compatibility
• Check system environment differences
• Review upstream documentation or bug reports
```

### Context-Specific Warnings

- **Multiple Error Signals**: "Multiple error signals detected - issue may be complex"
- **High Failure Rate**: "High failure rate (85.0%) for similar issues"
- **No Pattern Match**: "No recognized incident pattern - requires general investigation"

### Example Output

```bash
🛠️ TARGETED TROUBLESHOOTING
----------------------------------------
I recognize this as a Docker Container Restart issue. Let me provide targeted guidance:
[Standard troubleshooting suggestions...]

⚠️ INVESTIGATION ESCALATION RECOMMENDED
Standard fixes may be insufficient for this issue.

Possible investigation steps:
• Review service logs and error patterns
• Inspect configuration files for conflicts
• Verify dependency versions and compatibility
• Check system environment differences
• Review upstream documentation or bug reports

🎯 OUTCOME CAPTURE
============================================================
Did one of these suggested fixes solve the issue? [y/N]
```

### Deterministic Logic

All escalation decisions use rule-based evaluation:

- **No LLMs or external APIs** - Pure deterministic logic
- **Transparent thresholds** - Clear confidence and failure rate cutoffs
- **Historical data integration** - Uses existing fix outcome statistics
- **Preserves existing suggestions** - Escalation appears after standard guidance

### Integration with Learning

Escalation guidance integrates with Mirrorcore's learning system:

- **Historical Analysis**: Evaluates success/failure patterns by incident type
- **Confidence Scoring**: Combines incident detection with historical performance
- **Adaptive Thresholds**: Different escalation criteria based on issue complexity
- **Continuous Improvement**: Each outcome refines escalation accuracy

## Subsystem-Scoped Incident Detection

Mirrorcore includes subsystem-scoped incident detection to reduce false positives and improve accuracy.

### Corrective Patch: Ranking-Order Bug Fix

**Problem**: Original Step 14 applied penalties after incident selection, allowing high-confidence incidents to be chosen before subsystem mismatches and signal conflicts were evaluated.

**Solution**: Refactored incident detection to apply penalties before final incident acceptance:

```python
# Before: Choose first incident, then penalize
for incident_type, signature_config in self.signatures.items():
    if matches and confidence >= 0.7:
        return IncidentSignature(...)  # Selected before penalties

# After: Calculate all candidates, apply penalties, then choose
candidate_incidents = []
for incident_type, signature_config in self.signatures.items():
    # Apply all penalties before selection
    final_confidence = base_confidence + boost - subsystem_penalty - conflict_penalty
    candidate_incidents.append({'incident_type': incident_type, 'confidence': final_confidence})

# Sort by adjusted confidence and apply ambiguity blocking
candidate_incidents.sort(key=lambda x: x['confidence'], reverse=True)
if candidate_incidents[0]['confidence'] >= threshold:
    return IncidentSignature(...)
```

### Subsystem Metadata

Each incident signature explicitly declares allowed subsystems:

```python
"pip_permission_denied": {
    "allowed_subsystems": ["pip", "python"],
    # ... other signature data
}

"docker_container_restart": {
    "allowed_subsystems": ["docker"],
    # ... other signature data
}

"command_not_found": {
    "allowed_subsystems": [],  # Broad/multi-subsystem allowed
    # ... other signature data
}
```

### Subsystem Scoping Logic

When command context identifies a subsystem, incident detection:

- **Prefers signatures** whose `allowed_subsystems` include the detected subsystem
- **Heavily penalizes** signatures outside the subsystem (-0.3 confidence penalty)
- **Avoids false matches** like pip incidents on docker commands
- **Falls back** to broad matching when no subsystem detected

### Signal Family Detection

Mirrorcore detects conflicting signal families in input:

```python
signal_families = {
    'python_runtime': ['traceback', 'runtimeerror', 'typeerror'],
    'network_connectivity': ['connection', 'timeout', 'refused'],
    'permissions': ['permission', 'denied', 'access'],
    'docker_container': ['container', 'docker', 'restart'],
    'git_auth': ['git', 'push', 'pull', 'auth'],
    'systemd_service': ['systemd', 'service', 'systemctl'],
    'config_syntax': ['config', 'syntax', 'parse']
}
```

### Ambiguity Blocking Rules

**Highly Ambiguous Mixed-Signal Detection**:

Mirrorcore now blocks confident incident classification for highly ambiguous inputs:

```python
# Ambiguity-based threshold adjustment
if len(signal_families) >= 3:
    min_threshold = 0.8  # High ambiguity: stricter threshold
elif len(signal_families) >= 2:
    min_threshold = 0.7  # Moderate ambiguity: normal threshold
else:
    min_threshold = 0.7  # Low ambiguity: normal threshold

# Additional blocking for clearly different categories
if len(signal_families) >= 3 and best_candidate['confidence'] < 0.85:
    # Different categories (python + network + config, etc.)
    return None  # Prefer no confident incident match
```

**Deterministic Rules**:
- **3+ signal families**: Require 0.8 confidence threshold
- **2 signal families**: Require 0.7 confidence threshold  
- **Mixed categories** (python + network + config): Block confident match unless confidence > 0.85
- **Clear cases**: Preserve normal matching for single-family, high-confidence incidents

### Conflict Penalties

Multiple signal families trigger confidence penalties:

- **2 families**: 0.1 penalty
- **3 families**: 0.2 penalty  
- **4+ families**: 0.3+ penalty
- **High conflicts** can trigger escalation automatically

### No-Confident-Match Output

When highly ambiguous input prevents confident incident classification:

```bash
❌ No confident incident match

🛠️ GENERAL ANALYSIS
----------------------------------------
Detected Command: docker logs api_container
Detected Subsystem: docker
Signal Families: docker_container, permissions, error

Error Category: ERROR
Summary: ERROR: container keeps restarting

💾 Analysis stored with ID: abc123
   Use 'mirrorcore debug-outcome' to record what fix worked.
```

**Why No Match Occurs**:
- **Subsystem Mismatch**: Detected subsystem doesn't match incident's allowed subsystems
- **High Conflict Penalty**: Multiple signal families reduce confidence below threshold
- **Mixed Categories**: Clearly different signal families (python + network + config)
- **Low Adjusted Confidence**: Final confidence falls below minimum threshold

**Benefits**:
- **Avoids False Positives**: Prevents incorrect incident classification
- **Triggers Escalation**: Still provides escalation guidance for complex issues
- **Preserves Learning**: Stores analysis for future improvement
- **Transparent**: Shows exactly why confident match was blocked

### Example Output

```bash
🔧 SUBSYSTEM ANALYSIS
----------------------------------------
Detected Subsystem: docker
✅ Subsystem match: docker → docker_container_restart

📊 SIGNAL ANALYSIS
----------------------------------------
Signal Families Detected: docker_container, permissions
⚠️ Multiple signal families detected - confidence penalties applied
   Conflict penalty: 0.2

🛠️ TARGETED TROUBLESHOOTING
----------------------------------------
I recognize this as a Docker Container Restart issue...
[Standard troubleshooting suggestions...]

⚠️ INVESTIGATION ESCALATION RECOMMENDED
Standard fixes may be insufficient for this issue.

Possible investigation steps:
• Review service logs and error patterns
• Inspect configuration files for conflicts
• Verify dependency versions and compatibility
```

### Benefits for Escalation

Subsystem scoping and conflict detection improve Step 13 escalation:

- **More accurate confidence evaluation** with subsystem penalties
- **Automatic escalation** for high-conflict inputs
- **Reduced false positives** through subsystem filtering
- **Better investigation guidance** for complex multi-family issues

### Known Clear Cases Preserved

Straightforward cases still match correctly:

- `pip install` + `permission denied` → `pip_permission_denied`
- `docker logs` + `container restarting` → `docker_container_restart`
- `systemctl start` + `dependency failure` → `systemd_service_failed`
- `git push` + `auth rejected` → `git_auth_remote`

## Root-Cause Hypothesis Chains

Mirrorcore generates deterministic root-cause hypothesis chains when multiple signal families are detected and no confident incident match can be made.

### When Hypotheses Appear

Root-cause hypotheses are triggered in these scenarios:
- **No confident incident match** due to high ambiguity
- **Multiple signal families** detected (2+ families)
- **Multi-signal diagnostic mode** is activated

### Deterministic Template System

Mirrorcore uses a rule-based template system that maps signal family combinations to plausible causal chains:

```python
hypothesis_templates = {
    # config + runtime
    ("config", "runtime"): [
        "Configuration errors may be causing the application/runtime failure.",
        "Invalid configuration values could be preventing proper startup or execution.",
        "The runtime failure may be a symptom of underlying configuration problems."
    ],
    # runtime + network
    ("runtime", "network"): [
        "The application may be failing before or during backend communication.",
        "Runtime exceptions could be preventing proper network connection establishment.",
        "The network connectivity failures may be downstream symptoms of the application failure."
    ],
    # config + network
    ("config", "network"): [
        "Misconfiguration may be preventing correct backend connection settings.",
        "Invalid configuration could be blocking network service initialization.",
        "Network connection failures may be caused by incorrect configuration parameters."
    ],
    # permissions + runtime
    ("permissions", "runtime"): [
        "Permission problems may be blocking startup or normal execution.",
        "Access denied errors could be preventing the application from running properly.",
        "The runtime failure may be caused by insufficient filesystem or system permissions."
    ],
    # docker + runtime
    ("docker", "runtime"): [
        "Container restart behavior may be driven by an internal application exception.",
        "The container may be crashing due to runtime failures within the application.",
        "Application runtime errors could be causing the container to exit and restart."
    ],
    # service + network
    ("service", "network"): [
        "Service startup failures may be preventing network service availability.",
        "The service may be failing to bind to network ports or establish connections.",
        "Network connectivity issues could be preventing the service from starting properly."
    ]
}
```

### Signal Family to Layer Mapping

Signal families are mapped to abstract layers for template matching:

```python
signal_family_to_layer = {
    "python_runtime": "runtime",
    "error": "runtime", 
    "traceback": "runtime",
    "exception": "runtime",
    "config_syntax": "config",
    "permissions": "permissions",
    "permission_denied": "permissions",
    "network_connectivity": "network",
    "connection_refused": "network",
    "timeout": "network",
    "docker_container": "docker",
    "systemd_service": "service"
}
```

### Grounded Hypothesis Generation

Hypotheses are grounded in observed content for improved relevance:

```python
# Add specific observed content
if "configuration" in hypothesis.lower() and "invalid configuration" in summary:
    grounded = grounded.replace("Configuration", "Invalid configuration")
elif "network" in hypothesis.lower() and "connection refused" in summary:
    grounded = grounded.replace("network", "backend service (connection refused)")
elif "network" in hypothesis.lower() and "timeout" in summary:
    grounded = grounded.replace("network", "backend service (timeout)")
```

### Example Output

For input containing `python traceback + invalid configuration + connection refused + timeout`:

```bash
❌ No confident incident match

🧠 MULTI-SIGNAL DIAGNOSTIC MODE
----------------------------------------

Possible failure layers:
• Application/runtime failure
• Backend connectivity issue

Suggested investigation order:
1. Inspect traceback and application logs
2. Confirm backend service availability and connectivity

🧩 ROOT-CAUSE HYPOTHESES
----------------------------------------
1. Invalid configuration may be causing the application/runtime failure.
2. The runtime failure may be preventing correct backend service (connection refused) communication, making the connection errors downstream symptoms.
3. Investigate whether backend service unavailability and the local application failure are separate but concurrent issues.

🛠️ GENERAL ANALYSIS
----------------------------------------
Detected Command: python app.py
Detected Subsystem: python
Signal Families: error, connection_refused, timeout
```

### Supported Family Combinations

Mirrorcore handles these key combinations:

- **Application/runtime + Network/connectivity**: Runtime failures causing network issues
- **Configuration + Application/runtime**: Config errors preventing startup
- **Configuration + Network/connectivity**: Misconfig blocking connections
- **Permissions + Application/runtime**: Access issues blocking execution
- **Docker/container + Application/runtime**: Container crashes from app failures
- **Service/systemd + Network/connectivity**: Service failures affecting network

### Complex Case Handling

For 3+ signal families, Mirrorcore uses specialized templates:

```python
# config + runtime + network (complex case)
("config", "runtime", "network"): [
    "Configuration errors may be causing both runtime failures and network connectivity problems.",
    "Invalid configuration could be preventing proper application startup and backend communication.",
    "The runtime failure may be primary, with network issues as downstream symptoms of misconfiguration."
]
```

### Integration with Escalation

Root-cause hypotheses integrate with Step 13 escalation logic:
- **Ambiguous inputs** trigger both hypotheses and escalation guidance
- **Multiple hypotheses** indicate complex issues requiring investigation
- **Grounded content** provides specific investigation starting points
- **Deterministic output** ensures consistent, explainable guidance

### Benefits

1. **Structured Investigation**: Provides clear causal chains to follow
2. **Grounded Analysis**: Uses observed content for relevant hypotheses
3. **Deterministic**: Same input always produces same hypotheses
4. **Explainable**: Clear logic for why each hypothesis was generated
5. **Lightweight**: No LLMs or heavy dependencies required
6. **Local-First**: All processing happens locally

### Limitations

- **Template-Based**: Limited to predefined combinations
- **Conservative**: Uses cautious phrasing ("may be", "could be")
- **Focused**: Designed for common troubleshooting scenarios
- **Deterministic**: No probabilistic reasoning or uncertainty quantification

## Outcome-Weighted Root-Cause Hypothesis Ranking

Mirrorcore ranks root-cause hypotheses deterministically using historical outcomes and structural relevance to prioritize investigation paths for ambiguous issues.

### When Ranking is Applied

Hypothesis ranking is activated in these scenarios:
- **No confident incident match** due to high ambiguity
- **Multiple signal families** detected (2+ families)
- **Multi-signal diagnostic mode** is triggered
- **Historical data available** for relevant categories

### Structured Hypothesis Object

Each hypothesis is represented as a structured object:

```python
@dataclass
class RootCauseHypothesis:
    text: str                    # Hypothesis description
    related_families: List[str]   # Associated signal families
    related_subsystems: List[str] # Associated subsystems
    category: str                # Hypothesis category (config_runtime, etc.)
    score: float                 # Ranking score (0.0-1.0)
    reason: str                  # Explanation of ranking
```

### Deterministic Scoring Formula

Hypotheses are scored using a transparent formula:

```python
score = 
    family_overlap_weight * overlap_count
  + subsystem_match_weight
  + historical_success_weight * success_count
  - historical_failure_weight * failure_count
  + recency_bonus
```

**Scoring Weights**:
- **Family Overlap**: 0.4 (signal family overlap with current input)
- **Subsystem Match**: 0.2 (matches detected subsystem)
- **Historical Success**: 0.3 (past successful resolutions)
- **Historical Failure**: 0.2 (past failed attempts)
- **Recency Bonus**: 0.1 (recent successful resolutions)

### Historical Support Retrieval

Mirrorcore retrieves historical support data from the local database:

```python
# Map signal families to fix patterns
family_fix_patterns = {
    'config': ['config', 'configuration', 'settings', 'environment'],
    'runtime': ['restart', 'reload', 'reboot', 'debug', 'fix'],
    'network': ['connection', 'network', 'backend', 'service', 'port'],
    'permissions': ['permission', 'access', 'chmod', 'chown', 'user'],
    'docker': ['docker', 'container', 'image', 'build', 'run'],
    'service': ['service', 'systemctl', 'daemon', 'start', 'stop']
}

# Categorize historical incidents by pattern matching
for incident in incidents:
    solution_lower = incident.get('solution', '').lower()
    success = incident.get('success', False)
    
    # Match to category based on solution patterns
    for category, patterns in family_fix_patterns.items():
        if any(pattern in solution_lower for pattern in patterns):
            category_stats[category]['success_count'] += success
            category_stats[category]['failure_count'] += not success
```

### Hypothesis Categories

Supported hypothesis categories for ranking:

- **config_runtime**: Configuration errors causing runtime failures
- **runtime_network**: Runtime issues preventing network communication
- **config_network**: Misconfiguration blocking network connections
- **permissions_runtime**: Permission issues blocking execution
- **docker_runtime**: Container crashes from application failures
- **service_network**: Service failures affecting network
- **concurrent_independent**: Multiple unrelated issues

### Ranked Output Format

Instead of a flat list, Mirrorcore shows ranked hypotheses:

```bash
🧩 ROOT-CAUSE HYPOTHESES
----------------------------------------
Most likely hypothesis:
1. Invalid configuration may be causing the application/runtime failure.
   Reason: strong runtime_config overlap; matches python subsystem; similar sessions resolved by runtime_config correction.

Other plausible hypotheses:
2. The application may be failing before backend communication.
   Reason: matches python subsystem; runtime + network signals are present in the current log.

3. Backend service unavailability may be a concurrent issue.
   Reason: runtime + network signals are present in the current log; historical support weaker for runtime_network.
```

### Ranking Example Process

**Input**: `python traceback + invalid configuration + connection refused + timeout`

**Step 1**: Generate basic hypotheses
1. Configuration errors may be causing the application/runtime failure
2. The application may be failing before or during backend communication
3. Backend service unavailability may be a concurrent issue

**Step 2**: Calculate scores
- **Hypothesis 1** (config_runtime):
  - Family overlap: 2/3 families match → 0.8
  - Subsystem match: python → +0.2
  - Historical success: 3 similar successes → +0.18
  - Historical failure: 1 similar failure → -0.04
  - **Total score: 1.14 → normalized to 1.0**

- **Hypothesis 2** (runtime_network):
  - Family overlap: 2/3 families match → 0.8
  - Subsystem match: python → +0.2
  - Historical success: 1 similar success → +0.06
  - Historical failure: 2 similar failures → -0.08
  - **Total score: 0.98**

- **Hypothesis 3** (runtime_network):
  - Family overlap: 2/3 families match → 0.8
  - Subsystem match: python → +0.2
  - Historical success: 0 similar successes → +0.0
  - Historical failure: 3 similar failures → -0.12
  - **Total score: 0.88**

**Step 3**: Sort and display with reasons

### Conservative Ranking

The ranking system maintains conservative phrasing:
- **Hypotheses**: Still use "may be", "could be", "investigate whether"
- **Reasons**: Explain why ranked where, but don't overstate certainty
- **Scores**: Internal ranking, not displayed to avoid false confidence
- **Transparency**: Clear explanation of ranking factors

### Integration with Existing Features

- **Multi-Signal Diagnostic Mode**: Ranking enhances existing layer analysis
- **Escalation Logic**: Ranked hypotheses inform escalation decisions
- **Learning System**: Historical outcomes improve future ranking accuracy
- **Clear Cases**: Ranking only activates for ambiguous cases

### Benefits

1. **Prioritized Investigation**: Users know which hypothesis to investigate first
2. **Historical Context**: Ranking considers what worked in similar situations
3. **Transparent Logic**: Clear reasons explain ranking decisions
4. **Deterministic**: Same input always produces same ranking
5. **Conservative**: Maintains cautious approach to uncertainty
6. **Local-First**: All ranking logic runs locally with no external dependencies

### Limitations

- **Template-Based**: Limited to predefined hypothesis categories
- **Historical Dependency**: Ranking quality depends on available historical data
- **Simplified Scoring**: Linear formula without complex probabilistic modeling
- **Pattern Matching**: Historical categorization based on keyword matching
- **No Confidence Intervals**: Scores are relative rankings, not probability estimates

## Hypothesis-Driven Diagnostic Command Suggestions

Mirrorcore suggests deterministic next diagnostic commands or checks tailored to the most likely hypothesis, guiding investigation by recommending concrete actions that gather the most relevant evidence.

### When Diagnostic Commands Are Suggested

Diagnostic command suggestions appear in these scenarios:
- **No confident incident match** due to high ambiguity
- **Multiple signal families** detected (2+ families)
- **Multi-signal diagnostic mode** is triggered
- **Ranked hypotheses** are generated for the top hypothesis

### Diagnostic Command Template System

Mirrorcore uses a deterministic template system mapping hypothesis categories to suggested diagnostic commands:

```python
diagnostic_templates = {
    # config_runtime
    "config_runtime": [
        {
            "description": "Inspect configuration file for syntax errors",
            "command": "cat config.yaml | grep -n -A5 -B5 error",
            "subsystem_specific": {
                "python": "python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"",
                "node": "node -e \"require('./config.json')\"",
                "docker": "docker exec <container> cat /app/config.yaml"
            }
        },
        {
            "description": "Validate environment variables",
            "command": "env | grep -E '(CONFIG|APP|DATABASE)' | sort",
            "subsystem_specific": {
                "python": "python -c \"import os; print({k:v for k,v in os.environ.items() if 'CONFIG' in k or 'APP' in k or 'DB' in k})\"",
                "docker": "docker exec <container> env | grep -E '(CONFIG|APP|DATABASE)'"
            }
        }
    ],
    # runtime_network
    "runtime_network": [
        {
            "description": "Inspect the full traceback",
            "command": "cat application.log | grep -A20 Traceback",
            "subsystem_specific": {
                "python": "python -m traceback -l application.log",
                "node": "node --trace-warnings app.js",
                "docker": "docker logs <container> | grep -A20 Traceback"
            }
        },
        {
            "description": "Run the application in verbose/debug mode",
            "command": "python app.py --debug",
            "subsystem_specific": {
                "python": "python app.py --debug -v",
                "node": "DEBUG=* node app.js",
                "docker": "docker exec <container> python app.py --debug"
            }
        }
    ]
}
```

### Supported Hypothesis Categories

Each hypothesis category has tailored diagnostic commands:

- **config_runtime**: Configuration validation, environment variables, syntax checking
- **runtime_network**: Traceback inspection, debug mode, connectivity testing
- **config_network**: Network configuration, endpoint validation, DNS resolution
- **permissions_runtime**: File permissions, ownership checks, user context
- **docker_runtime**: Container logs, restart status, configuration inspection
- **service_network**: Service status, log inspection, port verification
- **concurrent_failures**: System resources, connectivity, system logs

### Subsystem-Aware Command Adaptation

Commands adapt to detected subsystems when possible:

**Python subsystem**:
- `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
- `python -m traceback -l application.log`
- `python app.py --debug -v`

**Node.js subsystem**:
- `node -e "require('./config.json')"`
- `node --trace-warnings app.js`
- `DEBUG=* node app.js`

**Docker subsystem**:
- `docker exec <container> cat /app/config.yaml`
- `docker logs <container> | grep -A20 Traceback`
- `docker exec <container> python app.py --debug`

**Systemd subsystem**:
- `systemctl status python-app.service`
- `journalctl -u python-app.service -n 20 --no-pager`
- `netstat -tulnp | grep :8080`

### Output Format

Diagnostic commands appear in a dedicated section:

```bash
🧩 ROOT-CAUSE HYPOTHESES
----------------------------------------
Most likely hypothesis:
1. The application may be failing before or during backend communication.
   Reason: strong runtime_network overlap; matches python subsystem

🔎 NEXT DIAGNOSTIC STEPS
----------------------------------------
Recommended checks for this hypothesis:

1. Inspect the full traceback
   Command: python -m traceback -l application.log

2. Run the application in verbose/debug mode
   Command: python app.py --debug -v

3. Check backend connectivity directly
   Command: python -c "import requests; print(requests.get('http://localhost:8080/health').status_code)"

Other useful checks:
- Verify backend service status
  Command: systemctl status python-app.service
```

### Command Selection Logic

Commands are chosen deterministically based on:

1. **Hypothesis Category**: Primary template selection
2. **Detected Subsystem**: Command adaptation when available
3. **Signal Families**: Template refinement for specific signals
4. **Priority Order**: First 3 commands are primary, next 2 are additional

### Deterministic Selection Process

```python
# Get diagnostic templates for the hypothesis category
category = top_hypothesis.category
templates = diagnostic_templates.get(category, [])

# Generate primary commands (first 3)
primary_commands = []
for template in templates[:3]:
    command = adapt_command_to_subsystem(template, detected_subsystem)
    primary_commands.append({
        "description": template["description"],
        "command": command
    })

# Generate additional commands (next 2)
additional_commands = []
for template in templates[3:5]:
    command = adapt_command_to_subsystem(template, detected_subsystem)
    additional_commands.append({
        "description": template["description"],
        "command": command
    })
```

### Safety and Execution Policy

**Important**: Mirrorcore never executes commands automatically. It only recommends them.

- **No Automatic Execution**: Commands are displayed for user consideration only
- **User Control**: Users decide which commands to run
- **Safety First**: All commands are read-only or non-destructive
- **Educational Purpose**: Commands help users learn diagnostic techniques

### Integration with Existing Features

- **Multi-Signal Diagnostic Mode**: Commands enhance hypothesis analysis
- **Ranked Hypotheses**: Commands tailored to top-ranked hypothesis
- **Subsystem Detection**: Commands adapt to detected runtime environment
- **Clear Cases**: Commands only appear for ambiguous multi-signal cases

### Benefits

1. **Guided Investigation**: Users get concrete next steps
2. **Evidence Gathering**: Commands target specific hypothesis validation
3. **Subsystem Awareness**: Commands adapt to detected environment
4. **Educational Value**: Users learn relevant diagnostic techniques
5. **Deterministic**: Same hypothesis always produces same commands
6. **Safe**: No automatic execution, user maintains control

### Limitations

- **Template-Based**: Limited to predefined command sets
- **Subsystem Coverage**: Only supports common subsystems (Python, Node, Docker, Systemd)
- **Environment Specific**: Some commands may need adaptation for specific environments
- **Read-Only Focus**: Commands are primarily diagnostic, not corrective
- **No Context Awareness**: Commands don't consider specific configuration details

## Evidence-Aware Reanalysis and Hypothesis Updating

Mirrorcore supports iterative troubleshooting by accepting follow-up diagnostic evidence and deterministically re-ranking hypotheses based on new information.

### Follow-Up Evidence Workflow

Mirrorcore enables an iterative troubleshooting loop:

1. **Initial Analysis**: Run `mirrorcore analyze-log` with ambiguous multi-signal issues
2. **Diagnostic Suggestions**: Get specific commands to gather evidence
3. **Evidence Collection**: User runs suggested commands and pastes output
4. **Evidence Update**: Run `mirrorcore analyze-followup` with new evidence
5. **Hypothesis Re-ranking**: Mirrorcore updates hypotheses based on evidence

### When to Use Follow-Up Analysis

Use follow-up analysis when:
- **Initial analysis was ambiguous** with multiple hypotheses
- **Diagnostic commands were suggested** in the initial analysis
- **New evidence is available** from running diagnostic commands
- **Investigation path needs refinement** based on concrete evidence

### Evidence Patterns Detected

Mirrorcore parses follow-up evidence using deterministic regex patterns:

**Connection Evidence:**
- `connection_confirmed`: "connected", "200 OK", "Connection successful"
- `connection_refused`: "Connection refused", "Failed to connect", "curl: (7)"
- `timeout`: "timeout", "timed out", "Connection timed out"

**Service Evidence:**
- `service_active`: "Active: active", "running", "UP", "LISTEN"
- `service_inactive`: "Active: inactive", "stopped", "failed", "DOWN"

**Port Evidence:**
- `port_listening`: "LISTEN", "listening", "Port open"
- `port_not_listening`: "No such file", "Address already in use", "Port closed"

**Configuration Evidence:**
- `config_valid`: "config valid", "syntax OK", "Validation passed"
- `config_invalid`: "config invalid", "syntax error", "Validation failed"

**Permission Evidence:**
- `permission_granted`: "permission granted", "Access granted", "rw-"
- `permission_denied`: "Permission denied", "Access denied", "read-only"

**Module Evidence:**
- `module_present`: "module found", "installed", "loaded"
- `module_missing`: "module not found", "No module named", "cannot import"

### Evidence-to-Hypothesis Effects

Mirrorcore uses deterministic rules to update hypothesis scores:

```python
evidence_effects = {
    'connection_confirmed': {
        'boost': ['config_network'],
        'penalize': ['runtime_network', 'service_network']
    },
    'connection_refused': {
        'boost': ['runtime_network', 'service_network'],
        'penalize': ['config_network']
    },
    'service_active': {
        'boost': ['config_runtime', 'permissions_runtime'],
        'penalize': ['service_network']
    },
    'config_invalid': {
        'boost': ['config_runtime'],
        'penalize': ['runtime_network', 'permissions_runtime']
    }
    # ... more rules
}
```

### Score Update Formula

```python
updated_score = base_score + evidence_support_bonus - evidence_conflict_penalty

# Where:
# evidence_support_bonus = +0.2 per supporting evidence
# evidence_conflict_penalty = -0.3 per conflicting evidence
# Final score is clamped to [0.0, 1.0]
```

### Follow-Up Analysis Example

**Initial Analysis:**
```bash
🧩 ROOT-CAUSE HYPOTHESES
----------------------------------------
Most likely hypothesis:
1. The application may be failing before or during backend communication.
   Reason: strong runtime_network overlap; matches python subsystem

🔎 NEXT DIAGNOSTIC STEPS
----------------------------------------
1. Check backend connectivity directly
   Command: curl localhost:8080/health
```

**User runs diagnostic command:**
```bash
$ curl localhost:8080/health
curl: (7) Failed to connect to localhost port 8080: Connection refused
```

**Follow-Up Analysis:**
```bash
$ mirrorcore analyze-followup
🔄 EVIDENCE-BASED REANALYSIS
==================================================
Paste follow-up command output or additional logs
^D

📋 Using analysis session from: 2026-03-14T19:30:15.123456
Original subsystem: python
Original top hypothesis: runtime_network

🔍 Detected evidence signals: connection_refused

🔄 EVIDENCE UPDATE
----------------------------------------
New evidence supports:
- backend connectivity failure

New evidence weakens:
- runtime-first failure hypothesis

🧩 UPDATED ROOT-CAUSE HYPOTHESES
----------------------------------------
Most likely hypothesis:
1. Backend service unavailability may be the primary issue.
   Reason: connection_refused supports runtime_network; connection_refused weakens config_network

Other plausible hypotheses:
2. The application may still have runtime issues, but current evidence more strongly supports backend failure.

🔎 UPDATED NEXT DIAGNOSTIC STEPS
----------------------------------------
1. Check whether the backend service is running
   Command: systemctl status backend-service
2. Check listening ports
   Command: ss -tulnp | grep :8080
3. Inspect backend logs
   Command: journalctl -u backend-service -n 50

✅ Follow-up analysis complete.
Session abc123-def456 updated with new evidence.
```

### Session Management

Mirrorcore automatically stores analysis sessions for multi-signal diagnostic cases:

```python
# Session data stored
session_data = {
    'detected_subsystem': 'python',
    'signal_families': ['error', 'connection_refused', 'timeout'],
    'original_hypotheses': [
        {
            'text': 'The application may be failing before or during backend communication.',
            'category': 'runtime_network',
            'score': 0.95,
            'reason': 'strong runtime_network overlap; matches python subsystem'
        }
        # ... more hypotheses
    ],
    'top_hypothesis_category': 'runtime_network',
    'suggested_commands': [...],
    'analysis_summary': 'traceback: backend connectivity issues'
}
```

### Deterministic Evidence Rules

All evidence interpretation follows strict deterministic rules:

- **No LLM inference**: Only regex and keyword matching
- **Transparent logic**: Clear evidence-to-hypothesis mapping
- **Consistent results**: Same evidence always produces same updates
- **Explainable reasoning**: Clear support/weakens explanations

### Integration with Existing Features

- **Ranked Hypotheses**: Evidence updates work with ranked hypothesis system
- **Diagnostic Commands**: Updated commands reflect new top hypothesis
- **Session Storage**: Automatic session management for multi-signal cases
- **Local-First**: All evidence processing happens locally

### Benefits

1. **Iterative Investigation**: Refine understanding as evidence accumulates
2. **Evidence-Based Decisions**: Concrete data drives hypothesis prioritization
3. **Adaptive Guidance**: Diagnostic steps update based on findings
4. **Transparent Reasoning**: Clear explanation of how evidence affects hypotheses
5. **Deterministic Results**: Same evidence always produces same updates
6. **Educational Value**: Users learn evidence-driven troubleshooting

### Limitations

- **Evidence Pattern Coverage**: Limited to predefined regex patterns
- **Binary Evidence**: Treats evidence as present/absent, no nuanced interpretation
- **No Contextual Understanding**: Doesn't understand complex multi-line evidence
- **Session Isolation**: Follow-up only works with most recent session
- **Evidence Quality**: Depends on user providing relevant command output

## Investigation State Tracking and Hypothesis Resolution

Mirrorcore tracks investigation progress to avoid repeating diagnostic steps and eliminates hypotheses when strongly contradicted by evidence.

### Investigation Step Lifecycle

Each diagnostic suggestion becomes a tracked investigation step:

1. **Step Creation**: Initial analysis generates pending investigation steps
2. **Step Execution**: User runs suggested commands and provides output
3. **Step Completion**: Evidence is associated and step marked as completed
4. **Step Deduplication**: Completed steps are excluded from future suggestions
5. **Investigation Progress**: Status shows completed vs remaining steps

### Step Storage Structure

```python
investigation_step = {
    'id': 'unique-step-uuid',
    'step_id': 1,  # Sequential order
    'description': 'Check backend connectivity directly',
    'command': 'curl localhost:8080/health',
    'status': 'pending',  # pending | completed
    'evidence_generated': ['connection_refused', 'service_inactive'],
    'timestamp_completed': '2026-03-14T19:30:15.123456'
}
```

### Step Completion Detection

Mirrorcore uses deterministic matching to associate evidence with steps:

1. **Exact Command Match**: `curl localhost:8080/health` matches stored command exactly
2. **Command Similarity**: `curl -v localhost:8080/health` contains stored command
3. **Sequential Fallback**: First pending step if no match found

### Hypothesis Elimination Rules

Strong evidence can eliminate hypotheses entirely:

```python
elimination_rules = {
    'connection_confirmed': {
        'eliminate': ['runtime_network', 'service_network'],
        'reason': 'backend connectivity confirmed successful'
    },
    'config_valid': {
        'eliminate': ['config_runtime'],
        'reason': 'configuration validation passed'
    },
    'service_active': {
        'eliminate': ['service_network'],
        'reason': 'backend service is running'
    },
    'module_present': {
        'eliminate': ['permissions_runtime'],
        'reason': 'required module is available'
    }
}
```

### Investigation Status Display

Follow-up analysis shows comprehensive investigation progress:

```bash
📋 INVESTIGATION STATUS
----------------------------------------
Eliminated hypotheses:
✗ runtime_network
  Reason: backend connectivity confirmed successful

Completed steps:
✓ Check backend connectivity directly
  Command: curl localhost:8080/health

Pending steps:
- Inspect traceback
  Command: python app.py --debug -v

Next recommended step:
- Inspect traceback
  Command: python app.py --debug -v
```

### Updated Hypothesis Structure

Hypotheses now track elimination and evidence:

```python
@dataclass
class RootCauseHypothesis:
    # ... existing fields ...
    eliminated: bool = False
    elimination_reason: Optional[str] = None
    supporting_evidence: List[str] = None
    conflicting_evidence: List[str] = None
```

### Diagnostic Deduplication

Mirrorcore avoids repeating completed diagnostic steps:

1. **Track Completed Commands**: Maintain set of executed commands
2. **Filter New Suggestions**: Exclude completed commands from recommendations
3. **Generate New Steps**: If all steps completed, suggest escalation

### Investigation Workflow Example

**Initial Analysis:**
```bash
🧩 ROOT-CAUSE HYPOTHESES
Most likely hypothesis:
1. The application may be failing before or during backend communication.

🔎 NEXT DIAGNOSTIC STEPS
1. Check backend connectivity directly
   Command: curl localhost:8080/health
2. Inspect traceback
   Command: python app.py --debug -v

💾 Analysis session stored with ID: abc123
```

**After First Follow-Up:**
```bash
$ mirrorcore analyze-followup
🔄 EVIDENCE UPDATE
New evidence weakens:
- backend connectivity failure

📋 INVESTIGATION STATUS
----------------------------------------
Completed steps:
✓ Check backend connectivity directly
  Command: curl localhost:8080/health

Pending steps:
- Inspect traceback
  Command: python app.py --debug -v

Next recommended step:
- Inspect traceback
  Command: python app.py --debug -v
```

**After Contradictory Evidence:**
```bash
$ mirrorcore analyze-followup
🔄 EVIDENCE UPDATE
New evidence supports:
- backend connectivity confirmed

📋 INVESTIGATION STATUS
----------------------------------------
Eliminated hypotheses:
✗ runtime_network
  Reason: backend connectivity confirmed successful

🧩 UPDATED ROOT-CAUSE HYPOTHESES
----------------------------------------
Most likely hypothesis:
1. Configuration or runtime failure may now be more likely.
```

### Investigation Completion

When all diagnostic steps are completed:

```bash
🔎 INVESTIGATION PROGRESS
----------------------------------------
Completed 3/3 investigation steps.
All suggested diagnostic commands have been executed.
Consider reviewing all evidence or escalating investigation.
```

### Benefits

1. **No Repetition**: Completed diagnostic steps are never suggested again
2. **Progressive Elimination**: Contradicted hypotheses are removed from consideration
3. **Clear Status**: Users see exactly what's completed vs. pending
4. **Evidence Association**: Each step tracks what evidence it generated
5. **Deterministic Matching**: Consistent step completion detection
6. **Investigation Focus**: Next steps always prioritize remaining unknowns

### Integration with Existing Features

- **Evidence-Aware Reanalysis**: Step tracking works with evidence updates
- **Hypothesis Ranking**: Eliminated hypotheses sorted to bottom
- **Diagnostic Commands**: Deduplication prevents repetition
- **Session Management**: Investigation steps stored with analysis sessions
- **Local-First**: All tracking and elimination happens locally

## Privacy

- All data stored locally in SQLite database
- No external data transmission required
- Learning signals and persona updates are traceable and reversible
- Full control over data retention and deletion
- Assessment responses and persona traits stored locally

## Development

See `docs/` for detailed documentation on architecture, memory models, and development guidelines.

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific intake tests
python -m pytest tests/test_intake.py
```

## License

MIT License - see LICENSE file for details.
