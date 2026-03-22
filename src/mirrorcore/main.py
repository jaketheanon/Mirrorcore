#!/usr/bin/env python3
"""
Mirrorcore Main Entry Point

This module provides the main CLI interface for Mirrorcore.
"""

import sys
import argparse
import uuid
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .config import get_config
from .db.store import DatabaseStore
from .intake.scenario_engine import ScenarioEngine
from .intake.profiler import Profiler

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.shortcuts import confirm
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

__version__ = "0.1.0"
__description__ = "A local-first reasoning assistant for personalized troubleshooting and decision support"


def _infer_strategy_family(category: Optional[str]) -> str:
    """Map a hypothesis category to an investigation strategy family."""
    cat = (category or "").lower()
    if "network" in cat or "connect" in cat or "timeout" in cat:
        return "connectivity"
    if "service" in cat or "systemd" in cat:
        return "service"
    if "permission" in cat or "access" in cat:
        return "permissions"
    return "configuration"


def create_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="mirrorcore",
        description=__description__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mirrorcore start                              # Guided entry menu
  mirrorcore                                    # Start interactive session
  mirrorcore debug "git push failed"            # Debug a command
  mirrorcore decide "use PostgreSQL or SQLite" # Get decision help
  mirrorcore analyze-log                        # Analyze logs interactively
  mirrorcore analyze-followup                   # Update analysis with new evidence
  mirrorcore profile                            # View your profile
  mirrorcore interview                          # Run a decision interview
  mirrorcore respond-like-me                    # Likely-you answer from saved memory
  mirrorcore ask "Should I buy this now?"       # Route plain-language to the right flow
        """
    )
    
    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"Mirrorcore {__version__}"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Suppress non-error output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Guided entry (Phase 30)
    subparsers.add_parser(
        "start",
        help="Guided onboarding and quick access to main features",
    )

    # Interactive mode (default)
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive session"
    )
    
    # Debug command
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debug terminal issues"
    )
    debug_parser.add_argument(
        "problem",
        help="Problem description or command that failed"
    )
    debug_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read problem from stdin"
    )
    
    # Debug outcome command
    debug_outcome_parser = subparsers.add_parser(
        "debug-outcome",
        help="Record outcome of debugging attempt"
    )
    
    # Debug history command
    debug_history_parser = subparsers.add_parser(
        "debug-history",
        help="View ranked fix history for incident types"
    )
    debug_history_parser.add_argument(
        "incident_type",
        nargs="?",
        help="Specific incident type to view history for"
    )
    
    # Analyze log command
    analyze_log_parser = subparsers.add_parser(
        "analyze-log",
        help="Analyze raw terminal output, logs, or stack traces"
    )
    
    # Analyze followup command
    analyze_followup_parser = subparsers.add_parser(
        "analyze-followup",
        help="Update previous analysis with new evidence"
    )
    
    # Decide command
    decide_parser = subparsers.add_parser(
        "decide",
        help="Get decision assistance"
    )
    decide_parser.add_argument(
        "decision",
        help="Decision context or question"
    )
    
    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage persona profile"
    )
    profile_parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze recent patterns"
    )
    profile_parser.add_argument(
        "--update",
        help="Update specific profile aspect"
    )
    
    # Assessment command
    assess_parser = subparsers.add_parser(
        "assess",
        help="Run initial assessment"
    )
    assess_parser.add_argument(
        "--continue",
        action="store_true",
        help="Continue previous assessment"
    )
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize persona assessment"
    )
    
    # Memory command
    memory_parser = subparsers.add_parser(
        "memory",
        help="Memory management"
    )
    memory_parser.add_argument(
        "action",
        choices=["search", "stats", "export", "import", "drift"],
        help="Memory action"
    )
    
    # Interview command (Phase 26)
    interview_parser = subparsers.add_parser(
        "interview",
        help="Run a structured decision interview"
    )

    # Style calibration command (Phase 28)
    calibrate_style_parser = subparsers.add_parser(
        "calibrate-style",
        help="Run a short style/persona calibration session"
    )

    respond_like_me_parser = subparsers.add_parser(
        "respond-like-me",
        help="Generate a likely-you response from saved decision and style memory",
    )
    respond_like_me_parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
        help="Scenario or question (omit to type it when prompted)",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Route one plain-language request into the right MirrorCore flow",
    )
    ask_parser.add_argument(
        "query",
        nargs="*",
        help="What you want help with (omit to type when prompted)",
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show system status"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management"
    )
    config_parser.add_argument(
        "--get",
        help="Get configuration value"
    )
    config_parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set configuration value"
    )
    
    return parser


def handle_ask(args):
    """Single-input router (Phase 31) — deterministic intent → existing handlers."""
    from types import SimpleNamespace

    from .router import (
        PROFILE_STYLE,
        resolve_full_route,
        routing_feedback,
    )

    parts = getattr(args, "query", None) or []
    text = " ".join(parts).strip()

    if not text:
        if PROMPT_TOOLKIT_AVAILABLE:
            try:
                text = prompt("What do you want help with? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            try:
                text = input("What do you want help with? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return

    if not text:
        print("No input — nothing to route.")
        return

    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)

    def read_choice(prompt_text: str) -> str:
        try:
            return input(prompt_text)
        except EOFError:
            return ""

    classification, route = resolve_full_route(text, read_choice)

    if classification.weak_input:
        print(
            "That's a bit vague — I'm not sure which track fits. "
            "Opening the main menu so you can choose."
        )
    else:
        print(routing_feedback(route))
    print()

    if route.category == "onboarding_or_help":
        handle_start(args)
        return

    if route.category == "decision_help":
        from .decision.routed_clarification import run_routed_decision_guidance

        guidance = run_routed_decision_guidance(
            initial_text=text,
            read_line=read_choice,
            db_store=db_store,
        )
        print()
        print(guidance)
        return

    if route.category == "personal_response":
        handle_respond_like_me(SimpleNamespace(scenario=text))
        return

    if route.category == "debug_help":
        handle_analyze_log(args)
        return

    if route.category == "profile_building":
        if route.profile_target == PROFILE_STYLE:
            handle_calibrate_style(args)
        else:
            handle_interview(args)
        return


def handle_start(args):
    """Unified guided entry — routes into existing command handlers."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .onboarding import default_read_choice, run_start_menu

    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)

    run_start_menu(
        db_store,
        handle_interview=handle_interview,
        handle_calibrate_style=handle_calibrate_style,
        handle_respond_like_me=handle_respond_like_me,
        handle_analyze_log=handle_analyze_log,
        handle_analyze_followup=handle_analyze_followup,
        args_namespace=args,
        read_choice=default_read_choice,
    )


def handle_interactive(args):
    """Handle interactive Mirrorcore session."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .memory.extractor import MemoryExtractor
    from .memory.updater import MemoryUpdater
    from .memory.retrieval import MemoryRetrieval
    import uuid
    import json
    from datetime import datetime
    
    print("Starting interactive Mirrorcore session...")
    print("Type 'help' for available commands or 'exit' to quit")
    
    # Initialize components
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    extractor = MemoryExtractor()
    updater = MemoryUpdater()
    retrieval = MemoryRetrieval()
    
    # Initialize reasoning response engine
    try:
        from .reasoning.response_engine import ReasoningResponseEngine
        reasoning_engine = ReasoningResponseEngine()
    except ImportError:
        reasoning_engine = None
    
    # Initialize prompt toolkit components
    if PROMPT_TOOLKIT_AVAILABLE:
        from prompt_toolkit import prompt
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.shortcuts import confirm
    else:
        # Fallback to basic input handling
        def prompt_input(text):
            return input(text)
        
        def simple_history():
            return []
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    conversation = []
    session_start = datetime.utcnow().isoformat()
    
    # Load current persona for response styling
    try:
        current_persona = db_store.load_persona_profile()
    except Exception:
        current_persona = {}
    
    print(f"\nSession ID: {session_id[:8]}...")
    print("Mirrorcore will learn from your conversation patterns.")
    print("Type 'exit' to end session.\n")
    
    if PROMPT_TOOLKIT_AVAILABLE:
        # Create interactive session with prompt_toolkit
        history = InMemoryHistory()
        
        def get_input():
            user_input = prompt("You: ", history=history)
            return user_input.strip() if user_input else ""
        
        def print_response(text):
            print(f"Mirrorcore: {text}")
        
        # Main interaction loop
        while True:
            try:
                user_input = get_input()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if user_input.lower() == 'help':
                print("Available commands:")
                print("  help  - Show this help")
                print("  exit  - End session")
                print("  status - Show learning status")
                print("\nJust chat normally - Mirrorcore learns from patterns!")
                continue
            
            # Add user message to conversation
            conversation.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Generate response using reasoning engine or fallback
            if reasoning_engine:
                response_result = reasoning_engine.generate_response(
                    user_input=user_input,
                    persona_profile=current_persona,
                    conversation_history=conversation,
                    session_id=session_id,
                    db_store=db_store
                )
                
                # Print the response
                print_response(response_result.response)

                # Show escalation guidance if present (print once per turn)
                if response_result.escalation_guidance:
                    print(response_result.escalation_guidance)
                
                # Handle assisted outcome capture for troubleshooting responses
                if response_result.should_prompt_outcome and response_result.incident_id:
                    outcome_prompt = reasoning_engine.prompt_assisted_outcome_capture(
                        response_result.incident_id,
                        response_result.suggested_fixes or []
                    )
                    print(outcome_prompt)
                    
                    # Handle user response
                    try:
                        user_response = input().strip().lower()
                        if user_response in ['y', 'yes', 'yeah', 'yep']:
                            _collect_assisted_outcome(db_store, response_result.incident_id, response_result.suggested_fixes or [])
                        else:
                            print("No problem. You can use 'mirrorcore debug-outcome' anytime to record outcomes.")
                    except (EOFError, KeyboardInterrupt):
                        print("\nNo problem. You can use 'mirrorcore debug-outcome' anytime to record outcomes.")
            else:
                # Fallback to simple echo if reasoning engine unavailable
                response = f"I understand you said: '{user_input}'. I'm processing this and learning from patterns."
                print_response(response)
            
            # Add assistant response to conversation
            if reasoning_engine:
                conversation.append({
                    "role": "assistant", 
                    "content": response_result.response,
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                conversation.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    else:
        # Fallback to basic input handling without prompt_toolkit
        print("Starting interactive Mirrorcore session...")
        print("Type 'help' for available commands or 'exit' to quit")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        conversation = []
        session_start = datetime.utcnow().isoformat()
        
        # Initialize components
        db_path = Path("data") / "mirrorcore.db"
        db_store = DatabaseStore(db_path)
        extractor = MemoryExtractor()
        updater = MemoryUpdater()
        retrieval = MemoryRetrieval()
        
        # Initialize reasoning response engine
        try:
            from .reasoning.response_engine import ReasoningResponseEngine
            reasoning_engine = ReasoningResponseEngine()
        except ImportError:
            reasoning_engine = None
        
        # Load current persona for response styling
        try:
            current_persona = db_store.load_persona_profile()
        except Exception:
            current_persona = {}
        
        print(f"\nSession ID: {session_id[:8]}...")
        print("Mirrorcore will learn from your conversation patterns.")
        print("Type 'exit' to end session.\n")
        
        try:
            while True:
                # Get user input
                user_input = prompt_input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                if user_input.lower() == 'help':
                    print("Available commands:")
                    print("  help  - Show this help")
                    print("  exit  - End session")
                    print("  status - Show learning status")
                    print("\nJust chat normally - Mirrorcore learns from patterns!")
                    continue
                
                # Add user message to conversation
                conversation.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Generate response using reasoning engine or fallback
                if reasoning_engine:
                    response = reasoning_engine.generate_response(
                        user_input=user_input,
                        persona_profile=current_persona,
                        conversation_history=conversation,
                        session_id=session_id,
                        db_store=db_store
                    )
                else:
                    # Fallback to simple echo if reasoning engine unavailable
                    response = f"I understand you said: '{user_input}'. I'm processing this and learning from patterns."
                
                print(f"Mirrorcore: {response}")
                
                # Add assistant response to conversation
                conversation.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        except KeyboardInterrupt:
            print("\nSession interrupted by user")
    
    # End of session - extract and store learning signals
    signals = []  # Initialize signals variable
    if len(conversation) > 2:  # Need some conversation to learn from
        print(f"\nProcessing session {session_id[:8]}... for learning signals...")
        
        # Extract signals from conversation
        signals = extractor.extract_conversation_signals(session_id, conversation)
        
        if signals:
            print(f"✓ Extracted {len(signals)} learning signals")
            
            # Store signals
            signal_ids = db_store.save_learning_signals(signals)
            print(f"✓ Stored {len(signal_ids)} learning signals")
            
            # Evaluate for persona drift
            current_persona = db_store.load_persona_profile()
            drift_evaluations = updater.evaluate_persona_drift(
                current_persona=current_persona,
                recent_signals=signals,
                memory_store=db_store,
                retrieval=retrieval
            )
            
            if drift_evaluations:
                print(f"✓ Evaluated {len(drift_evaluations)} persona drift evaluations")
        
                # Store drift evaluations for traceability
                for evaluation in drift_evaluations:
                    db_store.save_drift_evaluation(evaluation)
            
            # Legacy: evaluate for persona refinements
            refinements = updater.evaluate_learning_signals(
                [s.__dict__ for s in signals], current_persona, db_store
            )
            
            if refinements:
                print(f"✓ Generated {len(refinements)} persona refinement suggestions")
                
                # Apply conservative updates
                applied, suggested = updater.apply_conservative_updates(refinements, db_store)
                
                if applied > 0:
                    print(f"✓ Applied {applied} high-confidence refinements")
                if suggested > 0:
                    print(f"✓ Stored {suggested} refinement suggestions for review")
            else:
                print("✓ No clear persona refinements detected in this session")
        else:
            print("✓ No learning signals detected in this conversation")
    
    print(f"\nSession {session_id[:8]}... ended. Learning data saved.")
    print("Mirrorcore will use these insights to provide better personalized assistance.")
    
    # Save session log to database
    session_end = datetime.utcnow().isoformat()
    interaction_count = len(conversation)
    db_store.save_session_log(
        session_id=session_id,
        session_start=session_start,
        interaction_count=interaction_count,
        session_end=session_end,
        notes=f"Processed {len(signals) if signals else 0} learning signals"
    )
    print(f"✓ Session logged to database")


def handle_debug(args):
    """Handle terminal debugging command."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .terminal.parser import TerminalParser
    from .terminal.diagnostician import TerminalDiagnostician
    from .terminal.fixer import TerminalFixer
    
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    
    # Parse the command
    parser = TerminalParser()
    parsed_command = parser.parse(args.command)
    
    # Diagnose the issue
    diagnostician = TerminalDiagnostician()
    diagnosis = diagnostician.diagnose(parsed_command, {})
    
    # Generate fixes
    fixer = TerminalFixer()
    fixes = fixer.generate_fixes(diagnosis, {})
    
    # Display results
    print(f"Command: {parsed_command.command}")
    print(f"Arguments: {' '.join(parsed_command.arguments)}")
    print(f"Error Type: {diagnosis.issue_type}")
    print(f"Root Cause: {diagnosis.root_cause}")
    print(f"Suggested Fixes:")
    for i, fix in enumerate(fixes, 1):
        print(f"  {i}. {fix.description}")
        print(f"     Command: {fix.command}")
        print(f"     Confidence: {fix.confidence:.2f}")
    
    # Store the incident for learning
    if diagnosis.issue_type != "unknown":
        incident_id = db_store.save_terminal_incident(
            command=args.command,
            error_message=diagnosis.issue_type,
            root_cause=diagnosis.root_cause,
            solution=fixes[0].description if fixes else "No fixes available",
            success=False
        )
        print(f"\n✓ Incident stored with ID: {incident_id}")


def _collect_assisted_outcome(db_store, incident_id: str, suggested_fixes: List[str]):
    """Collect and store outcome information with user assistance."""
    print("\n" + "=" * 60)
    print("📝 RECORDING OUTCOME")
    print("=" * 60)
    
    # Get attempted fix
    attempted_fix = ""
    if suggested_fixes:
        print("Suggested fixes from this session:")
        for i, fix in enumerate(suggested_fixes, 1):
            print(f"  {i}. {fix}")
        print("  0. Other (specify your own)")
        
        while True:
            try:
                choice = input("\nWhich fix did you attempt? [0-{}] ".format(len(suggested_fixes))).strip()
                if choice == '0':
                    attempted_fix = input("Enter the fix you attempted: ").strip()
                    break
                elif choice.isdigit() and 1 <= int(choice) <= len(suggested_fixes):
                    attempted_fix = suggested_fixes[int(choice) - 1]
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        attempted_fix = input("What fix did you attempt? ").strip()
    
    if not attempted_fix:
        print("No fix specified. Cancelled.")
        return
    
    # Get result status
    print("\nResult status:")
    print("  1. success - Fix completely resolved the issue")
    print("  2. partial - Fix helped but issue persists")
    print("  3. failed - Fix didn't work or made things worse")
    print("  4. unknown - Not sure if it worked")
    
    result_mapping = {'1': 'success', '2': 'partial', '3': 'failed', '4': 'unknown'}
    result_status = ""
    
    while True:
        try:
            choice = input("\nWhat was the result? [1-4] ").strip()
            if choice in result_mapping:
                result_status = result_mapping[choice]
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
    
    # Get confirmed root cause (optional)
    confirmed_root_cause = input("\nConfirmed root cause (optional, press Enter to skip): ").strip()
    if not confirmed_root_cause:
        confirmed_root_cause = None
    
    # Get notes (optional)
    notes = input("Additional notes (optional, press Enter to skip): ").strip()
    if not notes:
        notes = None
    
    # Save the outcome
    try:
        outcome_id = db_store.save_terminal_fix_outcome(
            incident_id=incident_id,
            suggested_fix=suggested_fixes[0] if suggested_fixes else attempted_fix,
            attempted_fix=attempted_fix,
            result_status=result_status,
            confirmed_root_cause=confirmed_root_cause,
            notes=notes
        )
        
        print(f"\n✅ Outcome recorded successfully!")
        print(f"   Incident ID: {incident_id}")
        print(f"   Attempted fix: {attempted_fix}")
        print(f"   Result: {result_status}")
        print(f"   Outcome ID: {outcome_id}")
        print("\nThis information will help improve future troubleshooting suggestions.")
        
    except Exception as e:
        print(f"\n❌ Failed to save outcome: {e}")
        print("You can try using 'mirrorcore debug-outcome' to record this manually.")


def handle_analyze_log(args):
    """Handle analyze-log command for raw terminal output analysis."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .intake.log_parser import LogParser
    from .terminal.signatures import IncidentSignatureDetector
    from .reasoning.response_engine import ReasoningResponseEngine
    from .memory.retrieval import MemoryRetrieval
    
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    retrieval = MemoryRetrieval()
    
    print("📋 Paste raw terminal output, logs, or stack traces")
    print("   End input with Ctrl+D (Unix) or Ctrl+Z then Enter (Windows)")
    print("   Or type 'exit' to cancel")
    print("=" * 60)
    
    # Read multiline input
    lines = []
    try:
        while True:
            try:
                line = input()
                if line.lower() in ['exit', 'quit']:
                    print("Cancelled.")
                    return
                lines.append(line)
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    if not lines:
        print("No input provided.")
        return
    
    raw_text = '\n'.join(lines)
    
    # Parse the log
    parser = LogParser()
    parsed_log = parser.parse_log(raw_text)
    
    print("\n" + "=" * 60)
    print("🔍 LOG ANALYSIS RESULTS")
    print("=" * 60)
    
    # Show extracted error signals
    print(f"\n📊 Detected Error Category: {parsed_log.likely_error_category.upper()}")
    print(f"🔑 Keywords Found: {', '.join(parsed_log.likely_keywords) if parsed_log.likely_keywords else 'None'}")
    
    # Show command context if detected
    if parsed_log.detected_command:
        print(f"\n⚡ Detected Command: {parsed_log.detected_command}")
        print(f"   Subsystem: {parsed_log.detected_subsystem}")
        print(f"   Context: {parsed_log.command_context}")
    
    if parsed_log.extracted_lines:
        print(f"\n📝 Relevant Error Lines ({len(parsed_log.extracted_lines)}):")
        for i, line in enumerate(parsed_log.extracted_lines[:10], 1):  # Show first 10
            print(f"   {i}. {line}")
        
        if len(parsed_log.extracted_lines) > 10:
            print(f"   ... and {len(parsed_log.extracted_lines) - 10} more lines")
    
    # Show compact summary
    print(f"\n📄 Summary: {parsed_log.compact_summary_text}")
    
    # Use existing incident detection on parsed content
    print(f"\n🎯 INCIDENT DETECTION")
    print("-" * 40)
    
    detector = IncidentSignatureDetector()
    
    # Prepare command context for enhanced detection
    command_context = None
    if parsed_log.detected_command and parsed_log.detected_subsystem:
        command_context = {
            'command': parsed_log.detected_command,
            'subsystem': parsed_log.detected_subsystem,
            'context': parsed_log.command_context
        }
    
    # Try detection on compact summary first
    incident_without_context = detector.detect_incident(parsed_log.compact_summary_text, None)
    incident_with_context = detector.detect_incident(parsed_log.compact_summary_text, command_context)
    
    # If no match, try on extracted lines
    if not incident_with_context and parsed_log.extracted_lines:
        for line in parsed_log.extracted_lines[:3]:
            incident_without_context = detector.detect_incident(line, None)
            incident_with_context = detector.detect_incident(line, command_context)
            if incident_with_context:
                break
    
    confidence_without = incident_without_context.confidence if incident_without_context else 0.0
    confidence_with = incident_with_context.confidence if incident_with_context else 0.0
    
    if incident_with_context:
        print(f"✅ Recognized Incident Type: {incident_with_context.incident_type}")
        print(f"   Confidence: {confidence_with:.2f}")
        
        # Store the incident for outcome linking
        try:
            incident_id = db_store.save_terminal_incident(
                command="analyze-log",
                error_message=f"Incident: {incident_with_context.incident_type}",
                root_cause=f"Detected with confidence: {confidence_with:.2f}",
                solution=parsed_log.compact_summary_text,
                success=False
            )
        except Exception:
            incident_id = None  # Continue even if storage fails
        
        # Show command context contribution
        if command_context and command_context.get('subsystem'):
            print(f"   Command Context: {command_context['subsystem']} command detected")
            context_boost = detector._calculate_context_boost(incident_with_context.incident_type, command_context)
            if context_boost > 0:
                print(f"   Context Boost: +{context_boost:.2f} confidence")
        
        # Get ranked historical fixes for this incident type
        try:
            from .terminal.historical_fix_gate import should_surface_historical_fix_guidance

            ranked_fixes = db_store.get_ranked_fixes_by_incident_type(
                incident_with_context.incident_type, limit=3
            )
            successful_ranked = [f for f in ranked_fixes if f.get("success_count", 0) > 0]
            show_historical = False
            if successful_ranked:
                _ctx_text = parsed_log.compact_summary_text
                if parsed_log.extracted_lines:
                    _ctx_text = _ctx_text + "\n" + "\n".join(parsed_log.extracted_lines[:5])
                show_historical = should_surface_historical_fix_guidance(
                    successful_ranked[0],
                    incident_with_context.incident_type,
                    confidence_with,
                    likely_error_category=parsed_log.likely_error_category,
                    signal_families=parsed_log.signal_families,
                    detected_subsystem=parsed_log.detected_subsystem,
                    user_input=_ctx_text,
                )

            if ranked_fixes and show_historical:
                print(f"\n💡 HISTORICAL FIX GUIDANCE")
                print("-" * 40)
                
                # Show top ranked fix with explanation
                top_fix = successful_ranked[0]
                print(f"Most successful approach: {top_fix['normalized_fix']}")
                print(f"   Reason: {top_fix['success_count']} successful, {top_fix['failed_count']} failed attempts")
                
                # Show other successful options
                successful_fixes = [f for f in ranked_fixes if f['success_count'] > 0]
                if len(successful_fixes) > 1:
                    print(f"\nOther successful approaches:")
                    for i, fix in enumerate(successful_fixes[1:3], 2):
                        print(f"  {i}. {fix['normalized_fix']} ({fix['success_count']} success)")
                
                # Warn about commonly failed approaches
                failed_fixes = [f for f in ranked_fixes if f['failed_count'] > f['success_count']]
                if failed_fixes:
                    print(f"\nApproaches that typically fail:")
                    for fix in failed_fixes[:2]:  # Show top 2 failed
                        print(f"  • {fix['normalized_fix']} ({fix['failed_count']} failed)")
                
                print("\n")
            
        except Exception:
            pass  # Don't fail if historical retrieval fails

        # Phase 24: Learned cross-session fix patterns
        try:
            matched_pattern = retrieval.match_best_fix_pattern_for_issue(
                parsed_log=parsed_log,
                ranked_hypotheses=None,
                memory_store=db_store,
            )
        except Exception:
            matched_pattern = None

        if matched_pattern:
            pattern, strategies = matched_pattern
            print(f"\n💡 LEARNED FIX PATTERNS")
            print("-" * 40)
            for idx, strategy in enumerate(strategies, 1):
                rate_pct = int(round(strategy.success_rate * 100))
                print(
                    f"{idx}. {strategy.name} "
                    f"({rate_pct}% success, "
                    f"{strategy.successful_attempts}/{strategy.total_attempts} attempts)"
                )
        
        # Show signal families and subsystem information
        if parsed_log.detected_subsystem:
            print(f"\n🔧 SUBSYSTEM ANALYSIS")
            print("-" * 40)
            print(f"Detected Subsystem: {parsed_log.detected_subsystem}")
            
            # Show allowed subsystems for the detected incident
            if hasattr(incident_with_context, 'allowed_subsystems') and incident_with_context.allowed_subsystems:
                if parsed_log.detected_subsystem in incident_with_context.allowed_subsystems:
                    print(f"✅ Subsystem match: {parsed_log.detected_subsystem} → {incident_with_context.incident_type}")
                else:
                    print(f"⚠️ Subsystem mismatch: {parsed_log.detected_subsystem} vs {incident_with_context.allowed_subsystems}")
            else:
                print(f"✅ Broad incident type: {incident_with_context.incident_type}")
        
        # Show signal families and conflicts
        if parsed_log.signal_families:
            print(f"\n📊 SIGNAL ANALYSIS")
            print("-" * 40)
            print(f"Signal Families Detected: {', '.join(parsed_log.signal_families)}")
            
            if len(parsed_log.signal_families) > 1:
                print(f"⚠️ Multiple signal families detected - confidence penalties applied")
                print(f"   Conflict penalty: {0.1 * len(parsed_log.signal_families):.1f}")
        
        # Generate targeted response using existing reasoning engine
        print(f"\n🛠️ TARGETED TROUBLESHOOTING")
        print("-" * 40)
        
        reasoning_engine = ReasoningResponseEngine()
        response_result = reasoning_engine._generate_targeted_response(
            incident_with_context, 
            type('MockContext', (), {
                'user_input': parsed_log.compact_summary_text,
                'persona_profile': {},
                'conversation_history': [],
                'memory_store': db_store,
                'incident_id': incident_id,  # Pass the stored incident ID
                'signal_families': parsed_log.signal_families  # Pass signal families for conflict detection
            })
        )
        
        print(response_result.response)
        
        # Show escalation guidance if present
        if response_result.escalation_guidance:
            print(response_result.escalation_guidance)
        
        # Prompt for assisted outcome capture if applicable
        if response_result.should_prompt_outcome and response_result.incident_id:
            outcome_prompt = reasoning_engine.prompt_assisted_outcome_capture(
                response_result.incident_id, 
                response_result.suggested_fixes or []
            )
            print(outcome_prompt)
            
            # Handle user response
            try:
                user_response = input().strip().lower()
                if user_response in ['y', 'yes', 'yeah', 'yep']:
                    _collect_assisted_outcome(db_store, response_result.incident_id, response_result.suggested_fixes or [])
                else:
                    print("No problem. You can use 'mirrorcore debug-outcome' anytime to record outcomes.")
            except (EOFError, KeyboardInterrupt):
                print("\nNo problem. You can use 'mirrorcore debug-outcome' anytime to record outcomes.")
    
    else:
        print("❌ No confident incident match")

        # Multi-signal diagnostic mode
    if parsed_log.signal_families and len(parsed_log.signal_families) > 1:
        print("\n🧠 MULTI-SIGNAL DIAGNOSTIC MODE")
        print("-" * 40)

        layer_map = {
            "python_runtime": "Application/runtime failure",
            "error": "Application/runtime failure",
            "config": "Configuration issue",
            "connection_refused": "Backend connectivity issue",
            "timeout": "Backend connectivity issue",
            "network_connectivity": "Network communication failure",
            "permissions": "Permission or filesystem issue",
            "permission_denied": "Permission or filesystem issue",
        }

        layers = []
        seen = set()

        for family in parsed_log.signal_families:
            if family in layer_map and layer_map[family] not in seen:
                layers.append(layer_map[family])
                seen.add(layer_map[family])

        if layers:
            print("\nPossible failure layers:")
            for layer in layers:
                print(f"• {layer}")

            print("\nSuggested investigation order:")

            step = 1
            for layer in layers:
                if "Configuration" in layer:
                    print(f"{step}. Validate configuration values and environment variables")
                elif "Application" in layer:
                    print(f"{step}. Inspect traceback and application logs")
                elif "Backend" in layer or "Network" in layer:
                    print(f"{step}. Confirm backend service availability and connectivity")
                elif "Permission" in layer:
                    print(f"{step}. Verify filesystem permissions and service user access")
                step += 1
        
        # Generate and display ranked root-cause hypotheses
        reasoning_engine = ReasoningResponseEngine()
        ranked_hypotheses = reasoning_engine.generate_ranked_root_cause_hypotheses(parsed_log, command_context, db_store)
        
        if ranked_hypotheses:
            print(f"\n🧩 ROOT-CAUSE HYPOTHESES")
            print("-" * 40)
            
            # Show most likely hypothesis
            if ranked_hypotheses:
                top_hypothesis = ranked_hypotheses[0]
                print(f"Most likely hypothesis:")
                print(f"1. {top_hypothesis.text}")
                print(f"   Reason: {top_hypothesis.reason}")
            
            # Show other plausible hypotheses
            if len(ranked_hypotheses) > 1:
                print(f"\nOther plausible hypotheses:")
                for i, hypothesis in enumerate(ranked_hypotheses[1:], 2):
                    print(f"{i}. {hypothesis.text}")
                    print(f"   Reason: {hypothesis.reason}")
            
            # Phase 23: Investigation Memory Retrieval
            try:
                similar_investigations = retrieval.retrieve_similar_investigations(
                    parsed_log, ranked_hypotheses, db_store, limit=3
                )
            except Exception:
                similar_investigations = None

            print(f"\nSIMILAR PAST INVESTIGATIONS")
            print("-" * 40)
            if similar_investigations:
                for match in similar_investigations:
                    subsystem_label = match.subsystem or "unknown"
                    root_cause_label = match.root_cause_category or "unknown"
                    strategy_label = match.strategy_family or "unknown"
                    
                    print(f"{match.timestamp}  [Subsystem: {subsystem_label}]")
                    print(f"  Root cause category: {root_cause_label}")
                    print(f"  Strategy used: {strategy_label}")
                    if match.resolution_summary:
                        print(f"  Resolution summary: {match.resolution_summary}")
                    print()
            else:
                print("No similar resolved investigations found.\n")

            # Phase 24: Learned cross-session fix patterns
            try:
                matched_pattern = retrieval.match_best_fix_pattern_for_issue(
                    parsed_log=parsed_log,
                    ranked_hypotheses=ranked_hypotheses,
                    memory_store=db_store,
                )
            except Exception:
                matched_pattern = None

            if matched_pattern:
                pattern, strategies = matched_pattern
                print(f"\n💡 LEARNED FIX PATTERNS")
                print("-" * 40)
                for idx, strategy in enumerate(strategies, 1):
                    rate_pct = int(round(strategy.success_rate * 100))
                    print(
                        f"{idx}. {strategy.name} "
                        f"({rate_pct}% success, "
                        f"{strategy.successful_attempts}/{strategy.total_attempts} attempts)"
                    )
            
            # Generate and display diagnostic command suggestions
            if ranked_hypotheses:
                top_hypothesis = ranked_hypotheses[0]
                diagnostic_commands = reasoning_engine.generate_diagnostic_commands(top_hypothesis, parsed_log, command_context)
                
                if diagnostic_commands["primary"]:
                    print(f"\n🔎 NEXT DIAGNOSTIC STEPS")
                    print("-" * 40)
                    print(f"Recommended checks for this hypothesis:")
                    
                    for i, cmd in enumerate(diagnostic_commands["primary"], 1):
                        print(f"{i}. {cmd['description']}")
                        print(f"   Command: {cmd['command']}")
                    
                    if diagnostic_commands["additional"]:
                        print(f"\nOther useful checks:")
                        for cmd in diagnostic_commands["additional"]:
                            print(f"- {cmd['description']}")
                            print(f"  Command: {cmd['command']}")
                
        print(f"\n🛠️ GENERAL ANALYSIS")
        print("-" * 40)
        
        # Show detected command context even if no incident
        if command_context:
            print(f"Detected Command: {command_context.get('command', 'Unknown')}")
            print(f"Detected Subsystem: {command_context.get('subsystem', 'Unknown')}")
        
        # Show signal families if detected
        if parsed_log.signal_families:
            print(f"Signal Families: {', '.join(parsed_log.signal_families)}")
        
        # Show error category and summary
        print(f"\nError Category: {parsed_log.likely_error_category}")
        print(f"Summary: {parsed_log.compact_summary_text}")
        
        # Store analysis session if we have ranked hypotheses (multi-signal mode)
        if ranked_hypotheses:
            try:
                session_data = {
                    'detected_subsystem': command_context.get('subsystem') if command_context else None,
                    'signal_families': parsed_log.signal_families,
                    'original_hypotheses': [
                        {
                            'text': h.text,
                            'category': h.category,
                            'score': h.score,
                            'reason': h.reason
                        } for h in ranked_hypotheses
                    ],
                    'top_hypothesis_category': ranked_hypotheses[0].category if ranked_hypotheses else None,
                    'suggested_commands': diagnostic_commands if 'diagnostic_commands' in locals() else [],
                    'analysis_summary': f"{parsed_log.likely_error_category}: {parsed_log.compact_summary_text}"
                }
                
                session_id = db_store.store_analysis_session(session_data)
                
                # Store investigation steps if diagnostic commands were generated
                if 'diagnostic_commands' in locals() and diagnostic_commands:
                    db_store.store_investigation_steps(session_id, diagnostic_commands)
                # Initialize investigation tracking
                initial_strategy = _infer_strategy_family(session_data['top_hypothesis_category'])
                progress_metrics = {'completion_rate': 0.0, 'strategies_attempted': 1}
                                
                db_store.update_investigation_tracking(
                    session_id,
                    initial_strategy,
                    [initial_strategy],
                    progress_metrics,
                    'active',
                    [],
                    []
                )
                print(f"\n💾 Analysis session stored with ID: {session_id}")
                print("   Use 'mirrorcore analyze-followup' to update with new evidence.")
            except Exception:
                print("\n⚠️ Could not store analysis session for follow-up")
        
        # Store general analysis for learning
        try:
            incident_id = db_store.save_terminal_incident(
                command="analyze-log",
                error_message=f"General analysis: {parsed_log.likely_error_category}",
                root_cause="No specific incident pattern matched",
                solution=parsed_log.compact_summary_text,
                success=False
            )
            print(f"\n💾 Analysis stored with ID: {incident_id}")
            print("   Use 'mirrorcore debug-outcome' to record what fix worked.")
        except Exception:
            print("\n⚠️ Could not store analysis for learning")
    
    print()  # Add spacing after analysis


def handle_analyze_followup(args):
    """Handle analyze-followup command for evidence-based reanalysis."""
    from pathlib import Path
    from .intake.log_parser import LogParser
    from .reasoning.response_engine import ReasoningResponseEngine, RootCauseHypothesis
    from .db.store import DatabaseStore

    print("🔄 EVIDENCE-BASED REANALYSIS")
    print("=" * 50)
    print("Paste follow-up command output or additional logs")
    print("End input with Ctrl+D (Unix) or Ctrl+Z then Enter (Windows)")
    print("Or type 'exit' to cancel")
    print("=" * 50)

    evidence_lines = []
    try:
        while True:
            try:
                line = input()
                if line.strip().lower() == 'exit':
                    print("Follow-up analysis cancelled.")
                    return
                evidence_lines.append(line)
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\nFollow-up analysis cancelled.")
        return

    if not evidence_lines:
        print("No evidence provided for follow-up analysis.")
        return

    evidence_text = '\n'.join(evidence_lines)

    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    log_parser = LogParser()
    reasoning_engine = ReasoningResponseEngine()

    session = db_store.get_latest_analysis_session()
    if not session:
        print("No recent analysis session found for follow-up.")
        print("Run 'mirrorcore analyze-log' first to create an analysis session.")
        return

    print(f"\n📋 Using analysis session from: {session['timestamp']}")
    print(f"Original subsystem: {session['detected_subsystem']}")
    print(f"Original top hypothesis: {session['top_hypothesis_category']}")

    evidence_signals = log_parser.parse_evidence(evidence_text)
    if not evidence_signals:
        print("\n⚠️ No specific evidence signals detected in follow-up.")
        print("Evidence text will be stored but hypotheses cannot be updated.")
        return

    print(f"\n🔍 Detected evidence signals: {', '.join(evidence_signals)}")

    evidence_summary = reasoning_engine.generate_evidence_summary(evidence_signals)
    print("\n🔄 EVIDENCE UPDATE")
    print("-" * 40)
    if evidence_summary['supports']:
        print("New evidence supports:")
        for support in evidence_summary['supports']:
            print(f"- {support}")
    if evidence_summary['weakens']:
        print("\nNew evidence weakens:")
        for weaken in evidence_summary['weakens']:
            print(f"- {weaken}")

    original_hypotheses = []
    for hyp_data in session['original_hypotheses']:
        original_hypotheses.append(
            RootCauseHypothesis(
                text=hyp_data['text'],
                related_families=session['signal_families'],
                related_subsystems=[session['detected_subsystem']] if session['detected_subsystem'] else [],
                category=hyp_data['category'],
                score=hyp_data['score'],
                reason=hyp_data['reason'],
            )
        )

    updated_hypotheses = reasoning_engine.update_hypotheses_with_evidence(
        original_hypotheses, evidence_signals, session
    )

    matching_step_id = db_store.find_matching_step(session['id'], evidence_text)
    if matching_step_id:
        db_store.complete_investigation_step(matching_step_id, evidence_signals)

    investigation_status = reasoning_engine.generate_investigation_status(
        session['id'], evidence_signals, db_store
    )
    if not investigation_status:
        investigation_status = {
            'completed_steps': [],
            'pending_steps': [],
            'next_step': None,
            'total_steps': 0,
            'completed_count': 0,
            'pending_count': 0,
        }

    command_context = {'subsystem': session['detected_subsystem']} if session['detected_subsystem'] else None
    top_hypothesis = next((h for h in updated_hypotheses if not h.eliminated), None)
    if top_hypothesis:
        diagnostic_commands = reasoning_engine.generate_deduplicated_diagnostic_commands(
            session['id'], top_hypothesis, command_context, db_store
        )
    else:
        diagnostic_commands = {'primary': [], 'additional': []}

    tracking_data = db_store.get_investigation_tracking(session['id'])
    current_strategy_family = tracking_data['current_strategy_family']
    if not current_strategy_family:
        current_strategy_family = _infer_strategy_family(session.get('top_hypothesis_category'))

    strategies_attempted = tracking_data['strategies_attempted'] or [current_strategy_family]
    current_evidence = sorted(set(evidence_signals))
    current_hypotheses = [
        {
            'category': h.category,
            'score': h.score,
            'eliminated': h.eliminated,
        }
        for h in updated_hypotheses
    ]
    evidence_history = (tracking_data['evidence_history'] or []) + [current_evidence]
    hypothesis_history = (tracking_data['hypothesis_history'] or []) + [current_hypotheses]

    stall_detection = reasoning_engine.detect_investigation_stall(
        investigation_status,
        current_strategy_family,
        strategies_attempted,
        evidence_history,
        hypothesis_history,
    )

    if stall_detection.is_stalled and stall_detection.alternative_family:
        new_strategies = list(strategies_attempted)
        if stall_detection.alternative_family not in new_strategies:
            new_strategies.append(stall_detection.alternative_family)
        db_store.update_investigation_tracking(
            session['id'],
            stall_detection.alternative_family,
            new_strategies,
            stall_detection.progress_metrics,
            'stalled',
            evidence_history,
            hypothesis_history,
        )
    else:
        current_progress = {
            'completion_rate': (
                investigation_status['completed_count'] / investigation_status['total_steps']
                if investigation_status['total_steps'] > 0 else 0.0
            ),
            'strategies_attempted': len(strategies_attempted),
            'evidence_diversity': len({tuple(e) for e in evidence_history}) if evidence_history else 0,
        }
        db_store.update_investigation_tracking(
            session['id'],
            current_strategy_family,
            strategies_attempted,
            current_progress,
            tracking_data['investigation_state'] or 'active',
            evidence_history,
            hypothesis_history,
        )

    print("\n📋 INVESTIGATION STATUS")
    print("-" * 40)

    eliminated_hypotheses = [h for h in updated_hypotheses if h.eliminated]
    if eliminated_hypotheses:
        print("Eliminated hypotheses:")
        for hypothesis in eliminated_hypotheses:
            print(f"✗ {hypothesis.category}")
            print(f"  Reason: {hypothesis.elimination_reason}")
        print()

    if investigation_status['completed_steps']:
        print("Completed steps:")
        for step in investigation_status['completed_steps']:
            print(f"✓ {step['description']}")
            print(f"  Command: {step['command']}")
        print()

    if investigation_status['pending_steps']:
        print("Pending steps:")
        for step in investigation_status['pending_steps']:
            print(f"- {step['description']}")
            print(f"  Command: {step['command']}")
        print()

    if investigation_status['next_step']:
        print("Next recommended step:")
        next_step = investigation_status['next_step']
        print(f"- {next_step['description']}")
        print(f"  Command: {next_step['command']}")
        print()

    if stall_detection.is_stalled:
        print("\n🚨 INVESTIGATION STALL DETECTED")
        print("-" * 40)
        print(f"Stall reason: {stall_detection.stall_reason}")
        print(f"Failed strategy families: {', '.join(stall_detection.failed_families)}")
        print(f"Progress metrics: {stall_detection.progress_metrics['completion_rate']:.1%} completion rate")

        if stall_detection.alternative_family:
            print(f"\n🔄 SWITCHING TO {stall_detection.alternative_family.upper()} STRATEGY")
            print("-" * 40)
            alt_commands = reasoning_engine.generate_alternative_strategy_commands(
                stall_detection.alternative_family, command_context
            )
            if alt_commands['primary']:
                print("\n🔎 ALTERNATIVE DIAGNOSTIC STEPS")
                print("-" * 40)
                print("Recommended checks for alternative strategy:")
                for i, cmd in enumerate(alt_commands['primary'], 1):
                    print(f"{i}. {cmd['description']}")
                    print(f"   Command: {cmd['command']}")

    print("\n🧩 UPDATED ROOT-CAUSE HYPOTHESES")
    print("-" * 40)
    active_hypotheses = [h for h in updated_hypotheses if not h.eliminated]
    if active_hypotheses:
        top_hypothesis = active_hypotheses[0]
        print("Most likely hypothesis:")
        print(f"1. {top_hypothesis.text}")
        print(f"   Reason: {top_hypothesis.reason}")
        if len(active_hypotheses) > 1:
            print("\nOther plausible hypotheses:")
            for i, hypothesis in enumerate(active_hypotheses[1:], 2):
                print(f"{i}. {hypothesis.text}")
                print(f"   Reason: {hypothesis.reason}")
    else:
        print("No active hypotheses remain after applying follow-up evidence.")

    if diagnostic_commands['primary']:
        print("\n🔎 UPDATED NEXT DIAGNOSTIC STEPS")
        print("-" * 40)
        print("Recommended checks for updated hypothesis:")
        for i, cmd in enumerate(diagnostic_commands['primary'], 1):
            print(f"{i}. {cmd['description']}")
            print(f"   Command: {cmd['command']}")
        if diagnostic_commands['additional']:
            print("\nOther useful checks:")
            for cmd in diagnostic_commands['additional']:
                print(f"- {cmd['description']}")
                print(f"  Command: {cmd['command']}")
    elif investigation_status['completed_count'] > 0:
        print("\n🔎 INVESTIGATION PROGRESS")
        print("-" * 40)
        print(f"Completed {investigation_status['completed_count']}/{investigation_status['total_steps']} investigation steps.")
        print("All suggested diagnostic commands have been executed.")
        print("Consider reviewing all evidence or escalating investigation.")

    print("\n✅ Follow-up analysis complete.")
    print(f"Session {session['id']} updated with new evidence.")

def handle_debug_history(args):
    """Handle debug history command to view ranked fix history."""
    from pathlib import Path
    from .db.store import DatabaseStore
    
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    
    if args.incident_type:
        # Show history for specific incident type
        print(f"Ranked fix history for: {args.incident_type}")
        print("=" * 50)
        
        ranked_fixes = db_store.get_ranked_fixes_by_incident_type(args.incident_type, limit=10)
        
        if not ranked_fixes:
            print("No fix history found for this incident type.")
            return
        
        for i, fix in enumerate(ranked_fixes, 1):
            print(f"\n{i}. {fix['normalized_fix']}")
            print(f"   Score: {fix['score']:.1f}")
            print(f"   Success: {fix['success_count']}, Partial: {fix['partial_count']}, Failed: {fix['failed_count']}")
            print(f"   Total attempts: {fix['total_attempts']}")
            if fix['latest_timestamp']:
                print(f"   Last tried: {fix['latest_timestamp'][:19]}")
            
            # Show original variations
            if len(fix['original_fixes']) > 1:
                print(f"   Variations: {len(fix['original_fixes'])} attempts recorded")
    else:
        # Show summary of all incident types with history
        print("Available incident types with fix history:")
        print("=" * 50)
        
        # Get all incident types with outcomes
        all_outcomes = db_store.get_terminal_fix_outcomes(limit=100)
        incident_types = {}
        
        for outcome in all_outcomes:
            incident_type = outcome.get('incident_type', 'Unknown')
            if incident_type not in incident_types:
                incident_types[incident_type] = {'success': 0, 'failed': 0, 'partial': 0}
            
            result = outcome.get('result_status', '').lower()
            if result == 'success':
                incident_types[incident_type]['success'] += 1
            elif result == 'failed':
                incident_types[incident_type]['failed'] += 1
            elif result == 'partial':
                incident_types[incident_type]['partial'] += 1
        
        if not incident_types:
            print("No fix history found.")
            return
        
        for incident_type, counts in sorted(incident_types.items()):
            total = counts['success'] + counts['failed'] + counts['partial']
            success_rate = (counts['success'] / total * 100) if total > 0 else 0
            print(f"  {incident_type}:")
            print(f"    Total attempts: {total}")
            print(f"    Success rate: {success_rate:.1f}%")
            print(f"    Success: {counts['success']}, Failed: {counts['failed']}, Partial: {counts['partial']}")
        
        print(f"\nUse: mirrorcore debug-history <incident_type> to see detailed fixes for a specific type.")


def handle_debug_outcome(args):
    """Handle debug outcome recording command."""
    from pathlib import Path
    from .db.store import DatabaseStore
    
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    
    # Get recent incidents for selection
    recent_incidents = db_store.get_terminal_incidents(limit=10)
    
    if not recent_incidents:
        print("No recent incidents found. Try debugging an issue first.")
        return
    
    print("Recent incidents:")
    for i, incident in enumerate(recent_incidents, 1):
        print(f"  {i}. {incident['timestamp'][:19]} - {incident['error_message']}")
    
    # Get user selection
    try:
        selection = int(input("Select incident number (or 0 to cancel): "))
        if selection == 0 or selection < 1 or selection > len(recent_incidents):
            print("Cancelled.")
            return
        
        selected_incident = recent_incidents[selection - 1]
        print(f"\nSelected incident: {selected_incident['error_message']}")
        
        # Get fix details
        suggested_fix = input("What fix did you try? ")
        result_status = input("What was the result? (success/failed/partial/unknown): ")
        confirmed_root_cause = input("What was the confirmed root cause? (optional): ") or None
        notes = input("Any additional notes? (optional): ") or None
        
        # Save the outcome
        outcome_id = db_store.save_terminal_fix_outcome(
            incident_id=selected_incident['id'],
            suggested_fix=selected_incident.get('solution', 'Initial suggestion'),
            attempted_fix=suggested_fix,
            result_status=result_status,
            confirmed_root_cause=confirmed_root_cause,
            notes=notes
        )
        
        print(f"\n✓ Fix outcome saved with ID: {outcome_id}")
        print(f"  Fix attempted: {suggested_fix}")
        print(f"  Result: {result_status}")
        if confirmed_root_cause:
            print(f"  Root cause: {confirmed_root_cause}")
        
        # Phase 22: automatically close the latest open investigation session on success
        if result_status.strip().lower() == "success":
            try:
                open_session = db_store.get_latest_analysis_session()
            except Exception:
                open_session = None
            
            # Match get_latest_analysis_session: open = not completed/resolved
            _ss = (open_session.get("session_status") or "").lower()
            if open_session and _ss not in ("completed", "resolved"):
                try:
                    db_store.update_analysis_session_status(open_session["id"], "completed")
                    print(f"\n Investigation session {open_session['id']} marked as resolved.")
                    print("  Future analyze-followup calls will not attach to this session.")
                except Exception:
                    # Do not fail debug-outcome if session resolution update fails
                    pass
        
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
    except EOFError:
        print("\nCancelled.")


def handle_debug(args):
    """Handle debug command."""
    print(f"Debugging: {args.problem}")
    
    # Placeholder for debugging functionality
    # TODO: Implement debug logic using Terminal Agent
    print("Debug functionality not yet implemented")


def handle_decide(args):
    """Handle decide command."""
    print(f"Decision assistance for: {args.decision}")
    
    # Placeholder for decision assistance
    # TODO: Implement decision logic using Decision Agent
    print("Decision assistance not yet implemented")


def handle_profile(args):
    """Handle profile command."""
    if args.analyze:
        print("Analyzing recent patterns...")
    elif args.update:
        print(f"Updating profile aspect: {args.update}")
    else:
        print("Showing current profile...")
    
    # Placeholder for profile management
    # TODO: Implement profile logic using Persona Agent
    print("Profile management not yet implemented")


def handle_assess(args):
    """Handle assessment command."""
    if getattr(args, 'continue'):
        print("Continuing previous assessment...")
    else:
        print("Starting initial assessment...")
    
    # Placeholder for assessment functionality
    # TODO: Implement assessment logic using Intake Agent
    print("Assessment functionality not yet implemented")


def handle_init(args):
    """Handle init command - run the branching assessment."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .intake.scenario_engine import ScenarioEngine
    from .intake.profiler import Profiler
    
    # Check if persona already exists
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    
    if db_store.persona_exists():
        print("⚠️  Warning: Persona profile already exists.")
        print("A baseline assessment has already been completed.")
        print("To reset and retake the assessment, run:")
        print("  python3 scripts/reset_assessment.py")
        print("Then run 'mirrorcore init' again.")
        return
    
    print("Starting Mirrorcore Initial Assessment...")
    print("This will help Mirrorcore understand your reasoning patterns.")
    print("\nPress Enter to begin, or Ctrl+C to cancel:")
    
    try:
        input()  # Wait for user confirmation
    except KeyboardInterrupt:
        print("\nAssessment cancelled.")
        return
    
    # Run the assessment
    try:
        engine = ScenarioEngine()
        responses = engine.run_assessment()
        
        if not responses:
            print("No responses collected. Assessment incomplete.")
            return
        
        # Analyze responses
        print("\nAnalyzing your responses...")
        profiler = Profiler()
        persona_traits = profiler.analyze_responses(responses)
        
        # Convert to dict format for storage
        persona_dict = {}
        for trait_name, trait in persona_traits.items():
            persona_dict[trait_name] = {
                "value": trait.value,
                "confidence": trait.confidence,
                "evidence_count": trait.evidence_count,
                "contributing_responses": trait.contributing_responses
            }
        
        # Save to database
        print("Saving your persona profile...")
        trait_ids = db_store.save_persona_traits(persona_dict)
        
        # Save raw responses
        response_dicts = []
        for response in responses:
            response_dict = {
                "assessment_session_id": response.assessment_session_id,
                "question_id": response.question_id,
                "question_type": response.question_type.value,
                "selected_option_id": response.selected_option_id,
                "custom_response": response.custom_response,
                "explanation": response.explanation,
                "trigger_reason": response.trigger_reason,
                "order_index": response.order_index
            }
            response_dicts.append(response_dict)
        
        memory_ids = db_store.save_assessment_responses(response_dicts)
        
        # Display results
        print("\n" + "="*60)
        print("ASSESSMENT RESULTS")
        print("="*60)
        
        # Show improved summary
        summary = profiler.generate_assessment_summary(persona_traits, responses)
        print(summary)
        
        print(f"\n✓ Saved {len(trait_ids)} persona traits")
        print(f"✓ Saved {len(memory_ids)} raw assessment responses")
        print(f"✓ Assessment session ID: {responses[0].assessment_session_id if responses else 'unknown'}")
        
        # Show confidence summary
        avg_confidence = sum(t.confidence for t in persona_traits.values()) / len(persona_traits)
        confidence_pct = int(avg_confidence * 100)
        print(f"✓ Average confidence: {confidence_pct}%")
        
        print("\nYour baseline persona profile is now ready!")
        print("Mirrorcore will use this profile to provide personalized assistance.")
        
        db_store.close()
        
    except Exception as e:
        print(f"\n❌ Error during assessment: {e}")
        print("Your progress has not been saved.")
        print("You can safely restart the assessment.")
        return


def handle_memory(args):
    """Handle memory command."""
    print(f"Memory action: {args.action}")
    
    if args.action == "drift":
        # Show drift status
        handle_drift_status(args)
    else:
        # Placeholder for memory management
        # TODO: Implement memory logic using Memory Agent
        print("Memory management not yet implemented")


def handle_drift_status(args):
    """Handle drift status command."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .memory.retrieval import MemoryRetrieval
    
    # Initialize components
    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)
    retrieval = MemoryRetrieval()
    
    print("🔍 Persona Drift Analysis")
    print("=" * 50)
    
    # Get drift summary
    drift_summary = retrieval.summarize_drift_patterns(days_back=90, memory_store=db_store)
    
    if "message" in drift_summary:
        print(f"Status: {drift_summary['message']}")
        return
    
    print(f"Analysis Period: Last {drift_summary['evaluation_period_days']} days")
    print(f"Total Evaluations: {drift_summary['total_evaluations']}")
    print(f"Traits Under Pressure: {len(drift_summary['traits_under_pressure'])}")
    
    if drift_summary['traits_under_pressure']:
        print("Traits with active drift pressure:")
        for trait in drift_summary['traits_under_pressure']:
            print(f"  - {trait}")
    
    if drift_summary['revision_suggestions']:
        print(f"\nPending Revision Suggestions: {len(drift_summary['revision_suggestions'])}")
        for suggestion in drift_summary['revision_suggestions'][:5]:  # Show last 5
            print(f"  • {suggestion['trait']}: {suggestion['suggestion']}")
            print(f"    (Suggested {suggestion['timestamp']})")
    
    if drift_summary['confidence_trends']:
        print(f"\nConfidence Trends:")
        for trait, trend in drift_summary['confidence_trends'].items():
            direction = "↗️" if trend['trend'] == "improving" else "↘️"
            print(f"  {trait}: {direction} {trend['avg_adjustment']:+.2f} avg change")
    
    print(f"\nSummary: {drift_summary['summary']}")
    print("=" * 50)


def handle_interview(args):
    """Handle interview command — run a multi-scenario decision interview session."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .decision.interview import run_interview_session

    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)

    # Use a persisted rotation index so scenario order varies across runs.
    session_index = db_store.get_next_decision_interview_rotation_index()

    try:
        session = run_interview_session(session_index=session_index)
    except (EOFError, KeyboardInterrupt):
        print("\nInterview cancelled.")
        return

    entry_ids = []
    for result in session.results:
        entry_id = db_store.record_decision_memory(
            scenario_id=result.scenario_id,
            scenario_text=result.scenario_text,
            choice_label=result.choice_label,
            choice_value=result.choice_value,
            reasoning_label=result.reasoning_label,
            reasoning_value=result.reasoning_value,
            value_tags=result.value_tags,
            trait_signals=result.trait_signals,
            confidence_score=result.confidence_score,
        )
        entry_ids.append(entry_id)

    if session.correction:
        correction = session.correction
        metadata = {
            "status": correction.status,
            "accepted_traits": correction.accepted_traits,
            "rejected_traits": correction.rejected_traits,
            "replacement_choices": correction.replacement_choices,
        }
        correction_status = correction.status
        for eid in entry_ids:
            db_store.update_decision_memory_correction(
                eid, correction_status, metadata
            )

    print("\n" + "-" * 56)
    print(f"  Session complete — {len(session.results)} scenarios recorded.")
    if session.reflections:
        print(f"  Reflections generated: {len(session.reflections)}")
    if session.correction:
        print(f"  Correction: {session.correction.status}")
    print("-" * 56)


def handle_respond_like_me(args):
    """Handle respond-like-me — grounded likely-you reply from stored memory."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .persona.respond import generate_personal_response

    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)

    scenario = (args.scenario or "").strip()
    if not scenario:
        if PROMPT_TOOLKIT_AVAILABLE:
            try:
                scenario = prompt(
                    "Describe a situation or question (your scenario): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            try:
                scenario = input(
                    "Describe a situation or question (your scenario): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return

    if not scenario:
        print("No scenario text — nothing to respond to.")
        return

    pr = generate_personal_response(scenario, db_store)
    print()
    print("Likely response")
    print("-" * min(48, max(24, len(scenario) // 2 + 24)))
    print(pr.likely_answer)
    print()
    print("Why (brief)")
    print("-" * 13)
    print(pr.reasoning_brief)
    print()
    print(
        f"Confidence: {pr.confidence_label} ({pr.confidence:.2f})"
    )
    if pr.memory_basis:
        print("Grounded in:")
        for line in pr.memory_basis[:5]:
            print(f"  • {line}")
    elif pr.profile_hint:
        print(f"Grounded in: your saved tendencies ({pr.profile_hint})")
    else:
        print("Grounded in: little or no matching stored memory — see confidence above.")
    print()


def handle_calibrate_style(args):
    """Handle calibrate-style command — run style calibration session."""
    from pathlib import Path
    from .db.store import DatabaseStore
    from .persona.calibration import run_style_calibration_session

    db_path = Path("data") / "mirrorcore.db"
    db_store = DatabaseStore(db_path)

    prior_entries = db_store.get_recent_style_memory(limit=1000)
    session_index = len(prior_entries)

    try:
        session = run_style_calibration_session(session_index=session_index)
    except (EOFError, KeyboardInterrupt):
        print("\nStyle calibration cancelled.")
        return

    correction_status = "uncorrected"
    correction_metadata = None
    if session.correction:
        correction_status = session.correction.status
        correction_metadata = {
            "status": session.correction.status,
            "accepted_traits": session.correction.accepted_traits,
            "rejected_traits": session.correction.rejected_traits,
            "replacement_choices": session.correction.replacement_choices,
        }

    entry_ids = []
    for result in session.results:
        entry_id = db_store.record_style_memory(
            prompt_id=result.prompt_id,
            prompt_text=result.prompt_text,
            selected_label=result.selected_label,
            selected_value=result.selected_value,
            style_tags=result.style_tags,
            tone_signals=result.tone_signals,
            confidence_score=result.confidence_score,
            optional_notes=result.optional_notes,
            correction_status=correction_status,
            correction_metadata=correction_metadata,
            source="style_calibration",
        )
        entry_ids.append(entry_id)

    print("\n" + "-" * 56)
    print(f"  Session complete — {len(entry_ids)} style prompts recorded.")
    if session.reflections:
        print(f"  Reflections generated: {len(session.reflections)}")
    if session.correction:
        print(f"  Correction: {session.correction.status}")
    print("-" * 56)


def handle_status(args):
    """Handle status command."""
    print("Mirrorcore System Status")
    print("=" * 30)
    
    try:
        config = get_config()
        print(f"✓ Configuration loaded")
        
        # Use same database path as interactive sessions
        db_path = Path("data") / "mirrorcore.db"
        db_store = DatabaseStore(db_path)
        stats = db_store.get_database_stats()
        print("✓ Database connected")
        print(f"  - Persona traits: {stats['persona_traits']}")
        print(f"  - Memory entries: {stats['memory_entries']}")
        print(f"  - Terminal incidents: {stats['terminal_incidents']}")
        print(f"  - Decision records: {stats['decision_records']}")
        print(f"  - Total sessions: {stats['total_sessions']}")
            
        print(f"Version: {__version__}")
        
    except Exception as e:
        print(f"✗ Status check failed: {e}")


def handle_config(args):
    """Handle configuration command."""
    try:
        config = get_config()
        
        if args.get:
            value = config.get(args.get, "Not found")
            print(f"{args.get}: {value}")
        elif args.set:
            key, value = args.set
            print(f"Setting {key} = {value}")
            print("Configuration update not yet implemented")
        else:
            print("Current configuration:")
            print(config)
            
    except Exception as e:
        print(f"Configuration error: {e}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize database first
    try:
        db_path = Path("data") / "mirrorcore.db"
        db_store = DatabaseStore(db_path)
        db_store.initialize_database()
        if args.verbose:
            print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        sys.exit(1)
    
    # Handle no command (default to interactive)
    if not args.command:
        args.command = "interactive"
    
    # Check for persona existence (only for commands that require initialized persona)
    persona_required_commands = ["interactive", "debug", "decide", "profile", "assess", "memory"]
    
    if args.command in persona_required_commands:
        if db_store.persona_exists():
            # Load persona profile
            persona_profile = db_store.load_persona_profile()
            if args.verbose:
                print("✓ Persona profile loaded successfully.")
        else:
            print("Mirrorcore requires an initial reasoning assessment before first use.")
            print("Run: mirrorcore init")
            sys.exit(1)
    
    # Route to appropriate handler
    handlers = {
        "start": handle_start,
        "interactive": handle_interactive,
        "debug": handle_debug,
        "debug-outcome": handle_debug_outcome,
        "debug-history": handle_debug_history,
        "analyze-log": handle_analyze_log,
        "analyze-followup": handle_analyze_followup,
        "decide": handle_decide,
        "profile": handle_profile,
        "assess": handle_assess,
        "init": handle_init,
        "memory": handle_memory,
        "status": handle_status,
        "config": handle_config,
        "interview": handle_interview,
        "calibrate-style": handle_calibrate_style,
        "respond-like-me": handle_respond_like_me,
        "ask": handle_ask,
    }
    
    handler = handlers.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
