"""
Mirrorcore Configuration Management

This module handles configuration loading, validation, and management.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    "system": {
        "version": "0.1.0",
        "data_directory": "data",
        "log_level": "INFO",
        "max_memory_entries": 10000,
        "session_timeout_minutes": 60
    },
    "database": {
        "type": "sqlite",
        "path": "data/mirrorcore.db",
        "backup_enabled": True,
        "backup_interval_hours": 24
    },
    "persona": {
        "assessment_completed": False,
        "assessment_date": None,
        "confidence_threshold": 0.7,
        "learning_rate": 0.1,
        "profile_update_frequency_days": 7
    },
    "memory": {
        "retention_days": 365,
        "compression_enabled": True,
        "embedding_model": "local",
        "similarity_threshold": 0.8
    },
    "llm": {
        "provider": "local",
        "api_key": None,
        "model": "default",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout_seconds": 30
    },
    "privacy": {
        "data_export_enabled": True,
        "anonymous_usage_stats": False,
        "cloud_sync_enabled": False,
        "data_retention_policy": "manual"
    },
    "cli": {
        "default_format": "text",
        "color_enabled": True,
        "progress_bars": True,
        "completion_enabled": True
    },
    "agents": {
        "intake": {
            "enabled": True,
            "confidence_threshold": 0.8
        },
        "memory": {
            "enabled": True,
            "cache_size": 1000
        },
        "persona": {
            "enabled": True,
            "update_frequency": "daily"
        },
        "terminal": {
            "enabled": True,
            "pattern_matching": True
        },
        "decision": {
            "enabled": True,
            "tradeoff_analysis": True
        },
        "llm": {
            "enabled": True,
            "fallback_enabled": True
        }
    }
}


def find_config_file() -> Optional[Path]:
    """Find the configuration file in standard locations."""
    possible_paths = [
        Path.cwd() / "data" / "config.json",
        Path.home() / ".mirrorcore" / "config.json",
        Path("/etc/mirrorcore/config.json")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def get_default_config() -> Dict[str, Any]:
    """Get the default configuration."""
    return DEFAULT_CONFIG.copy()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from file, merging with defaults."""
    config = get_default_config()
    
    if config_path is None:
        config_path = find_config_file()
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge user config with defaults
            config = merge_configs(config, user_config)
            
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
    
    return config


def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config with default config."""
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and values."""
    required_sections = ["system", "database", "persona", "memory", "llm", "privacy"]
    
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section: {section}")
            return False
    
    # Validate specific values
    if config["system"]["max_memory_entries"] < 1:
        print("Error: max_memory_entries must be >= 1")
        return False
    
    if config["memory"]["retention_days"] < 1:
        print("Error: retention_days must be >= 1")
        return False
    
    if config["llm"]["temperature"] < 0 or config["llm"]["temperature"] > 2:
        print("Error: LLM temperature must be between 0 and 2")
        return False
    
    return True


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get and validate configuration."""
    if config_path:
        path = Path(config_path)
    else:
        path = find_config_file()
    
    config = load_config(path)
    
    if not validate_config(config):
        print("Configuration validation failed")
        sys.exit(1)
    
    return config


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> bool:
    """Save configuration to file."""
    if config_path is None:
        config_path = find_config_file()
        if config_path is None:
            config_path = Path.cwd() / "data" / "config.json"
    
    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Failed to save config to {config_path}: {e}")
        return False


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation."""
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """Set a configuration value using dot notation."""
    keys = key.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the final value
    current[keys[-1]] = value
    
    return config


def get_data_directory(config: Optional[Dict[str, Any]] = None) -> Path:
    """Get the data directory path."""
    if config is None:
        config = get_config()
    
    data_dir = Path(config["system"]["data_directory"])
    
    # Make relative paths absolute
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    
    return data_dir


def get_database_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Get the database file path."""
    if config is None:
        config = get_config()
    
    data_dir = get_data_directory(config)
    db_path = Path(config["database"]["path"])
    
    # Make relative paths absolute
    if not db_path.is_absolute():
        db_path = data_dir / db_path
    
    return db_path


# Environment variable overrides
def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    env_mappings = {
        "MIRRORCORE_DATA_DIR": "system.data_directory",
        "MIRRORCORE_LOG_LEVEL": "system.log_level",
        "MIRRORCORE_LLM_PROVIDER": "llm.provider",
        "MIRRORCORE_LLM_API_KEY": "llm.api_key",
        "MIRRORCORE_LLM_MODEL": "llm.model",
        "MIRRORCORE_DB_PATH": "database.path"
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            set_config_value(config, config_key, value)
    
    return config
