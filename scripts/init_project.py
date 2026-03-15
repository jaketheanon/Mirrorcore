#!/usr/bin/env python3
"""
Mirrorcore Project Initialization Script

This script initializes a new Mirrorcore installation by:
1. Setting up the SQLite database with required tables
2. Creating default configuration
3. Setting up initial data structures
4. Validating the installation
"""

import os
import sys
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.config import get_default_config
from mirrorcore.db.models import initialize_database


def create_data_directory():
    """Create the data directory if it doesn't exist."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory: {data_dir}")
    return data_dir


def setup_database(data_dir):
    """Initialize the SQLite database with required tables."""
    db_path = data_dir / "mirrorcore.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create tables using the database models
        initialize_database(conn)
        
        conn.commit()
        conn.close()
        
        print(f"✓ Database initialized: {db_path}")
        return db_path
        
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        sys.exit(1)


def create_configuration(data_dir):
    """Create default configuration file."""
    config_path = data_dir / "config.json"
    
    try:
        config = get_default_config()
        
        # Add installation-specific values
        config["installation"] = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "data_directory": str(data_dir)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration created: {config_path}")
        return config_path
        
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        sys.exit(1)


def create_initial_profile(data_dir):
    """Create a placeholder for the initial persona profile."""
    profiles_dir = data_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    
    placeholder_path = profiles_dir / "initial_profile.json"
    
    try:
        placeholder = {
            "status": "pending_assessment",
            "created_at": datetime.utcnow().isoformat(),
            "message": "Initial profile will be created after user assessment"
        }
        
        with open(placeholder_path, 'w') as f:
            json.dump(placeholder, f, indent=2)
        
        print(f"✓ Profile placeholder created: {placeholder_path}")
        
    except Exception as e:
        print(f"✗ Profile placeholder creation failed: {e}")


def validate_installation(data_dir):
    """Validate that the installation was successful."""
    print("\nValidating installation...")
    
    checks = [
        ("Database", data_dir / "mirrorcore.db"),
        ("Configuration", data_dir / "config.json"),
        ("Profile placeholder", data_dir / "profiles" / "initial_profile.json")
    ]
    
    all_good = True
    for name, path in checks:
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: Missing {path}")
            all_good = False
    
    if all_good:
        print("\n🎉 Mirrorcore initialization completed successfully!")
        print("\nNext steps:")
        print("1. Run: python -m mirrorcore --init-assessment")
        print("2. Complete the interactive questionnaire")
        print("3. Start using Mirrorcore: python -m mirrorcore")
    else:
        print("\n❌ Installation validation failed")
        sys.exit(1)


def main():
    """Main initialization function."""
    print("Mirrorcore Project Initialization")
    print("=" * 40)
    
    # Create data directory
    data_dir = create_data_directory()
    
    # Setup database
    setup_database(data_dir)
    
    # Create configuration
    create_configuration(data_dir)
    
    # Create initial profile placeholder
    create_initial_profile(data_dir)
    
    # Validate installation
    validate_installation(data_dir)


if __name__ == "__main__":
    main()
