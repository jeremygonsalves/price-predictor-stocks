#!/usr/bin/env python3
"""
Setup script for the Real-Time Stock Prediction System
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_reddit_credentials():
    """Create Reddit credentials files template"""
    cred_files = ["client_id.txt", "client_secret.txt", "pw.txt"]
    
    for file_name in cred_files:
        if os.path.exists(file_name):
            print(f"⚠️  {file_name} already exists, skipping...")
            continue
        
        with open(file_name, 'w') as f:
            if file_name == "client_id.txt":
                f.write("YOUR_REDDIT_CLIENT_ID")
            elif file_name == "client_secret.txt":
                f.write("YOUR_REDDIT_CLIENT_SECRET")
            elif file_name == "pw.txt":
                f.write("YOUR_REDDIT_PASSWORD")
        
        print(f"📝 Created {file_name} template")
    
    print("   Please edit these files with your Reddit API credentials")
    print("   Get credentials from: https://www.reddit.com/prefs/apps")
    print("   Note: You must provide your own Reddit credentials in the secrets/ directory")

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "data", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("📁 Created necessary directories")

def test_installation():
    """Test if the installation works"""
    print("Testing installation...")
    try:
        import pandas
        import numpy
        import yfinance
        import tensorflow
        import sklearn
        import textblob
        import schedule
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Real-Time Stock Prediction System")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Create directories
    create_directories()
    
    # Create Reddit credentials template
    create_reddit_credentials()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        return
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit config.json to customize your settings")
    print("2. Edit reddit_credentials.json with your Reddit API credentials (optional)")
    print("3. Run: python src/enhanced_stock_alert.py")
    print("\nFor help, see the README.md file")

if __name__ == "__main__":
    main() 