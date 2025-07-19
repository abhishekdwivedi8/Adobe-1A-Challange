#!/usr/bin/env python3
import os
import sys
import glob

def check_model_size():
    """Check if model size is within limits."""
    model_path = os.path.join('model', 'heading_model.joblib')
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return False
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"Model size: {size_mb:.2f} MB")
    if size_mb <= 200:
        print("[PASS] Model size is within limits (<= 200MB)")
        return True
    else:
        print("[FAIL] Model size exceeds 200MB limit!")
        return False

def check_docker_file():
    """Check if Dockerfile exists and has correct platform."""
    if not os.path.exists('Dockerfile'):
        print("❌ Dockerfile not found!")
        return False
    
    with open('Dockerfile', 'r') as f:
        content = f.read()
    
    if '--platform=linux/amd64' in content:
        print("[PASS] Dockerfile specifies correct platform (linux/amd64)")
        return True
    else:
        print("[WARN] Dockerfile should specify platform: FROM --platform=linux/amd64")
        return False

def check_requirements_file():
    """Check if requirements.txt exists and doesn't have network dependencies."""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    network_keywords = ['requests', 'urllib', 'http', 'ftp', 'socket', 'api', 'cloud']
    for keyword in network_keywords:
        if keyword in content.lower():
            print(f"[WARN] requirements.txt may contain network dependencies: {keyword}")
            return False
    
    print("[PASS] requirements.txt doesn't contain obvious network dependencies")
    return True

def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = ['app', 'model', 'input', 'output']
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    
    if missing_dirs:
        print(f"[FAIL] Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("[PASS] Project structure is correct")
    return True

def main():
    print("=== Docker Requirements Check ===\n")
    
    checks = [
        check_model_size(),
        check_docker_file(),
        check_requirements_file(),
        check_project_structure()
    ]
    
    print("\n=== Summary ===")
    if all(checks):
        print("[PASS] All checks passed! Project meets Docker requirements.")
        return 0
    else:
        print("[FAIL] Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())