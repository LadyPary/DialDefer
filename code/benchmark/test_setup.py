"""
Test setup and validation for the experiment system.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from common import create_client, chat
        print("common.api_client imported")
        
        from common import load_jsonl, extract_json_from_response, GROUND_TRUTHS
        print("utils imported")
        
        from prompts import c1_TF_factual_question_prompt, c2_convo_judge_prompt
        print("prompts imported")
        
        from run_experiment import run_experiment
        print("run_experiment imported")
        
        from analyze_jsonl import analyze_file
        print("analyze_jsonl imported")
        
        print("\nAll imports successful!")
        return True
    except Exception as e:
        print(f"\nImport failed: {e}")
        return False


def test_env():
    """Test environment variables."""
    print("\nTesting environment variables...")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"OPENROUTER_API_KEY set (length: {len(api_key)})")
    else:
        print("OPENROUTER_API_KEY not set")
    
    return api_key is not None


if __name__ == '__main__':
    print("=" * 70)
    print("EXPERIMENT SYSTEM SETUP TEST")
    print("=" * 70)
    
    imports_ok = test_imports()
    env_ok = test_env()
    
    print("\n" + "=" * 70)
    if imports_ok and env_ok:
        print("All tests passed! System is ready.")
    else:
        print("Some tests failed. Check errors above.")
    print("=" * 70)
