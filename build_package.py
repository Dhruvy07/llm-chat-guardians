import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def clean_build():
    """Clean build artifacts"""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ["build", "dist", "ai_agents_chatbot.egg-info", "ai_agents_enterprise.egg-info"]
    
    for pattern in dirs_to_clean:
        if os.path.exists(pattern):
            if os.path.isdir(pattern):
                shutil.rmtree(pattern)
                print(f"  Removed directory: {pattern}")
            else:
                os.remove(pattern)
                print(f"  Removed file: {pattern}")
    
    # Handle glob patterns separately
    import glob
    for pattern in glob.glob("*.egg-info"):
        if os.path.exists(pattern):
            if os.path.isdir(pattern):
                shutil.rmtree(pattern)
                print(f"  Removed directory: {pattern}")
            else:
                os.remove(pattern)
                print(f"  Removed file: {pattern}")
    
    print("✅ Build artifacts cleaned")
    return True

def install_build_tools():
    """Install required build tools"""
    return run_command(
        "pip install build twine",
        "Installing build tools"
    )

def test_imports():
    """Test that all main components can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        # Test main package import
        from agentic_chatbot import (
            SecurityAgent, 
            ContextAgent, 
            ModelSelectionAgent,
            AdvancedConversationAgent,
            create_basic_agent,
            create_openai_agent,
            create_enterprise_agent
        )
        print("✅ Main package imports successful")
        
        # Test individual agent imports
        from agentic_chatbot.security_agent import SecurityAgent
        from agentic_chatbot.context_agent import ContextAgent
        from agentic_chatbot.model_selection_agent import ModelSelectionAgent
        from agentic_chatbot.advanced_conversation_agent import AdvancedConversationAgent
        print("✅ Individual agent imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_agent():
    """Test basic agent functionality (no API key required)"""
    print("🤖 Testing basic agent...")
    
    try:
        from agentic_chatbot import create_basic_agent
        
        # Create basic agent
        agent = create_basic_agent()
        print("✅ Basic agent created successfully")
        
        # Test conversation
        response = agent.invoke("test_user", "Hello! What can you do?")
        print(f"✅ Basic agent response: {response[:100]}...")
        
        # Test conversation history
        history = agent.get_conversation_history("test_user")
        print(f"✅ Conversation history length: {len(history)}")
        
        return True
    except Exception as e:
        print(f"❌ Basic agent error: {e}")
        return False

def test_security_agent():
    """Test security agent functionality"""
    print("🛡️ Testing security agent...")
    
    try:
        from agentic_chatbot import SecurityAgent
        
        # Create security agent
        security_agent = SecurityAgent()
        print("✅ Security agent created successfully")
        
        # Test security analysis
        result = security_agent.analyze_security("Hello, how are you?")
        print(f"✅ Security analysis result: {result['threat_level']}")
        
        return True
    except Exception as e:
        print(f"❌ Security agent error: {e}")
        return False

def test_context_agent():
    """Test context agent functionality"""
    print("🧠 Testing context agent...")
    
    try:
        from agentic_chatbot import ContextAgent
        
        # Create context agent
        context_agent = ContextAgent()
        print("✅ Context agent created successfully")
        
        # Test context analysis
        result = context_agent.analyze_context("What is the weather like?")
        print(f"✅ Context analysis result: {result['relevance_level']}")
        
        return True
    except Exception as e:
        print(f"❌ Context agent error: {e}")
        return False

def test_model_selection_agent():
    """Test model selection agent functionality"""
    print("🎯 Testing model selection agent...")
    
    try:
        from agentic_chatbot import ModelSelectionAgent
        
        # Create model selection agent
        model_agent = ModelSelectionAgent()
        print("✅ Model selection agent created successfully")
        
        # Test model selection
        result = model_agent.select_model("What is artificial intelligence?")
        print(f"✅ Model selection result: {result['selected_model']}")
        
        return True
    except Exception as e:
        print(f"❌ Model selection agent error: {e}")
        return False

def test_package_info():
    """Test package information"""
    print("📦 Testing package information...")
    
    try:
        import agentic_chatbot
        
        print(f"✅ Package version: {agentic_chatbot.__version__}")
        print(f"✅ Package author: {agentic_chatbot.__author__}")
        print(f"✅ Package license: {agentic_chatbot.__license__}")
        
        return True
    except Exception as e:
        print(f"❌ Package info error: {e}")
        return False

def run_tests():
    """Run package tests"""
    print("🧪 Running package tests...")
    
    tests = [
        test_imports,
        test_package_info,
        test_basic_agent,
        test_security_agent,
        test_context_agent,
        test_model_selection_agent,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your package is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

def build_package():
    """Build the package"""
    return run_command(
        "python -m build",
        "Building package"
    )

def test_built_package():
    """Test the built package"""
    print("🧪 Testing built package...")
    
    # Find the wheel file
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory found")
        return False
    
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel files found in dist/")
        return False
    
    wheel_file = wheel_files[0]
    print(f"📦 Testing wheel: {wheel_file}")
    
    # Test installation
    return run_command(
        f"pip install {wheel_file}",
        f"Installing {wheel_file}"
    )

def check_package_structure():
    """Check package structure"""
    print("📁 Checking package structure...")
    
    required_files = [
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "agentic_chatbot/__init__.py",
        "agentic_chatbot/security_agent.py",
        "agentic_chatbot/context_agent.py",
        "agentic_chatbot/model_selection_agent.py",
        "agentic_chatbot/advanced_conversation_agent.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ Package structure looks good")
    return True

def main():
    """Main build process"""
    print("🚀 AI Agents Enterprise Package Builder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("❌ Please run this script from the package root directory")
        return 1
    
    steps = [
        ("Checking package structure", check_package_structure),
        ("Cleaning build artifacts", clean_build),
        ("Installing build tools", install_build_tools),
        ("Running tests", run_tests),
        ("Building package", build_package),
        ("Testing built package", test_built_package),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_func():
            print(f"❌ Build failed at step: {step_name}")
            return 1
    
    print("\n" + "=" * 50)
    print("🎉 Package build completed successfully!")
    print("\n📦 Built files:")
    
    dist_dir = Path("dist")
    if dist_dir.exists():
        for file in dist_dir.iterdir():
            print(f"  - {file}")
    
    print("\n🚀 Next steps:")
    print("1. Test the package: python build_package.py")
    print("2. Upload to Test PyPI: twine upload --repository testpypi dist/*")
    print("3. Upload to PyPI: twine upload dist/*")
    print("4. Install from PyPI: pip install agentic-chatbot")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
