#!/usr/bin/env python3
"""
Integration Test for SEAL-Inspired Agentic Tuner

This script tests the complete Strategist + Tactician integration
by simulating an optimization request through the CLI agent.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cli_ai.core.prompts import get_optimization_strategy_prompt
from src.cli_ai.tools.tools import optimize_workload
import asyncio

def test_strategy_generation():
    """Test 1: LLM prompt for strategy generation"""
    print("\n" + "="*80)
    print("TEST 1: Strategy Generation Prompt")
    print("="*80)
    
    workload = "PostgreSQL database with heavy transaction processing"
    prompt = get_optimization_strategy_prompt(workload)
    
    print(f"\nWorkload: {workload}")
    print(f"\nPrompt Length: {len(prompt)} characters")
    print("\nPrompt Preview (first 500 chars):")
    print(prompt[:500] + "...")
    
    # Check if prompt includes required elements
    required_elements = [
        "workload_name",
        "reward_metric",
        "benchmark_command",
        "action_space",
        "state_space",
        "training_config"
    ]
    
    all_present = all(elem in prompt for elem in required_elements)
    
    if all_present:
        print("\n✓ Prompt includes all required configuration elements")
        return True
    else:
        print("\n✗ Prompt missing some configuration elements")
        return False

async def test_tool_with_sample_config():
    """Test 2: optimize_workload tool with sample configuration"""
    print("\n" + "="*80)
    print("TEST 2: Tool Execution (Dry-Run)")
    print("="*80)
    
    # Sample configuration (simplified for testing)
    sample_config = {
        "workload_name": "Test Web Server",
        "reward_metric": "requests_per_second",
        "benchmark_command": "echo 'Requests per second: 1234.56'",
        "action_space": [
            {"param": "net.core.somaxconn", "min": 128, "max": 1024, "type": "int"}
        ],
        "state_space": [
            {"metric": "cpu_utilization", "source": "/proc/stat"}
        ],
        "training_config": {
            "total_timesteps": 100,  # Small number for testing
            "max_steps_per_episode": 5,
            "learning_rate": 0.0003
        }
    }
    
    print(f"\nTest Configuration:")
    print(json.dumps(sample_config, indent=2))
    
    try:
        # Call the tool
        result = await optimize_workload(
            workload_description="Test workload",
            config_json=json.dumps(sample_config)
        )
        
        if "result" in result:
            print("\n✓ Tool executed successfully")
            print(f"\nResult keys: {list(result['result'].keys())}")
            return True
        elif "error" in result:
            print(f"\n✗ Tool returned error: {result['error']}")
            return False
        else:
            print(f"\n? Unexpected result format: {result}")
            return False
            
    except Exception as e:
        print(f"\n✗ Tool execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_validation():
    """Test 3: Configuration validation"""
    print("\n" + "="*80)
    print("TEST 3: Configuration Validation")
    print("="*80)
    
    # Test with invalid config (missing required fields)
    invalid_config = {
        "workload_name": "Test"
        # Missing other required fields
    }
    
    print("\nTesting with invalid configuration...")
    
    async def run_test():
        result = await optimize_workload(
            workload_description="Test",
            config_json=json.dumps(invalid_config)
        )
        
        if "error" in result and "missing required fields" in result["error"].lower():
            print("✓ Invalid configuration correctly rejected")
            return True
        else:
            print(f"✗ Expected validation error, got: {result}")
            return False
    
    return asyncio.run(run_test())

def test_tool_schema():
    """Test 4: Tool schema validation"""
    print("\n" + "="*80)
    print("TEST 4: Tool Schema")
    print("="*80)
    
    from src.cli_ai.tools.tools import tools_schema
    
    # Find optimize_workload in schema
    opt_tool = None
    for tool in tools_schema:
        if tool.get("function", {}).get("name") == "optimize_workload":
            opt_tool = tool
            break
    
    if opt_tool:
        print("✓ optimize_workload found in tools_schema")
        print(f"\nTool description: {opt_tool['function']['description'][:100]}...")
        
        params = opt_tool['function']['parameters']['properties']
        required = opt_tool['function']['parameters']['required']
        
        print(f"\nParameters: {list(params.keys())}")
        print(f"Required: {required}")
        
        if 'workload_description' in params and 'config_json' in params:
            print("✓ All required parameters present")
            return True
        else:
            print("✗ Missing required parameters")
            return False
    else:
        print("✗ optimize_workload NOT found in tools_schema")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SEAL-INSPIRED AGENTIC TUNER - INTEGRATION TESTS")
    print("="*80)
    
    results = {}
    
    # Test 1: Strategy Generation Prompt
    results['strategy_prompt'] = test_strategy_generation()
    
    # Test 2: Tool Execution (skip if in CI/no system access)
    if not os.environ.get('CI'):
        results['tool_execution'] = asyncio.run(test_tool_with_sample_config())
    else:
        print("\n[Skipping tool execution test in CI environment]")
        results['tool_execution'] = None
    
    # Test 3: Configuration Validation
    results['config_validation'] = test_config_validation()
    
    # Test 4: Tool Schema
    results['tool_schema'] = test_tool_schema()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        
        print(f"{test_name:.<50} {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
