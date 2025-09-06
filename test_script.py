#!/usr/bin/env python3
"""
Quick test script for the loader drop-off monitor.
Run this to test different scenarios easily.
"""

import os
import subprocess
import sys

def run_test(dataset_name, dataset_id, threshold=15):
    """Run a test with specific parameters."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {dataset_name} (Threshold: {threshold}%)")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env.update({
        'GOOGLE_APPLICATION_CREDENTIALS': '/Users/mukhtadir/Envs/MCPs/tata_mcp_bq_server.json',
        'BQ_PROJECT': 'tata-new-experience',
        'BQ_TABLE': dataset_id,
        'THRESHOLD_PCT': str(threshold),
        'MOCK_MATTERMOST': 'true'
    })
    
    result = subprocess.run([
        sys.executable, 'alert_loader_dropoff.py'
    ], env=env, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """Run all tests."""
    print("🚀 Loader Drop-off Alert - Test Suite")
    print("This will test both Harrier and Curvv datasets with different thresholds.\n")
    
    # Test Harrier
    run_test("Harrier", "analytics_490128245", 15)
    run_test("Harrier (High Threshold)", "analytics_490128245", 30)
    
    # Test Curvv
    run_test("Curvv", "analytics_452739489", 15)
    run_test("Curvv (High Threshold)", "analytics_452739489", 30)
    
    print(f"\n{'='*60}")
    print("✅ All tests completed!")
    print("📝 Note: All tests run in MOCK mode - no actual alerts sent.")
    print("🔧 To use with real Mattermost: Set MOCK_MATTERMOST=false and provide MATTERMOST_WEBHOOK")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
