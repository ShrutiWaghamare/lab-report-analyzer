#!/usr/bin/env python3
"""
Quick test to verify the parser fix for Triglycerides bug
Tests the _extract_inline_value() method directly
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr.parser import ReportParser

# Test cases that should now work correctly
test_cases = [
    # Format A: value then range (should work before and after)
    ("Triglycerides 375 mg/dL 200-499", "Triglycerides", 375.0),
    
    # Format B: reference threshold then value (BUG - was picking 200, should pick 375)
    ("Triglycerides 200-499 375 mg/dL", "Triglycerides", 375.0),
    ("Triglycerides 200 499 375 mg/dL", "Triglycerides", 375.0),
    
    # Format C: with comparison operators
    ("Total Cholesterol <200 180 mg/dL", "Total Cholesterol", 180.0),
    ("LDL Cholesterol >150 165 mg/dL", "LDL Cholesterol", 165.0),
    
    # Format D: complex lipid profile
    ("Triglycerides 150 200-499 375 \u2265240", "Triglycerides", 375.0),
]

def test_parser():
    parser = ReportParser()
    print("=" * 80)
    print("PARSER FIX TEST - Testing _extract_inline_value()")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for remainder_str, canonical, expected_value in test_cases:
        try:
            # Call the fixed method directly
            value, unit, ref_range = parser._extract_inline_value(remainder_str, canonical)
            
            status = "✓ PASS" if value == expected_value else "✗ FAIL"
            if value == expected_value:
                passed += 1
            else:
                failed += 1
            
            print(f"\n{status}")
            print(f"  Input:    {remainder_str}")
            print(f"  Expected: {expected_value}")
            print(f"  Got:      {value}")
            if unit or ref_range:
                print(f"  Unit:     {unit}")
                print(f"  Ref Range: {ref_range}")
        
        except Exception as e:
            failed += 1
            print(f"\n✗ ERROR")
            print(f"  Input: {remainder_str}")
            print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)
