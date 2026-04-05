#!/usr/bin/env python3
"""
Test script to extract and verify Abhishek Datta's lab report
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load .env file FIRST before importing modules that use environment variables
load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.lab_crew import LabReportCrew

# Path to Abhishek Datta's PDF
PDF_PATH = r"e:\AIML Projects\lab_analyzer\samples\input\apollo\complete report of abhishek datta.pdf"

def main():
    print("="*80)
    print("Testing Extraction for: Abhishek Datta")
    print("="*80)
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        return
    
    print(f"📄 PDF found: {PDF_PATH}\n")
    
    # Initialize crew WITH LLM enabled (will use Azure if keys are set in .env)
    crew = LabReportCrew(use_llm=True)
    
    # Run analysis
    print("⏳ Extracting and analyzing...\n")
    result = crew.run(file_path=PDF_PATH)
    
    # Display results
    print("\n" + "="*80)
    print("EXTRACTION RESULTS")
    print("="*80)
    
    # Patient info
    patient_info = result.get("patient_info", {})
    if patient_info:
        print("\n👤 PATIENT INFO:")
        for key, value in patient_info.items():
            print(f"  {key:20s}: {value}")
    
    # Test results
    tests = result.get("test_results", [])
    if tests:
        print("\n🔬 TEST RESULTS:")
        print(f"  Total tests found: {len(tests)}\n")
        
        # Print first test structure to see what keys are available
        if tests:
            print(f"  Sample test structure: {tests[0]}\n")
        
        for test in tests[:10]:  # Show first 10
            test_name = test.get('test_name') or test.get('name', 'Unknown')
            test_value = test.get('value', 'N/A')
            test_unit = test.get('unit', '')
            test_ref = test.get('reference_range', '')
            status = "✓" if not test.get("abnormal") else "⚠"
            print(f"  {status} {test_name:30s} | {str(test_value):10s} {test_unit:10s} | Ref: {test_ref}")
        
        if len(tests) > 10:
            print(f"\n  ... and {len(tests) - 10} more tests")
    
    # Rule-based analysis
    rule_analysis = result.get("rule_analysis")
    if rule_analysis:
        print("\n📋 RULE-BASED ANALYSIS:")
        
        if rule_analysis.get("abnormal_tests"):
            print(f"\n  Abnormal Tests ({len(rule_analysis['abnormal_tests'])}):")
            for test in rule_analysis["abnormal_tests"][:5]:
                severity = test.get("severity", "unknown")
                print(f"    - {test['test_name']:30s} [{severity:6s}] {test['value']} {test['unit']}")
        
        if rule_analysis.get("detected_conditions"):
            print(f"\n  Detected Conditions ({len(rule_analysis['detected_conditions'])}):")
            for cond in rule_analysis["detected_conditions"][:5]:
                print(f"    - {cond.get('name', 'Unknown'):30s} [severity: {cond.get('severity', 'N/A')}]")
    
    # Validation notes
    validation_notes = result.get("validation_notes", [])
    if validation_notes:
        print("\n⚠️  VALIDATION NOTES:")
        for note in validation_notes:
            print(f"  - {note}")
    
    # Errors
    errors = result.get("errors", [])
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
    
    # Save full JSON output
    output_file = r"e:\AIML Projects\lab_analyzer\samples\output\abhishek_datta_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        "file_path": PDF_PATH,
        "patient_info": patient_info,
        "test_results": tests,
        "rule_analysis": rule_analysis,
        "validation_notes": validation_notes,
        "errors": errors
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Full results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
