#!/usr/bin/env python3
"""
Test file upload to the API directly using curl/requests-like approach
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test file path
TEST_PDF = r"e:\AIML Projects\lab_analyzer\samples\input\apollo\complete report of abhishek datta.pdf"
API_URL = "http://localhost:8000/analyze/file"

def test_file_upload():
    print("="*80)
    print("Testing File Upload to API")
    print("="*80)
    
    if not os.path.exists(TEST_PDF):
        print(f"❌ Test file not found: {TEST_PDF}")
        return
    
    print(f"📄 Test file: {TEST_PDF}")
    print(f"🌐 API endpoint: {API_URL}\n")
    
    try:
        with open(TEST_PDF, 'rb') as f:
            files = {'file': f}
            print("⏳ Uploading file to API...")
            response = requests.post(API_URL, files=files, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! API responded correctly.\n")
            result = response.json()
            
            # Show patient info
            patient = result.get("patient_info", {})
            print(f"👤 Patient: {patient.get('name')} | Age: {patient.get('age')} | Gender: {patient.get('gender')}")
            
            # Show test count
            tests = result.get("test_results", [])
            print(f"🔬 Tests extracted: {len(tests)}")
            
            # Show abnormal count
            abnormal = [t for t in tests if t.get('abnormal')]
            print(f"⚠️  Abnormal tests: {len(abnormal)}")
            
            # Show if Azure analysis was done
            if result.get("azure_analysis"):
                print(f"🤖 Azure analysis: ✅ Completed")
            else:
                print(f"🤖 Azure analysis: ❌ Not available")
            
            # Show errors if any
            if result.get("errors"):
                print(f"\n❌ Errors during processing:")
                for err in result["errors"]:
                    print(f"   - {err}")
            
        else:
            print(f"❌ ERROR! Status: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR: Cannot reach {API_URL}")
        print(f"   Make sure the server is running: python run.py")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("="*80)

if __name__ == "__main__":
    test_file_upload()
