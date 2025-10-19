#!/usr/bin/env python3
"""
Simple test script to verify the AI Sewer Pipe Analyzer Service is working
"""

import requests
import json
import time

SERVICE_URL = "http://127.0.0.1:8766"

def test_service():
    """Test all service endpoints"""
    print("Testing AI Sewer Pipe Analyzer Service...")
    print(f"Service URL: {SERVICE_URL}")
    print("=" * 50)

    # Test 1: Health check
    try:
        print("1. Testing health check...")
        response = requests.get(f"{SERVICE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check: {data}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    # Test 2: Status check
    try:
        print("2. Testing status endpoint...")
        response = requests.get(f"{SERVICE_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: Models loaded = {data['models_loaded']}")
            print(f"   📊 OCR cache size: {data['ocr_cache_size']}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Status check error: {e}")
        return False

    # Test 3: Cache info
    try:
        print("3. Testing cache info...")
        response = requests.get(f"{SERVICE_URL}/cache_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Cache info: {data}")
        else:
            print(f"   ❌ Cache info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cache info error: {e}")
        return False

    print("\n✅ All tests passed! Service is ready.")
    print("\n📋 Available endpoints:")
    print(f"   Health check: {SERVICE_URL}/")
    print(f"   Status: {SERVICE_URL}/status")
    print(f"   Cache info: {SERVICE_URL}/cache_info")
    print(f"   Analyze video: {SERVICE_URL}/analyze")
    print(f"   Clear cache: {SERVICE_URL}/clear_cache")

    return True

if __name__ == "__main__":
    success = test_service()
    if not success:
        print("\n❌ Service tests failed. Make sure the service is running:")
        print("   python service.py")
        exit(1)
    else:
        print("\n🎉 Service is ready for video analysis!")
