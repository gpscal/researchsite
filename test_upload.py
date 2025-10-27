#!/usr/bin/env python3
"""Test script to verify PDF upload functionality after fixing datetime serialization issue."""

import requests
import os
from pathlib import Path

def test_pdf_upload():
    """Test PDF upload to verify the datetime serialization fix."""
    # Use one of the existing uploaded PDFs for testing
    uploads_dir = Path("data/uploads")
    pdf_files = list(uploads_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in uploads directory")
        return

    # Use the first PDF file for testing
    test_pdf = pdf_files[0]
    print(f"Testing upload with: {test_pdf.name}")

    # Flask app should be running on localhost:8080
    url = "http://localhost:8080/api/rag/upload-pdf"

    try:
        with open(test_pdf, "rb") as f:
            files = {"file": (test_pdf.name, f, "application/pdf")}
            response = requests.post(url, files=files, timeout=60)

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', False)}")
            if result.get('success'):
                print(f"Document ID: {result.get('document_id')}")
                print(f"Pages: {result.get('page_count')}")
                print(f"Chunks: {result.get('chunks')}")
                print("✅ PDF upload successful - datetime serialization fixed!")
            else:
                print(f"❌ Upload failed: {result.get('error')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the Flask app running on localhost:8080?")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_pdf_upload()

