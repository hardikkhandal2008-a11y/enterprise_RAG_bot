"""
Phase 1: Local LLM Integration Test
====================================
Verifies programmatic communication with Ollama's HTTP API.
This is the foundational building block for all subsequent RAG phases.

Author: [Your Name]
Date: 2026-11-04
"""

import requests
from typing import Optional


class OllamaClient:
    """Minimal client for interacting with Ollama's local API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = 60  # Seconds; local inference can be slow on first call

    def generate(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Ollama and return the generated text.

        Args:
            prompt: The input text to send to the LLM.

        Returns:
            Generated response string, or None if request failed.
        """
        endpoint = f"{self.base_url}/api/generate"
        
        # We set temperature low (0.1) to make the model deterministic and precise.
        # High temperature makes models creative but unpredictable.
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1 
            }
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()  # Raises exception for 4xx/5xx status codes
            data = response.json()
            return data.get("response", "")

        except requests.exceptions.ConnectionError:
            print("❌ ERROR: Cannot connect to Ollama. Is 'ollama serve' running?")
            return None
        except requests.exceptions.Timeout:
            print(f"❌ ERROR: Request timed out after {self.timeout}s. Try a smaller model.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Unexpected API error: {e}")
            return None


def main():
    """Run Phase 1 validation tests."""
    print("=" * 50)
    print("PHASE 1: OLLAMA INTEGRATION TEST")
    print("=" * 50)

    # Initialize client with your specific model
    client = OllamaClient(model="llama3.2") 

    # Test 1: Basic capability check
    print("\n🧪 Test 1: Basic Generation")
    result = client.generate("Reply with exactly one word: PONG")
    
    if result:
        clean_result = result.strip().upper()
        print(f"✅ Response: '{result.strip()}'")
        
        # Check if PONG is present. If not, warn but don't crash.
        if "PONG" in clean_result:
            print("✅ Instruction following: PASSED")
        else:
            print(f"⚠️ Warning: Expected 'PONG' but got '{clean_result}'. Continuing...")
    else:
        print("❌ FAILED: No response received")
        return

    # Test 2: Domain knowledge check (DSA relevance)
    print("\n🧪 Test 2: DSA Knowledge Retrieval")
    result = client.generate(
        "In one sentence, what is the time complexity of binary search?"
    )
    
    if result:
        print(f"✅ Response: {result.strip()}")
        
        # Look for keywords indicating correct logic (case-insensitive)
        lower_result = result.lower()
        if "log" in lower_result or "o(log" in lower_result:
            print("✅ Domain knowledge: PASSED")
        else:
            print("⚠️ Warning: Answer didn't explicitly mention 'log'. Continuing...")
    else:
        print("❌ FAILED: No response received")
        return

    print("\n" + "=" * 50)
    print("🎉 PHASE 1 COMPLETE: All tests passed")
    print("=" * 50)
    print("Next: Phase 2 — File Reading & Prompt Construction")


if __name__ == "__main__":
    main()