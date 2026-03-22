"""
Quick test script for the /match endpoint.

Usage: python -m scripts.test_match
Requires the server to be running: uvicorn app.main:app
"""

from __future__ import annotations

import requests

BASE_URL = "http://localhost:8000"


def test_match(feeling: str, k: int = 3) -> None:
    print(f"\n🎵 Testing: \"{feeling}\" (k={k})")
    print("-" * 50)

    response = requests.post(
        f"{BASE_URL}/match",
        json={"feeling": feeling, "k": k},
    )

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return

    data = response.json()
    print(f"🎯 Matches for: \"{data['feeling']}\"\n")

    for i, match in enumerate(data["matches"], 1):
        print(f"  {i}. {match['path']}")
        print(f"     Score: {match['score']:.4f}")

    if not data["matches"]:
        print("  No matches found.")


if __name__ == "__main__":
    test_match("dark and moody", k=3)
    test_match("happy upbeat summer vibes", k=3)
    test_match("chill lo-fi beats to relax to", k=5)
    test_match("confident and swagger with a heavy rhythmic bounce", k=3)
    test_match("aggressive and furious with raw screaming energy", k=3)
    print()
