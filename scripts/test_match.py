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
    test_match(
        "Hard-hitting aggressive rap with deep booming bass, rattling hi-hats, "
        "and a gritty, menacing beat that sounds like it belongs in a dark alley "
        "at two in the morning, raw and unapologetic with heavy 808s shaking the speakers",
        k=3,
    )
    test_match(
        "Bright and radiant with warm acoustic guitars, hand claps, and a carefree "
        "whistling melody that feels like sunshine pouring through an open window on "
        "a lazy Sunday morning, pure joy and lighthearted energy from start to finish",
        k=3,
    )
    test_match(
        "Crushing distorted electric guitars with relentless double-kick drumming, "
        "guttural screaming vocals, and a wall of feedback that feels like standing "
        "in front of a massive amplifier stack while the floor shakes beneath you",
        k=3,
    )
    test_match(
        "Eerie and unsettling with dissonant strings, creeping low-frequency drones, "
        "sudden jarring stabs of sound, and an atmosphere so tense and suffocating it "
        "feels like something is watching you from the darkest corner of the room",
        k=3,
    )
    test_match(
        "Tender and intimate with a soft piano, gentle fingerpicked guitar, and a "
        "warm breathy vocal floating over lush strings, delicate and vulnerable like "
        "whispering sweet nothings in the quiet stillness of a candlelit room",
        k=3,
    )
    print()
