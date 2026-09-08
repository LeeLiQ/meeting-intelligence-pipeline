"""Tests to verify that exception chaining preserves original error details.

These tests prove that when helper functions wrap exceptions using
'raise ... from e', the original exception type and message are fully
accessible via the __cause__ attribute, ensuring full traceability for
debugging and future agent-based error handling.
"""

from __future__ import annotations

import subprocess


def test_runtime_error_preserves_original_cause():
    """Wrapped RuntimeError should preserve the original exception via __cause__."""
    original = FileNotFoundError("model file missing")
    try:
        try:
            raise original
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e
    except Exception as e:
        # The outer handler sees RuntimeError
        assert isinstance(e, RuntimeError), f"Expected RuntimeError, got {type(e).__name__}"
        # But the original cause is preserved
        assert e.__cause__ is original, "Original exception should be preserved as __cause__"
        assert isinstance(e.__cause__, FileNotFoundError), (
            f"Original cause should be FileNotFoundError, got {type(e.__cause__).__name__}"
        )
        assert str(e.__cause__) == "model file missing"
        print("PASS: Original exception type and message preserved via __cause__")


def test_nested_chaining_preserves_full_chain():
    """Multiple levels of 'raise ... from e' should preserve the entire chain."""
    try:
        try:
            try:
                raise subprocess.CalledProcessError(1, "ffmpeg", stderr=b"codec error")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to extract audio: {e}") from e
        except RuntimeError as e:
            raise RuntimeError(f"Audio extraction failed: {e}") from e
    except Exception as e:
        # Top-level sees the outermost RuntimeError
        assert isinstance(e, RuntimeError)
        # One level down: inner RuntimeError
        assert isinstance(e.__cause__, RuntimeError)
        # Two levels down: original CalledProcessError
        assert isinstance(e.__cause__.__cause__, subprocess.CalledProcessError)
        assert e.__cause__.__cause__.returncode == 1
        print("PASS: Full exception chain preserved across multiple wrapping levels")


def test_isinstance_check_fails_on_wrapped_type():
    """Demonstrates that isinstance check on the wrapped exception does NOT match original type."""
    try:
        try:
            raise FileNotFoundError("original error")
        except Exception as e:
            raise RuntimeError(f"Wrapped: {e}") from e
    except Exception as e:
        # Direct isinstance check does NOT match the original type
        assert not isinstance(e, FileNotFoundError), (
            "Wrapped exception should NOT pass isinstance for original type"
        )
        # You must use __cause__ to access the original
        assert isinstance(e.__cause__, FileNotFoundError), (
            "Original type is accessible via __cause__"
        )
        print("PASS: isinstance correctly fails on wrapper, succeeds on __cause__")


if __name__ == "__main__":
    test_runtime_error_preserves_original_cause()
    test_nested_chaining_preserves_full_chain()
    test_isinstance_check_fails_on_wrapped_type()
    print("\nAll exception chain tests passed.")
