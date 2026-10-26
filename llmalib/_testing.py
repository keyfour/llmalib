"""
llmalib/_testing.py

Test seam for the Rust extension: install a fake model-call transport
instead of patching `llmalib.reliability.retry.call` / httpx.

Usage in tests:

    from llmalib._testing import set_caller

    def fake_caller(messages, config):
        return '{"label": "positive", "confidence": 0.9}'

    set_caller(fake_caller)     # install
    set_caller(None)            # reset (restores the real HTTP transport)
"""

from llmalib._rust import set_caller

__all__ = ["set_caller"]
