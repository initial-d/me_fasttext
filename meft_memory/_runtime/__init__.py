"""Source-checkout bridge; installed wheels map this package directly to tools/."""
from pathlib import Path

_checkout_tools = Path(__file__).resolve().parents[2] / "tools"
if _checkout_tools.is_dir():
    __path__.append(str(_checkout_tools))
