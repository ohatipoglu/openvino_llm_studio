"""
utils/security.py

Security utilities for prompt sanitization, input validation, and rate limiting.
"""

import re
import time
import logging
from collections import defaultdict
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class PromptSanitizer:
    """
    Sanitizes user prompts to prevent injection attacks.
    
    Security concerns addressed:
    - Prompt injection (e.g., "ignore previous instructions")
    - System prompt leakage attempts
    - Role-playing jailbreaks
    """
    
    DANGEROUS_PATTERNS = [
        # Prompt injection
        r"ignore\s+(previous|all)\s+(instructions|rules)",
        r"bypass\s+(security|rules|filters)",
        r"jailbreak",
        r"dan\s*=",  # DAN jailbreak
        
        # System prompt extraction
        r"(what|show|tell)\s+(me|us)?\s*(your)?\s*(system|initial|first)\s*(prompt|instructions)",
        r"repeat\s+(the)?\s*(text|words|instructions)\s*(above|before)",
        r"output\s+(your)?\s*(system|configuration)\s*(prompt|settings)",
        
        # Role-playing attacks
        r"act\s+as\s+(a|an)?\s*(developer|admin|system)",
        r"pretend\s+(to)?\s*be\s+(a|an)?\s*(developer|admin)",
        r"you\s+are\s+now\s+(in|under)\s*(test|development)\s*mode",
        
        # Token manipulation
        r"