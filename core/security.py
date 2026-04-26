"""
core/security.py

Security utilities for prompt sanitization and injection protection.

Features:
- Prompt length validation
- Dangerous pattern detection
- Input sanitization for templates
- Role-based message separation
- SQL injection prevention for database logging
"""

import re
import json
import logging
from typing import Optional
from dataclasses import dataclass

from core.constants import SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of prompt sanitization check."""
    is_safe: bool
    sanitized_text: str
    issues: list[str]
    risk_level: str  # 'low', 'medium', 'high', 'critical'


class PromptSanitizer:
    """
    Sanitize and validate user prompts to prevent injection attacks.
    
    Security measures:
    1. Length limiting
    2. Dangerous pattern detection
    3. Control character removal
    4. Template injection prevention
    """
    
    def __init__(self):
        self._config = SecurityConfig()
        self._dangerous_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> list[re.Pattern]:
        """Compile regex patterns for dangerous content detection."""
        patterns = []
        
        # Prompt injection attempts
        injection_patterns = [
            r"ignore\s+(previous|all)\s+(instructions|rules)",
            r"bypass\s+(security|filters|restrictions)",
            r"jailbreak|break\s+the\s+rules",
            r"system\s+prompt|initial\s+instructions",
            r"forget\s+(everything|all|your)\s+(instructions|rules)",
            r"act\s+as\s+(another\s+)?(ai|assistant|model)",
            r"output\s+(your|the)\s+(system|initial|prompt)",
            r"print\s+(your|the)\s+(instructions|prompt)",
        ]
        
        # Template injection
        template_patterns = [
            r"\{\{.*\}\}",  # Jinja2/Mustache
            r"\$\{.*\}",   # JavaScript template literals
            r"<%.*%>",     # ERB/ASP
            r"\{0\}",      # Python format
        ]
        
        for pattern in injection_patterns + template_patterns:
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        return patterns
    
    def sanitize(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> SanitizationResult:
        """
        Sanitize user input text.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length (default: from config)
        
        Returns:
            SanitizationResult with safety assessment
        """
        issues = []
        risk_level = "low"
        
        # Length check
        max_len = max_length or self._config.MAX_PROMPT_LENGTH
        if len(text) > max_len:
            issues.append(f"Text exceeds maximum length ({len(text)}/{max_len})")
            text = text[:max_len]
            risk_level = "medium"
        
        # Control character removal (except newlines and tabs)
        original_text = text
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        if text != original_text:
            issues.append("Removed control characters")
        
        # Dangerous pattern detection
        for pattern in self._dangerous_patterns:
            if pattern.search(text):
                issues.append(f"Dangerous pattern detected: {pattern.pattern}")
                risk_level = "high"
        
        # Unicode normalization
        text = text.strip()
        
        # Determine final risk level
        if len(issues) >= 3 or "critical" in [risk_level]:
            risk_level = "critical"
        elif len(issues) >= 2:
            risk_level = "high"
        elif len(issues) >= 1:
            risk_level = "medium"
        
        return SanitizationResult(
            is_safe=(risk_level not in ("high", "critical")),
            sanitized_text=text,
            issues=issues,
            risk_level=risk_level
        )
    
    def is_safe(self, text: str) -> bool:
        """Quick safety check without full sanitization."""
        result = self.sanitize(text)
        return result.is_safe


class TemplateManager:
    """
    Secure template management with variable separation.
    
    Prevents prompt injection by keeping user input separate from
    template structure.
    """
    
    def __init__(self):
        self._sanitizer = PromptSanitizer()
    
    def safe_format(
        self,
        template: str,
        variables: dict[str, str],
        escape_special: bool = True
    ) -> str:
        """
        Safely format template with variables.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values
            escape_special: Whether to escape special characters
        
        Returns:
            Formatted template string
        """
        # Sanitize all variables
        sanitized_vars = {}
        for key, value in variables.items():
            result = self._sanitizer.sanitize(value)
            
            if not result.is_safe:
                logger.warning(
                    f"Unsafe variable '{key}': {result.issues}"
                )
                # Still use sanitized version but log warning
            
            sanitized_vars[key] = result.sanitized_text
            
            if escape_special:
                # Escape template-like patterns
                sanitized_vars[key] = self._escape_template_chars(
                    sanitized_vars[key]
                )
        
        # Format template
        try:
            return template.format(**sanitized_vars)
        except KeyError as e:
            logger.error(f"Template variable not found: {e}")
            return template
    
    def _escape_template_chars(self, text: str) -> str:
        """Escape characters that could be interpreted as template syntax."""
        # Escape common template delimiters
        escapes = {
            '{': '{{',
            '}': '}}',
            '{{{': '{{{{',
            '}}}': '}}}}',
        }
        
        result = text
        for old, new in escapes.items():
            result = result.replace(old, new)
        
        return result


class MessageSeparator:
    """
    Separate and validate message roles in chat conversations.
    
    Prevents role confusion attacks where user tries to impersonate
    system or assistant roles.
    """
    
    ROLE_PREFIXES = {
        "system": "