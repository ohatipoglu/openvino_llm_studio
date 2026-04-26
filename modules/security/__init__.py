"""
modules/security/__init__.py
Security utilities.
"""

class PromptGuard:
    """Simple prompt security guard."""

    def __init__(self, strict_mode=False):
        self.strict_mode = strict_mode
        self.DANGEROUS_PATTERNS = ["ignore previous", "jailbreak", "bypass"]

    def validate(self, prompt):
        """Validate prompt for security."""
        issues = []
        risk_level = "low"

        if len(prompt) > 50000:
            issues.append("Prompt too long")
            risk_level = "medium"

        prompt_lower = prompt.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in prompt_lower:
                issues.append(f"Pattern: {pattern}")
                risk_level = "high"

        is_safe = risk_level != "high" or not self.strict_mode

        return type('SecurityCheck', (), {
            'is_safe': is_safe,
            'issues': issues,
            'sanitized_prompt': prompt[:50000],
            'risk_level': risk_level
        })()

    def safe_format(self, template, user_content, field_name="user_input"):
        """Safely format template."""
        safe_content = user_content[:50000]
        return template.format(**{field_name: f"\n<<<{field_name}>>>\n{safe_content}\n<<<end>>>\n"})


class MessageSeparator:
    """Role-based message separator."""

    def format_messages(self, messages):
        """Format messages with roles."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result.append(f"{role.upper()}: {content}")
        return "\n".join(result)


__all__ = ["PromptGuard", "MessageSeparator"]
