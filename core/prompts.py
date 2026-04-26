"""
core/prompts.py
Simple prompt templates.
"""

class BaseTemplates:
    @staticmethod
    def chain_of_thought(context: str = "", question: str = "") -> str:
        ctx = f"\n\nContext: {context}\n" if context else ""
        return f"{ctx}\n\nPlease answer step by step:\n{question}"
    
    @staticmethod
    def react(tool_desc: str, context: str = "", question: str = "") -> str:
        ctx = f"\nContext: {context}\n" if context else ""
        return (
            f"You are an AI assistant.\n"
            f"Tools: {tool_desc}\n"
            f"{ctx}\n"
            f"User: {question}\n"
            f"Assistant:"
        )
    
    @staticmethod
    def program_of_thought(context: str = "", problem: str = "") -> str:
        ctx = f"\nContext: {context}\n" if context else ""
        return f"{ctx}\n\nSolve this problem step by step:\n{problem}"
    
    @staticmethod
    def multi_chain_comparison(context: str = "", topic: str = "") -> str:
        ctx = f"\nContext: {context}\n" if context else ""
        return f"{ctx}\n\nCompare and contrast:\n{topic}"
    
    @staticmethod
    def summarize(context: str = "", topic: str = "") -> str:
        ctx = f"\nContext: {context}\n" if context else ""
        return f"{ctx}\n\nSummarize in 5 bullet points:\n{topic}"
    
    @staticmethod
    def predict(question: str = "") -> str:
        return question


class DomainTemplates:
    @staticmethod
    def banking_react(tool_desc: str, context: str = "", question: str = "") -> str:
        ctx = f"\nCustomer Info: {context}\n" if context else ""
        return (
            f"You are a BANKING ASSISTANT.\n"
            f"SECURITY RULES:\n"
            f"1. Verify all transactions\n"
            f"2. Never disclose customer info\n"
            f"Tools: {tool_desc}\n"
            f"{ctx}\n"
            f"Customer Request: {question}"
        )
    
    @staticmethod
    def research_react(tool_desc: str, context: str = "", question: str = "") -> str:
        ctx = f"\nExisting Info: {context}\n" if context else ""
        return (
            f"You are a RESEARCH ASSISTANT.\n"
            f"Tools: {tool_desc}\n"
            f"{ctx}\n"
            f"Research Question: {question}"
        )
    
    @staticmethod
    def code_assistant(context: str = "", task: str = "") -> str:
        ctx = f"\nContext: {context}\n" if context else ""
        return f"{ctx}\n\nProgramming Task:\n{task}\n\nProvide working code with explanation."


class ModelFormatters:
    @staticmethod
    def format_chatml(messages: list) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"