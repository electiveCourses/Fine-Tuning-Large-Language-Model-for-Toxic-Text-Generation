import re

def remove_pii(text: str) -> str:
    """Remove personally identifiable information"""
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSNs
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # IPs
    ]
    for pattern in patterns:
        text = re.sub(pattern, '[REDACTED]', text)
    return text