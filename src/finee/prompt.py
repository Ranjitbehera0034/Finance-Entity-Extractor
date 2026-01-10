"""
FinEE Prompt - LLM prompt templates for targeted extraction.

Uses field-specific prompts instead of generic extraction for better accuracy.
"""

from typing import List, Optional, Dict


# Base system prompt
SYSTEM_PROMPT = """You are a financial data extraction expert. Extract structured information from Indian banking transaction messages accurately. Always output valid JSON."""


# Field-specific targeted prompts (Tier 3)
TARGETED_PROMPTS = {
    'merchant': """Extract ONLY the merchant/vendor name from this transaction. 
Reply with just the name, nothing else. If you cannot determine the merchant, reply "unknown".

Transaction: {text}

Merchant name:""",

    'category': """What category does this transaction belong to?
Reply with ONE word from: food, shopping, transport, utilities, entertainment, transfer, salary, investment, healthcare, education, other

Transaction: {text}

Category:""",

    'date': """Extract ONLY the transaction date from this text.
Reply in DD-MM-YYYY format. If no date found, reply "unknown".

Transaction: {text}

Date:""",

    'reference': """Extract ONLY the transaction reference/UTR number from this text.
Reply with just the reference number (typically 12-16 digits). If not found, reply "unknown".

Transaction: {text}

Reference:""",

    'amount': """Extract ONLY the transaction amount from this text.
Reply with just the number (e.g., 2500.00). If not found, reply "unknown".

Transaction: {text}

Amount:""",

    'type': """Is this transaction a DEBIT (money going out) or CREDIT (money coming in)?
Reply with just "debit" or "credit".

Transaction: {text}

Type:""",

    'account': """Extract ONLY the account number (or last 4 digits) from this text.
Reply with just the number. If not found, reply "unknown".

Transaction: {text}

Account:""",

    'vpa': """Extract ONLY the UPI VPA (Virtual Payment Address) from this text.
A VPA looks like "username@bankcode" (e.g., swiggy@ybl).
Reply with just the VPA. If not found, reply "unknown".

Transaction: {text}

VPA:""",
}


# Full extraction prompt (fallback)
FULL_EXTRACTION_PROMPT = """Extract all financial entities from this Indian banking transaction message.
Return a JSON object with these fields (use null if not found):
- amount: transaction amount as a number
- type: "debit" or "credit"
- date: in DD-MM-YYYY format
- account: account number (or last 4 digits)
- reference: UPI/transaction reference number
- vpa: UPI Virtual Payment Address (e.g., swiggy@ybl)
- merchant: merchant/vendor name
- category: one of [food, shopping, transport, utilities, entertainment, transfer, salary, investment, healthcare, education, other]

Transaction:
{text}

JSON output:"""


# Chat-style prompt (for models that support chat format)
CHAT_EXTRACTION_TEMPLATE = {
    "system": SYSTEM_PROMPT,
    "user": """Extract financial entities from this transaction:

{text}

Return JSON with: amount, type, date, account, reference, vpa, merchant, category"""
}


def get_targeted_prompt(field: str, text: str) -> str:
    """
    Get a targeted prompt for extracting a specific field.
    
    Args:
        field: Field name to extract
        text: Transaction text
        
    Returns:
        Formatted prompt string
    """
    if field not in TARGETED_PROMPTS:
        raise ValueError(f"Unknown field: {field}. Available: {list(TARGETED_PROMPTS.keys())}")
    
    return TARGETED_PROMPTS[field].format(text=text)


def get_multi_field_prompt(fields: List[str], text: str) -> str:
    """
    Get a prompt for extracting multiple specific fields.
    
    Args:
        fields: List of field names to extract
        text: Transaction text
        
    Returns:
        Formatted prompt string
    """
    if not fields:
        return get_full_extraction_prompt(text)
    
    field_descriptions = {
        'amount': 'amount (as a number)',
        'type': 'type ("debit" or "credit")',
        'date': 'date (DD-MM-YYYY format)',
        'account': 'account (number or last 4 digits)',
        'reference': 'reference (UPI/transaction ID)',
        'vpa': 'vpa (UPI address like user@bank)',
        'merchant': 'merchant (vendor name)',
        'category': 'category (food/shopping/transport/etc)',
    }
    
    fields_list = ', '.join(field_descriptions.get(f, f) for f in fields)
    
    prompt = f"""Extract ONLY these fields from the transaction: {fields_list}

Return a JSON object with only these fields. Use null if not found.

Transaction:
{text}

JSON output:"""
    
    return prompt


def get_full_extraction_prompt(text: str) -> str:
    """
    Get the full extraction prompt.
    
    Args:
        text: Transaction text
        
    Returns:
        Formatted prompt string
    """
    return FULL_EXTRACTION_PROMPT.format(text=text)


def get_chat_messages(text: str) -> List[Dict[str, str]]:
    """
    Get chat-format messages for models that support it.
    
    Args:
        text: Transaction text
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": CHAT_EXTRACTION_TEMPLATE["system"]},
        {"role": "user", "content": CHAT_EXTRACTION_TEMPLATE["user"].format(text=text)},
    ]


def parse_targeted_response(field: str, response: str) -> Optional[str]:
    """
    Parse the response from a targeted prompt.
    
    Args:
        field: Field name that was extracted
        response: Raw LLM response
        
    Returns:
        Cleaned field value or None
    """
    if not response:
        return None
    
    # Clean response
    cleaned = response.strip()
    
    # Handle "unknown" responses
    if cleaned.lower() in ('unknown', 'null', 'none', 'n/a', ''):
        return None
    
    # Remove quotes if present
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    
    return cleaned if cleaned else None
