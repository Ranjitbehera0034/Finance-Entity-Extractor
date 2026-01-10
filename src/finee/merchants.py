"""
FinEE Merchants - Tier 2 VPA-to-Merchant and Merchant-to-Category mapping.

Rule-based mappings for enriching extracted entities with merchant names
and transaction categories.
"""

from typing import Optional, Dict, Tuple
import re


# VPA suffix to bank/app mapping
VPA_BANKS = {
    'ybl': 'PhonePe',
    'paytm': 'Paytm',
    'okaxis': 'Google Pay',
    'oksbi': 'Google Pay',
    'okhdfcbank': 'Google Pay',
    'axl': 'Google Pay',
    'ibl': 'ICICI Bank',
    'upi': 'Generic UPI',
    'apl': 'Amazon Pay',
    'fbl': 'Federal Bank',
    'icici': 'ICICI Bank',
    'hdfcbank': 'HDFC Bank',
    'sbi': 'SBI',
}


# Known merchant VPAs (exact match or prefix)
KNOWN_MERCHANTS = {
    # Food & Delivery
    'swiggy': ('Swiggy', 'food'),
    'zomato': ('Zomato', 'food'),
    'dominos': ("Domino's", 'food'),
    'pizzahut': ('Pizza Hut', 'food'),
    'mcdonalds': ("McDonald's", 'food'),
    'burgerking': ('Burger King', 'food'),
    'starbucks': ('Starbucks', 'food'),
    'kfc': ('KFC', 'food'),
    'subway': ('Subway', 'food'),
    'dunkin': ('Dunkin', 'food'),
    'blinkit': ('Blinkit', 'food'),
    'zepto': ('Zepto', 'food'),
    'bigbasket': ('BigBasket', 'food'),
    'instamart': ('Swiggy Instamart', 'food'),
    
    # Shopping
    'amazon': ('Amazon', 'shopping'),
    'flipkart': ('Flipkart', 'shopping'),
    'myntra': ('Myntra', 'shopping'),
    'ajio': ('Ajio', 'shopping'),
    'nykaa': ('Nykaa', 'shopping'),
    'meesho': ('Meesho', 'shopping'),
    'snapdeal': ('Snapdeal', 'shopping'),
    'tatacliq': ('Tata Cliq', 'shopping'),
    'reliance': ('Reliance', 'shopping'),
    'dmart': ('D-Mart', 'shopping'),
    'croma': ('Croma', 'shopping'),
    'vijaysales': ('Vijay Sales', 'shopping'),
    
    # Transport
    'uber': ('Uber', 'transport'),
    'ola': ('Ola', 'transport'),
    'rapido': ('Rapido', 'transport'),
    'irctc': ('IRCTC', 'transport'),
    'redbus': ('redBus', 'transport'),
    'makemytrip': ('MakeMyTrip', 'transport'),
    'goibibo': ('Goibibo', 'transport'),
    'yatra': ('Yatra', 'transport'),
    'cleartrip': ('Cleartrip', 'transport'),
    'easemytrip': ('EaseMyTrip', 'transport'),
    'metro': ('Metro', 'transport'),
    'fastag': ('FASTag', 'transport'),
    'iocl': ('Indian Oil', 'transport'),
    'bpcl': ('Bharat Petroleum', 'transport'),
    'hpcl': ('HP Petrol', 'transport'),
    
    # Utilities
    'jio': ('Jio', 'utilities'),
    'airtel': ('Airtel', 'utilities'),
    'vi': ('Vi', 'utilities'),
    'bsnl': ('BSNL', 'utilities'),
    'tatapower': ('Tata Power', 'utilities'),
    'adanigas': ('Adani Gas', 'utilities'),
    'mahanagar': ('Mahanagar Gas', 'utilities'),
    'bescom': ('BESCOM', 'utilities'),
    'electricity': ('Electricity', 'utilities'),
    'water': ('Water Bill', 'utilities'),
    'gas': ('Gas Bill', 'utilities'),
    
    # Entertainment
    'netflix': ('Netflix', 'entertainment'),
    'prime': ('Amazon Prime', 'entertainment'),
    'hotstar': ('Disney+ Hotstar', 'entertainment'),
    'spotify': ('Spotify', 'entertainment'),
    'bookmyshow': ('BookMyShow', 'entertainment'),
    'pvr': ('PVR', 'entertainment'),
    'inox': ('Inox', 'entertainment'),
    'youtube': ('YouTube', 'entertainment'),
    'zee5': ('Zee5', 'entertainment'),
    'sonyliv': ('SonyLiv', 'entertainment'),
    'jiocinema': ('JioCinema', 'entertainment'),
    
    # Healthcare
    'apollo': ('Apollo', 'healthcare'),
    'pharmeasy': ('PharmEasy', 'healthcare'),
    'netmeds': ('Netmeds', 'healthcare'),
    '1mg': ('1mg', 'healthcare'),
    'practo': ('Practo', 'healthcare'),
    'medplus': ('MedPlus', 'healthcare'),
    
    # Education
    'byjus': ("Byju's", 'education'),
    'unacademy': ('Unacademy', 'education'),
    'upgrad': ('upGrad', 'education'),
    'coursera': ('Coursera', 'education'),
    'udemy': ('Udemy', 'education'),
    'vedantu': ('Vedantu', 'education'),
    
    # Investment
    'zerodha': ('Zerodha', 'investment'),
    'groww': ('Groww', 'investment'),
    'upstox': ('Upstox', 'investment'),
    'paytmmoney': ('Paytm Money', 'investment'),
    'kuvera': ('Kuvera', 'investment'),
    'coin': ('Zerodha Coin', 'investment'),
    
    # Insurance
    'lic': ('LIC', 'investment'),
    'policybazaar': ('PolicyBazaar', 'investment'),
    'acko': ('Acko', 'investment'),
    'digit': ('Digit Insurance', 'investment'),
}


# Category keywords (fallback when VPA doesn't match)
CATEGORY_KEYWORDS = {
    'food': ['food', 'restaurant', 'cafe', 'coffee', 'lunch', 'dinner', 'breakfast', 
             'snack', 'meal', 'pizza', 'burger', 'biryani', 'curry', 'thali'],
    'shopping': ['shopping', 'purchase', 'order', 'buy', 'shop', 'store', 'mart',
                 'fashion', 'clothing', 'electronics', 'mobile', 'laptop'],
    'transport': ['cab', 'taxi', 'ride', 'travel', 'flight', 'train', 'bus', 
                  'petrol', 'diesel', 'fuel', 'toll', 'parking', 'metro'],
    'utilities': ['recharge', 'bill', 'electricity', 'water', 'gas', 'internet',
                  'broadband', 'postpaid', 'prepaid', 'dth'],
    'entertainment': ['movie', 'ticket', 'show', 'subscription', 'stream', 
                      'music', 'game', 'concert', 'event'],
    'transfer': ['transfer', 'sent', 'paid', 'payment'],
    'salary': ['salary', 'wages', 'income', 'pay'],
    'healthcare': ['hospital', 'clinic', 'medicine', 'pharmacy', 'doctor', 
                   'health', 'medical', 'diagnostic'],
    'education': ['school', 'college', 'university', 'course', 'tuition', 
                  'fees', 'education', 'training'],
}


def extract_merchant_from_vpa(vpa: str) -> Optional[str]:
    """
    Extract merchant name from UPI VPA.
    
    Args:
        vpa: UPI VPA (e.g., 'swiggy@ybl')
        
    Returns:
        Merchant name if found, None otherwise
    """
    if not vpa:
        return None
    
    vpa_lower = vpa.lower().strip()
    
    # Extract username part (before @)
    username = vpa_lower.split('@')[0] if '@' in vpa_lower else vpa_lower
    
    # Check for exact match
    if username in KNOWN_MERCHANTS:
        return KNOWN_MERCHANTS[username][0]
    
    # Check for prefix match
    for key, (merchant, _) in KNOWN_MERCHANTS.items():
        if username.startswith(key) or key in username:
            return merchant
    
    return None


def get_category_from_merchant(merchant: str) -> Optional[str]:
    """
    Get category from merchant name.
    
    Args:
        merchant: Merchant name
        
    Returns:
        Category string if found, None otherwise
    """
    if not merchant:
        return None
    
    merchant_lower = merchant.lower().strip()
    
    # Check known merchants
    for key, (name, category) in KNOWN_MERCHANTS.items():
        if key in merchant_lower or merchant_lower in name.lower():
            return category
    
    return None


def get_category_from_text(text: str) -> Optional[str]:
    """
    Infer category from transaction text using keywords.
    
    Args:
        text: Transaction description
        
    Returns:
        Category string if found, None otherwise
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Score each category
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return None


def get_merchant_and_category(vpa: Optional[str] = None, 
                               text: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Get merchant and category from VPA and/or text.
    
    Args:
        vpa: UPI VPA
        text: Transaction text
        
    Returns:
        Tuple of (merchant, category)
    """
    merchant = None
    category = None
    
    # Try VPA first
    if vpa:
        merchant = extract_merchant_from_vpa(vpa)
        if merchant:
            category = get_category_from_merchant(merchant)
    
    # Fallback to text
    if not category and text:
        category = get_category_from_text(text)
    
    return merchant, category


def get_bank_from_vpa(vpa: str) -> Optional[str]:
    """
    Get bank/app name from VPA suffix.
    
    Args:
        vpa: UPI VPA
        
    Returns:
        Bank/app name if found, None otherwise
    """
    if not vpa or '@' not in vpa:
        return None
    
    suffix = vpa.split('@')[1].lower()
    return VPA_BANKS.get(suffix)
