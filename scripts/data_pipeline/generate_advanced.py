#!/usr/bin/env python3
"""
Advanced Synthetic Data Generator v4.0
======================================

New Features:
1. Markov Chain for realistic message flow
2. Real data calibration from actual samples
3. Multilingual support (Hindi, Tamil, Telugu, Bengali, Kannada)
4. PDF/Image generation for document training
5. Statistical augmentation for rare edge cases

Author: Ranjit Behera
"""

from __future__ import annotations
import json
import random
import hashlib
import argparse
import math
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any, Iterator
from collections import defaultdict, Counter
import re


# ============================================================================
# MARKOV CHAIN MESSAGE GENERATOR
# ============================================================================

class MarkovChain:
    """
    Markov Chain for realistic message structure.
    
    Learns transition probabilities from real data to generate
    messages that follow actual patterns.
    """
    
    def __init__(self, order: int = 2):
        """
        Args:
            order: n-gram order (1 = unigram, 2 = bigram, etc.)
        """
        self.order = order
        self.transitions: Dict[Tuple, Counter] = defaultdict(Counter)
        self.start_states: Counter = Counter()
        
    def train(self, messages: List[str]):
        """Train on real messages."""
        for message in messages:
            tokens = self._tokenize(message)
            if len(tokens) <= self.order:
                continue
            
            # Start state
            start = tuple(tokens[:self.order])
            self.start_states[start] += 1
            
            # Build transitions
            for i in range(len(tokens) - self.order):
                state = tuple(tokens[i:i + self.order])
                next_token = tokens[i + self.order]
                self.transitions[state][next_token] += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize preserving structure."""
        # Split on whitespace but keep special tokens
        tokens = []
        for word in text.split():
            # Keep amounts as single tokens
            if re.match(r'^Rs\.?\d', word) or re.match(r'^₹\d', word):
                tokens.append(word)
            # Keep VPAs together
            elif '@' in word:
                tokens.append(word)
            else:
                tokens.append(word)
        return tokens
    
    def generate(self, rng: random.Random, max_length: int = 50) -> str:
        """Generate a message using learned transitions."""
        if not self.start_states:
            return ""
        
        # Sample start state
        states = list(self.start_states.keys())
        weights = [self.start_states[s] for s in states]
        current = list(rng.choices(states, weights=weights)[0])
        
        result = list(current)
        
        for _ in range(max_length - len(current)):
            state = tuple(current[-self.order:])
            
            if state not in self.transitions:
                break
            
            # Sample next token
            next_tokens = list(self.transitions[state].keys())
            weights = [self.transitions[state][t] for t in next_tokens]
            next_token = rng.choices(next_tokens, weights=weights)[0]
            
            result.append(next_token)
            current.append(next_token)
        
        return ' '.join(result)
    
    def save(self, path: Path):
        """Save trained model."""
        with open(path, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'transitions': dict(self.transitions),
                'start_states': dict(self.start_states),
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> 'MarkovChain':
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        chain = cls(order=data['order'])
        chain.transitions = defaultdict(Counter, {
            k: Counter(v) for k, v in data['transitions'].items()
        })
        chain.start_states = Counter(data['start_states'])
        return chain


class HybridGenerator:
    """
    Combines Markov Chain with template-based generation.
    
    Uses Markov for structure, templates for entity placement.
    """
    
    def __init__(self, markov: MarkovChain):
        self.markov = markov
        self.entity_patterns = {
            'AMOUNT': r'Rs\.?\s*[\d,]+(?:\.\d{2})?',
            'ACCOUNT': r'XX\d{4}',
            'DATE': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'REF': r'\d{12,16}',
            'VPA': r'[a-z0-9]+@[a-z]+',
        }
    
    def generate(self, entities: Dict[str, str], rng: random.Random) -> str:
        """
        Generate message with specific entities.
        
        1. Generate base structure from Markov
        2. Replace placeholders with actual entities
        """
        base = self.markov.generate(rng)
        
        # Replace detected patterns with actual entities
        for entity_type, pattern in self.entity_patterns.items():
            if entity_type in entities:
                base = re.sub(pattern, entities[entity_type], base, count=1)
        
        return base


# ============================================================================
# REAL DATA CALIBRATION
# ============================================================================

@dataclass
class DistributionFit:
    """Fitted statistical distribution."""
    name: str
    params: Dict[str, float]
    
    def sample(self, rng: random.Random) -> float:
        """Sample from fitted distribution."""
        if self.name == 'normal':
            value = rng.gauss(self.params['mean'], self.params['std'])
            return max(self.params.get('min', 0), 
                      min(self.params.get('max', float('inf')), value))
        
        elif self.name == 'lognormal':
            # Log-normal is better for amounts
            log_value = rng.gauss(self.params['mu'], self.params['sigma'])
            return math.exp(log_value)
        
        elif self.name == 'exponential':
            return rng.expovariate(1 / self.params['lambda'])
        
        elif self.name == 'uniform':
            return rng.uniform(self.params['min'], self.params['max'])
        
        elif self.name == 'categorical':
            items = list(self.params['categories'].keys())
            weights = list(self.params['categories'].values())
            return rng.choices(items, weights=weights)[0]
        
        return 0


class DataCalibrator:
    """
    Calibrate synthetic distributions to match real data.
    
    Fits statistical distributions to actual transaction data.
    """
    
    def __init__(self):
        self.amount_dist: Optional[DistributionFit] = None
        self.category_dist: Optional[DistributionFit] = None
        self.bank_dist: Optional[DistributionFit] = None
        self.hour_dist: Optional[DistributionFit] = None
        
        # Amount by category
        self.amount_by_category: Dict[str, DistributionFit] = {}
    
    def fit_from_data(self, data: List[Dict]):
        """Fit distributions from real data."""
        if not data:
            return
        
        # Extract amounts
        amounts = [r['amount'] for r in data if r.get('amount')]
        if amounts:
            mean_amt = sum(amounts) / len(amounts)
            var_amt = sum((x - mean_amt) ** 2 for x in amounts) / len(amounts)
            std_amt = math.sqrt(var_amt)
            
            # Use log-normal for amounts (always positive, right-skewed)
            log_amounts = [math.log(max(1, a)) for a in amounts]
            mu = sum(log_amounts) / len(log_amounts)
            sigma_sq = sum((x - mu) ** 2 for x in log_amounts) / len(log_amounts)
            
            self.amount_dist = DistributionFit(
                name='lognormal',
                params={'mu': mu, 'sigma': math.sqrt(sigma_sq)}
            )
        
        # Fit categories
        categories = Counter(r.get('category') for r in data if r.get('category'))
        if categories:
            total = sum(categories.values())
            self.category_dist = DistributionFit(
                name='categorical',
                params={'categories': {k: v/total for k, v in categories.items()}}
            )
        
        # Fit banks
        banks = Counter(r.get('bank') for r in data if r.get('bank'))
        if banks:
            total = sum(banks.values())
            self.bank_dist = DistributionFit(
                name='categorical',
                params={'categories': {k: v/total for k, v in banks.items()}}
            )
        
        # Fit amounts by category
        by_category = defaultdict(list)
        for r in data:
            if r.get('amount') and r.get('category'):
                by_category[r['category']].append(r['amount'])
        
        for cat, amounts in by_category.items():
            if len(amounts) >= 10:
                log_amounts = [math.log(max(1, a)) for a in amounts]
                mu = sum(log_amounts) / len(log_amounts)
                sigma_sq = sum((x - mu) ** 2 for x in log_amounts) / len(log_amounts)
                
                self.amount_by_category[cat] = DistributionFit(
                    name='lognormal',
                    params={'mu': mu, 'sigma': max(0.1, math.sqrt(sigma_sq))}
                )
    
    def sample_amount(self, category: Optional[str], rng: random.Random) -> float:
        """Sample amount, optionally by category."""
        if category and category in self.amount_by_category:
            return self.amount_by_category[category].sample(rng)
        elif self.amount_dist:
            return self.amount_dist.sample(rng)
        else:
            return rng.uniform(100, 10000)
    
    def sample_category(self, rng: random.Random) -> str:
        """Sample category from fitted distribution."""
        if self.category_dist:
            return self.category_dist.sample(rng)
        return 'shopping'
    
    def sample_bank(self, rng: random.Random) -> str:
        """Sample bank from fitted distribution."""
        if self.bank_dist:
            return self.bank_dist.sample(rng)
        return 'HDFC'
    
    def save(self, path: Path):
        """Save calibration."""
        with open(path, 'wb') as f:
            pickle.dump({
                'amount_dist': self.amount_dist,
                'category_dist': self.category_dist,
                'bank_dist': self.bank_dist,
                'amount_by_category': self.amount_by_category,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> 'DataCalibrator':
        """Load calibration."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls()
        calibrator.amount_dist = data.get('amount_dist')
        calibrator.category_dist = data.get('category_dist')
        calibrator.bank_dist = data.get('bank_dist')
        calibrator.amount_by_category = data.get('amount_by_category', {})
        return calibrator


# ============================================================================
# MULTILINGUAL SUPPORT
# ============================================================================

class Language(Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    KANNADA = "kn"
    MARATHI = "mr"
    GUJARATI = "gu"


@dataclass
class MultilingualTemplate:
    """Template with translations."""
    english: str
    translations: Dict[Language, str]
    
    def get(self, lang: Language) -> str:
        """Get template in specified language."""
        if lang == Language.ENGLISH:
            return self.english
        return self.translations.get(lang, self.english)


class MultilingualBank:
    """
    Bank SMS templates in multiple Indian languages.
    
    Based on actual bank SMS formats in different languages.
    """
    
    TEMPLATES = {
        'debit': MultilingualTemplate(
            english="{bank}: Rs.{amount} debited from A/c XX{account} on {date}. {vpa}. Ref: {ref}",
            translations={
                Language.HINDI: "{bank}: आपके खाते XX{account} से Rs.{amount} डेबिट हुआ। दिनांक {date}। {vpa}। संदर्भ: {ref}",
                Language.TAMIL: "{bank}: உங்கள் கணக்கு XX{account} இல் இருந்து Rs.{amount} டெபிட் செய்யப்பட்டது. தேதி {date}. Ref: {ref}",
                Language.TELUGU: "{bank}: మీ ఖాతా XX{account} నుండి Rs.{amount} డెబిట్ చేయబడింది. తేదీ {date}. {vpa}. Ref: {ref}",
                Language.BENGALI: "{bank}: আপনার অ্যাকাউন্ট XX{account} থেকে Rs.{amount} ডেবিট হয়েছে। তারিখ {date}। Ref: {ref}",
                Language.KANNADA: "{bank}: ನಿಮ್ಮ ಖಾತೆ XX{account} ನಿಂದ Rs.{amount} ಡೆಬಿಟ್ ಆಗಿದೆ. ದಿನಾಂಕ {date}. Ref: {ref}",
                Language.MARATHI: "{bank}: तुमच्या खात्यातून XX{account} Rs.{amount} डेबिट झाले. तारीख {date}. Ref: {ref}",
                Language.GUJARATI: "{bank}: તમારા ખાતા XX{account} માંથી Rs.{amount} ડેબિટ થયું. તારીખ {date}. Ref: {ref}",
            }
        ),
        'credit': MultilingualTemplate(
            english="{bank}: Rs.{amount} credited to A/c XX{account} on {date}. {sender}. Ref: {ref}",
            translations={
                Language.HINDI: "{bank}: आपके खाते XX{account} में Rs.{amount} क्रेडिट हुआ। दिनांक {date}। {sender}। संदर्भ: {ref}",
                Language.TAMIL: "{bank}: உங்கள் கணக்கு XX{account} க்கு Rs.{amount} கிரெடிட் செய்யப்பட்டது. தேதி {date}. Ref: {ref}",
                Language.TELUGU: "{bank}: మీ ఖాతా XX{account} కు Rs.{amount} క్రెడిట్ చేయబడింది. తేదీ {date}. Ref: {ref}",
                Language.BENGALI: "{bank}: আপনার অ্যাকাউন্ট XX{account} এ Rs.{amount} ক্রেডিট হয়েছে। তারিখ {date}। Ref: {ref}",
                Language.KANNADA: "{bank}: ನಿಮ್ಮ ಖಾತೆ XX{account} ಗೆ Rs.{amount} ಕ್ರೆಡಿಟ್ ಆಗಿದೆ. ದಿನಾಂಕ {date}. Ref: {ref}",
                Language.MARATHI: "{bank}: तुमच्या खात्यात XX{account} Rs.{amount} क्रेडिट झाले. तारीख {date}. Ref: {ref}",
                Language.GUJARATI: "{bank}: તમારા ખાતા XX{account} માં Rs.{amount} ક્રેડિટ થયું. તારીખ {date}. Ref: {ref}",
            }
        ),
        'otp': MultilingualTemplate(
            english="{bank}: Your OTP is {otp}. Valid for 10 mins. Do not share with anyone.",
            translations={
                Language.HINDI: "{bank}: आपका OTP {otp} है। 10 मिनट के लिए मान्य। किसी के साथ साझा न करें।",
                Language.TAMIL: "{bank}: உங்கள் OTP {otp}. 10 நிமிடங்களுக்கு செல்லுபடியாகும். யாருடனும் பகிர வேண்டாம்.",
                Language.TELUGU: "{bank}: మీ OTP {otp}. 10 నిమిషాలు చెల్లుబాటు. ఎవరితోనూ షేర్ చేయకండి.",
                Language.BENGALI: "{bank}: আপনার OTP হল {otp}। 10 মিনিটের জন্য বৈধ। কারো সাথে শেয়ার করবেন না।",
            }
        ),
        'balance': MultilingualTemplate(
            english="{bank}: Your A/c XX{account} balance is Rs.{balance}.",
            translations={
                Language.HINDI: "{bank}: आपके खाते XX{account} में शेष राशि Rs.{balance} है।",
                Language.TAMIL: "{bank}: உங்கள் கணக்கு XX{account} இருப்பு Rs.{balance}.",
                Language.TELUGU: "{bank}: మీ ఖాతా XX{account} బ్యాలెన్స్ Rs.{balance}.",
                Language.BENGALI: "{bank}: আপনার অ্যাকাউন্ট XX{account} ব্যালেন্স Rs.{balance}।",
            }
        ),
    }
    
    # Numbers in Indian languages
    NUMBERS = {
        Language.HINDI: {
            '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
            '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
        },
        Language.BENGALI: {
            '0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
            '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯',
        },
        Language.TAMIL: {
            '0': '௦', '1': '௧', '2': '௨', '3': '௩', '4': '௪',
            '5': '௫', '6': '௬', '7': '௭', '8': '௮', '9': '௯',
        },
        Language.KANNADA: {
            '0': '೦', '1': '೧', '2': '೨', '3': '೩', '4': '೪',
            '5': '೫', '6': '೬', '7': '೭', '8': '೮', '9': '೯',
        },
    }
    
    @classmethod
    def generate(
        cls,
        template_type: str,
        language: Language,
        params: Dict[str, str],
        use_native_numbers: bool = False,
        rng: random.Random = None
    ) -> str:
        """Generate message in specified language."""
        template = cls.TEMPLATES.get(template_type)
        if not template:
            return ""
        
        text = template.get(language)
        message = text.format(**params)
        
        # Optionally convert numbers to native script
        if use_native_numbers and language in cls.NUMBERS:
            for eng, native in cls.NUMBERS[language].items():
                message = message.replace(eng, native)
        
        return message


class MultilingualNameGenerator:
    """Generate names in multiple Indian languages."""
    
    NAMES = {
        Language.HINDI: [
            "राहुल शर्मा", "प्रिया सिंह", "अमित कुमार", "नेहा गुप्ता",
            "विजय पटेल", "दीपक वर्मा", "अंजलि मेहता", "राजेश नायर",
            "सुनीता अय्यर", "अरुण जोशी", "पूजा रेड्डी", "संजय मिश्रा",
        ],
        Language.TAMIL: [
            "முருகன் செல்வம்", "லக்ஷ்மி நாராயணன்", "கார்த்திக் சுப்பிரமணியம்",
            "மீனா குமார்", "அருண் பிரகாஷ்", "சரிதா வேணுகோபால்",
        ],
        Language.TELUGU: [
            "రవి కుమార్", "లక్ష్మీ దేవి", "సురేష్ రెడ్డి", "వెంకట రావు",
            "ప్రసాద్ నాయుడు", "కమల శర్మ", "రాజేష్ గుప్తా",
        ],
        Language.BENGALI: [
            "রাহুল ব্যানার্জী", "প্রিয়া দাস", "অমিত চক্রবর্তী",
            "সুমিতা সেন", "রাজেশ মুখার্জী", "কবিতা বসু",
        ],
        Language.KANNADA: [
            "ರಾಜೇಶ್ ಗೌಡ", "ಲಕ್ಷ್ಮೀ ನಾರಾಯಣ", "ಸುರೇಶ್ ಕುಮಾರ್",
            "ಮೀನಾ ಹೆಗ್ಡೆ", "ಪ್ರಕಾಶ್ ರಾವ್", "ನೇತ್ರಾ ಶೆಟ್ಟಿ",
        ],
    }
    
    @classmethod
    def get_name(cls, language: Language, rng: random.Random) -> str:
        """Get a random name in specified language."""
        names = cls.NAMES.get(language, cls.NAMES[Language.HINDI])
        return rng.choice(names)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmenter:
    """
    Advanced data augmentation techniques.
    
    Techniques:
    1. Back-translation (via templates)
    2. Synonym replacement
    3. Random insertion/deletion
    4. Noise injection
    5. Entity swapping
    """
    
    SYNONYMS = {
        'debited': ['withdrawn', 'deducted', 'paid', 'transferred', 'sent'],
        'credited': ['received', 'deposited', 'added', 'transferred'],
        'transaction': ['payment', 'transfer', 'txn'],
        'account': ['A/c', 'Acc', 'Acct', 'a/c'],
        'reference': ['Ref', 'UTR', 'Txn ID'],
        'available': ['Avl', 'remaining', 'left'],
        'balance': ['Bal', 'amt'],
    }
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def augment(
        self,
        text: str,
        ground_truth: Dict,
        techniques: List[str] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Generate augmented versions of a sample.
        
        Returns list of (augmented_text, ground_truth) tuples.
        """
        if techniques is None:
            techniques = ['synonym', 'noise', 'case']
        
        augmented = []
        
        if 'synonym' in techniques:
            aug = self._synonym_replace(text)
            augmented.append((aug, ground_truth))
        
        if 'noise' in techniques:
            aug = self._add_noise(text)
            augmented.append((aug, ground_truth))
        
        if 'case' in techniques:
            aug = self._vary_case(text)
            augmented.append((aug, ground_truth))
        
        if 'truncate' in techniques:
            aug = self._truncate(text)
            augmented.append((aug, ground_truth))
        
        if 'reorder' in techniques:
            aug = self._reorder_phrases(text)
            augmented.append((aug, ground_truth))
        
        return augmented
    
    def _synonym_replace(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,;:')
            if word_lower in self.SYNONYMS and self.rng.random() < 0.3:
                synonym = self.rng.choice(self.SYNONYMS[word_lower])
                # Preserve case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                words[i] = synonym
        return ' '.join(words)
    
    def _add_noise(self, text: str) -> str:
        """Add realistic noise."""
        # Random spacing
        if self.rng.random() < 0.3:
            text = text.replace('. ', '.')
        if self.rng.random() < 0.3:
            text = text.replace(': ', ':')
        
        # Abbreviations
        text = text.replace('Reference', 'Ref' if self.rng.random() < 0.5 else 'Reference')
        text = text.replace('Account', 'A/c' if self.rng.random() < 0.5 else 'Account')
        
        return text
    
    def _vary_case(self, text: str) -> str:
        """Vary text case."""
        r = self.rng.random()
        if r < 0.2:
            return text.upper()
        elif r < 0.4:
            return text.lower()
        return text
    
    def _truncate(self, text: str) -> str:
        """Truncate to SMS limit."""
        if len(text) > 160:
            return text[:157] + '...'
        return text
    
    def _reorder_phrases(self, text: str) -> str:
        """Reorder independent phrases."""
        # Split by common delimiters
        phrases = re.split(r'[.;]', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        if len(phrases) <= 2:
            return text
        
        # Keep first phrase, shuffle middle, keep last
        first = phrases[0]
        last = phrases[-1]
        middle = phrases[1:-1]
        self.rng.shuffle(middle)
        
        return '. '.join([first] + middle + [last])
    
    def augment_batch(
        self,
        data: List[Dict],
        augmentation_factor: int = 3
    ) -> List[Dict]:
        """Augment entire dataset."""
        augmented_data = []
        
        for record in data:
            text = record.get('text') or record.get('input', '')
            gt = record.get('ground_truth', record.get('output', {}))
            
            if isinstance(gt, str):
                gt = json.loads(gt)
            
            # Original
            augmented_data.append(record)
            
            # Augmented versions
            for aug_text, aug_gt in self.augment(text, gt)[:augmentation_factor-1]:
                augmented_data.append({
                    'text': aug_text,
                    'ground_truth': aug_gt,
                    'augmented': True,
                })
        
        return augmented_data


# ============================================================================
# RARE EDGE CASE OVERSAMPLING
# ============================================================================

class RareEdgeCaseSampler:
    """
    Oversample rare edge cases to improve model robustness.
    
    Uses importance sampling to increase representation of:
    - Failed transactions
    - Large amounts
    - Unusual formats
    - Rare banks
    - Unicode text
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # Define edge case conditions
        self.edge_cases = {
            'failed_txn': lambda r: r.get('status') == 'failed',
            'pending_txn': lambda r: r.get('status') == 'pending',
            'large_amount': lambda r: (r.get('amount') or 0) > 100000,
            'small_amount': lambda r: (r.get('amount') or float('inf')) < 10,
            'unicode': lambda r: any(ord(c) > 127 for c in str(r.get('text', ''))),
            'credit': lambda r: r.get('type') == 'credit',
        }
        
        # Oversampling weights (higher = more samples)
        self.oversample_weights = {
            'failed_txn': 5.0,
            'pending_txn': 3.0,
            'large_amount': 2.0,
            'small_amount': 2.0,
            'unicode': 4.0,
            'credit': 1.5,
        }
    
    def identify_edge_cases(self, record: Dict) -> List[str]:
        """Identify which edge cases a record matches."""
        return [
            name for name, condition in self.edge_cases.items()
            if condition(record)
        ]
    
    def calculate_sample_weight(self, record: Dict) -> float:
        """Calculate importance weight for a record."""
        weight = 1.0
        for edge_case in self.identify_edge_cases(record):
            weight *= self.oversample_weights.get(edge_case, 1.0)
        return weight
    
    def oversample(
        self,
        data: List[Dict],
        target_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Oversample data with edge case weighting.
        
        Returns dataset with increased representation of rare cases.
        """
        if target_size is None:
            target_size = len(data)
        
        # Calculate weights
        weights = [self.calculate_sample_weight(r) for r in data]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample with replacement
        indices = self.rng.choices(range(len(data)), weights=probs, k=target_size)
        
        oversampled = []
        for i in indices:
            record = data[i].copy()
            record['oversampled'] = True
            oversampled.append(record)
        
        return oversampled
    
    def generate_targeted_edge_cases(
        self,
        generator,
        edge_case_type: str,
        count: int
    ) -> List[Dict]:
        """Generate specific edge case samples."""
        samples = []
        
        if edge_case_type == 'failed_txn':
            from scripts.data_pipeline.generate_synthetic import TransactionStatus
            for _ in range(count):
                sample = generator.generate_transaction(
                    status=TransactionStatus.FAILED
                )
                samples.append(sample)
        
        elif edge_case_type == 'large_amount':
            for _ in range(count):
                sample = generator.generate_transaction()
                # Force large amount
                sample['ground_truth']['amount'] = self.rng.uniform(100000, 1000000)
                samples.append(sample)
        
        elif edge_case_type == 'unicode':
            for _ in range(count):
                sample = generator.generate_transaction()
                # Use Hindi name
                sample['ground_truth']['beneficiary'] = self.rng.choice([
                    "राहुल शर्मा", "प्रिया सिंह", "అమిత్ కుమార్"
                ])
                samples.append(sample)
        
        return samples


# ============================================================================
# DOCUMENT/PDF GENERATION (Placeholder - needs external libs)
# ============================================================================

class DocumentGenerator:
    """
    Generate synthetic bank statements and documents.
    
    Note: Full implementation requires:
    - reportlab for PDF generation
    - PIL for image processing
    - wkhtmltopdf for HTML to PDF
    """
    
    STATEMENT_TEMPLATE = """
    ============================================
    {bank} BANK
    ACCOUNT STATEMENT
    ============================================
    
    Account Holder: {name}
    Account Number: XXXXXXXX{account}
    Statement Period: {start_date} to {end_date}
    
    Opening Balance: Rs. {opening_balance}
    
    --------------------------------------------
    Date        Description          Debit    Credit    Balance
    --------------------------------------------
    {transactions}
    --------------------------------------------
    
    Closing Balance: Rs. {closing_balance}
    
    This is a computer-generated statement.
    """
    
    @classmethod
    def generate_text_statement(
        cls,
        transactions: List[Dict],
        bank: str,
        account: str,
        name: str,
        rng: random.Random
    ) -> str:
        """Generate a text-based bank statement."""
        if not transactions:
            return ""
        
        # Sort by date
        sorted_txns = sorted(
            transactions,
            key=lambda x: x.get('date', '2025-01-01')
        )
        
        # Calculate running balance
        opening = rng.randint(10000, 100000)
        balance = opening
        lines = []
        
        for txn in sorted_txns:
            amount = txn.get('amount', 0)
            txn_type = txn.get('type', 'debit')
            
            if txn_type == 'debit':
                balance -= amount
                debit = f"{amount:,.2f}"
                credit = ""
            else:
                balance += amount
                debit = ""
                credit = f"{amount:,.2f}"
            
            desc = txn.get('merchant') or txn.get('beneficiary') or 'Transaction'
            date_str = txn.get('date', '2025-01-01')
            
            line = f"{date_str}  {desc[:20]:<20}  {debit:>10}  {credit:>10}  {balance:>12,.2f}"
            lines.append(line)
        
        start_date = sorted_txns[0].get('date', '2025-01-01')
        end_date = sorted_txns[-1].get('date', '2025-01-31')
        
        return cls.STATEMENT_TEMPLATE.format(
            bank=bank,
            name=name,
            account=account[-4:],
            start_date=start_date,
            end_date=end_date,
            opening_balance=f"{opening:,.2f}",
            closing_balance=f"{balance:,.2f}",
            transactions='\n    '.join(lines)
        )
    
    @classmethod
    def generate_statement_image_data(
        cls,
        transactions: List[Dict],
        bank: str,
        rng: random.Random
    ) -> Dict:
        """
        Generate data for statement image (actual rendering needs PIL).
        
        Returns structured data that can be used with image generation.
        """
        return {
            'type': 'bank_statement',
            'bank': bank,
            'transactions': transactions,
            'format': 'image_data',
            'note': 'Use PIL/reportlab to render actual image'
        }


# ============================================================================
# UNIFIED ADVANCED GENERATOR
# ============================================================================

class AdvancedSyntheticGenerator:
    """
    Unified generator combining all advanced features.
    
    Features:
    1. Markov chain learning from real data
    2. Statistical calibration
    3. Multilingual support
    4. Data augmentation
    5. Edge case oversampling
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Components
        self.markov: Optional[MarkovChain] = None
        self.calibrator: Optional[DataCalibrator] = None
        self.augmenter = DataAugmenter(seed)
        self.edge_sampler = RareEdgeCaseSampler(seed)
    
    def train_on_real_data(self, real_data: List[Dict]):
        """Train/calibrate on real data."""
        print("Training on real data...")
        
        # Train Markov chain
        texts = [r.get('text') or r.get('input', '') for r in real_data]
        self.markov = MarkovChain(order=2)
        self.markov.train(texts)
        print(f"  Markov chain trained on {len(texts)} samples")
        
        # Calibrate distributions
        self.calibrator = DataCalibrator()
        parsed_data = []
        for r in real_data:
            gt = r.get('ground_truth') or r.get('output', {})
            if isinstance(gt, str):
                gt = json.loads(gt)
            parsed_data.append(gt)
        
        self.calibrator.fit_from_data(parsed_data)
        print("  Distributions calibrated")
    
    def generate(
        self,
        count: int,
        languages: List[Language] = None,
        include_documents: bool = False,
        augmentation_factor: int = 1,
        edge_case_ratio: float = 0.1,
    ) -> List[Dict]:
        """
        Generate synthetic data with all advanced features.
        
        Args:
            count: Number of records
            languages: Languages to include (None = English only)
            include_documents: Include bank statement format
            augmentation_factor: How many augmented versions per sample
            edge_case_ratio: Proportion of edge cases to include
        """
        if languages is None:
            languages = [Language.ENGLISH]
        
        records = []
        base_count = int(count / augmentation_factor)
        edge_count = int(base_count * edge_case_ratio)
        normal_count = base_count - edge_count
        
        print(f"Generating {count:,} records...")
        print(f"  Base: {base_count:,}, Edges: {edge_count:,}, Augmented: {count - base_count:,}")
        
        # Generate normal transactions
        for i in range(normal_count):
            lang = self.rng.choice(languages)
            
            # Sample from calibrated distributions if available
            if self.calibrator:
                category = self.calibrator.sample_category(self.rng)
                bank = self.calibrator.sample_bank(self.rng)
                amount = self.calibrator.sample_amount(category, self.rng)
            else:
                category = self.rng.choice(['shopping', 'food', 'transfer', 'bills'])
                bank = self.rng.choice(['HDFC', 'ICICI', 'SBI', 'Axis'])
                amount = self.rng.uniform(100, 10000)
            
            # Generate message
            is_debit = self.rng.random() < 0.7
            template_type = 'debit' if is_debit else 'credit'
            
            params = {
                'bank': bank,
                'amount': f"{amount:,.2f}",
                'account': str(self.rng.randint(1000, 9999)),
                'date': (date.today() - timedelta(days=self.rng.randint(0, 365))).strftime('%d-%m-%Y'),
                'vpa': f"{self.rng.choice(['swiggy', 'amazon', 'paytm'])}@ybl",
                'sender': 'PhonePe',
                'ref': ''.join(self.rng.choices('0123456789', k=12)),
            }
            
            text = MultilingualBank.generate(template_type, lang, params)
            
            records.append({
                'text': text,
                'ground_truth': {
                    'amount': round(amount, 2),
                    'type': 'debit' if is_debit else 'credit',
                    'bank': bank,
                    'category': category,
                    'language': lang.value,
                },
                'language': lang.value,
            })
            
            if (i + 1) % 5000 == 0:
                print(f"  Generated {i+1:,}/{base_count:,}")
        
        # Generate edge cases
        for i in range(edge_count):
            lang = self.rng.choice(languages)
            edge_type = self.rng.choice(['unicode', 'large_amount', 'small_amount'])
            
            if edge_type == 'unicode' and lang == Language.ENGLISH:
                lang = Language.HINDI
            
            amount = (
                self.rng.uniform(100000, 1000000) if edge_type == 'large_amount'
                else self.rng.uniform(0.5, 10) if edge_type == 'small_amount'
                else self.rng.uniform(100, 10000)
            )
            
            params = {
                'bank': self.rng.choice(['HDFC', 'ICICI', 'SBI']),
                'amount': f"{amount:,.2f}",
                'account': str(self.rng.randint(1000, 9999)),
                'date': date.today().strftime('%d-%m-%Y'),
                'vpa': 'merchant@ybl',
                'sender': MultilingualNameGenerator.get_name(lang, self.rng) if edge_type == 'unicode' else 'User',
                'ref': ''.join(self.rng.choices('0123456789', k=12)),
            }
            
            text = MultilingualBank.generate('debit', lang, params, use_native_numbers=(edge_type == 'unicode'))
            
            records.append({
                'text': text,
                'ground_truth': {
                    'amount': round(amount, 2),
                    'type': 'debit',
                    'language': lang.value,
                },
                'edge_case': edge_type,
                'language': lang.value,
            })
        
        # Augment if factor > 1
        if augmentation_factor > 1:
            print(f"  Augmenting {len(records):,} records...")
            records = self.augmenter.augment_batch(records, augmentation_factor)
        
        # Add statements if requested
        if include_documents:
            print("  Generating document samples...")
            for _ in range(min(100, count // 100)):
                bank = self.rng.choice(['HDFC', 'ICICI', 'SBI'])
                account = str(self.rng.randint(10000000, 99999999))
                name = self.rng.choice(['Rahul Sharma', 'Priya Singh', 'Amit Kumar'])
                
                # Use recent records for statement
                txns = [r['ground_truth'] for r in self.rng.sample(records, min(10, len(records)))]
                statement = DocumentGenerator.generate_text_statement(
                    txns, bank, account, name, self.rng
                )
                
                records.append({
                    'text': statement,
                    'ground_truth': {'document_type': 'bank_statement', 'bank': bank},
                    'document': True,
                })
        
        self.rng.shuffle(records)
        
        # Add IDs
        for i, r in enumerate(records):
            r['id'] = i + 1
        
        print(f"✅ Generated {len(records):,} total records")
        return records
    
    def save_training_data(self, records: List[Dict], output_path: Path):
        """Save in training format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in records:
                line = {
                    'input': r['text'],
                    'output': json.dumps(r['ground_truth'], ensure_ascii=False),
                    'id': r.get('id'),
                    'language': r.get('language', 'en'),
                }
                if r.get('edge_case'):
                    line['edge_case'] = r['edge_case']
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced Synthetic Data Generator v4.0")
    parser.add_argument("-n", "--count", type=int, default=10000, help="Number of records")
    parser.add_argument("-o", "--output", default="data/synthetic/advanced_synthetic.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--languages", nargs='+', default=['en'], 
                       help="Languages: en, hi, ta, te, bn, kn, mr, gu")
    parser.add_argument("--augment", type=int, default=1, help="Augmentation factor")
    parser.add_argument("--edge-ratio", type=float, default=0.1, help="Edge case ratio")
    parser.add_argument("--real-data", help="Path to real data for calibration")
    parser.add_argument("--documents", action="store_true", help="Include document samples")
    
    args = parser.parse_args()
    
    # Parse languages
    lang_map = {
        'en': Language.ENGLISH, 'hi': Language.HINDI, 'ta': Language.TAMIL,
        'te': Language.TELUGU, 'bn': Language.BENGALI, 'kn': Language.KANNADA,
        'mr': Language.MARATHI, 'gu': Language.GUJARATI,
    }
    languages = [lang_map.get(l, Language.ENGLISH) for l in args.languages]
    
    # Initialize generator
    generator = AdvancedSyntheticGenerator(seed=args.seed)
    
    # Train on real data if provided
    if args.real_data:
        real_path = Path(args.real_data)
        if real_path.exists():
            with open(real_path) as f:
                real_data = [json.loads(line) for line in f]
            generator.train_on_real_data(real_data)
    
    # Generate
    records = generator.generate(
        count=args.count,
        languages=languages,
        include_documents=args.documents,
        augmentation_factor=args.augment,
        edge_case_ratio=args.edge_ratio,
    )
    
    # Save
    output_path = Path(args.output)
    generator.save_training_data(records, output_path)
    
    # Summary
    print("\n📊 Summary:")
    lang_counts = Counter(r.get('language', 'en') for r in records)
    for lang, count in lang_counts.most_common():
        print(f"  {lang}: {count:,}")
    
    edge_counts = Counter(r.get('edge_case') for r in records if r.get('edge_case'))
    if edge_counts:
        print("\n📋 Edge Cases:")
        for edge, count in edge_counts.most_common():
            print(f"  {edge}: {count:,}")


if __name__ == "__main__":
    main()
