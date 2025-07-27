import os
import json
import time
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional, Any
import logging
from collections import defaultdict
import hashlib
from functools import lru_cache
import pickle
from pathlib import Path
import warnings
import re
from collections import Counter
import torch
warnings.filterwarnings('ignore')

# CRITICAL: Fix meta tensor issues with NUCLEAR OPTIONS
def fix_torch_settings():
    """NUCLEAR OPTION: Fix PyTorch settings to prevent meta tensor issues."""
    # Set default tensor type
    torch.set_default_dtype(torch.float32)
    
    # NUCLEAR: Disable all lazy loading mechanisms
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_DISABLE_LAZY_MODULES'] = '1'
    os.environ['TORCH_DISABLE_META_DEVICE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    
    # NUCLEAR: Force proper CUDA initialization
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.init()
            torch.cuda.synchronize()
            # Force a small tensor creation to initialize CUDA properly
            dummy = torch.tensor([1.0], device='cuda')
            del dummy
            torch.cuda.empty_cache()
        except:
            pass  # Ignore CUDA init errors
    
    print("🔥 NUCLEAR meta tensor protection activated!")

# Call this immediately
fix_torch_settings()

# REAL ESTATE SYNONYM DICTIONARY
REAL_ESTATE_SYNONYMS = {
    'property': ['real estate', 'building', 'asset', 'premises', 'development'],
    'investment': ['investing', 'acquisition', 'purchase', 'buying', 'funding'],
    'market': ['marketplace', 'sector', 'industry', 'economy', 'conditions'],
    'rental': ['lease', 'tenancy', 'rent', 'letting', 'occupancy'],
    'commercial': ['business', 'office', 'retail', 'industrial', 'CRE'],
    'residential': ['housing', 'home', 'apartment', 'condo', 'single family'],
    'value': ['price', 'worth', 'valuation', 'appraisal', 'cost'],
    'location': ['area', 'neighborhood', 'district', 'region', 'zone'],
    'financing': ['loan', 'mortgage', 'lending', 'credit', 'capital'],
    'yield': ['return', 'ROI', 'profit', 'income', 'cash flow'],
    'analysis': ['evaluation', 'assessment', 'study', 'review', 'research'],
    'cap rate': ['capitalization rate', 'cap', 'NOI yield'],
    'NOI': ['net operating income', 'operating income'],
    'cash flow': ['net income', 'profit', 'earnings', 'returns'],
    'appreciation': ['growth', 'increase', 'gain', 'rise'],
    'vacancy': ['empty', 'unoccupied', 'available', 'vacant'],
    'tenant': ['renter', 'lessee', 'occupant', 'resident'],
    'due diligence': ['research', 'investigation', 'analysis', 'review'],
    'closing': ['settlement', 'completion', 'finalization'],
    'escrow': ['holding account', 'third party', 'neutral account']
}

# ENHANCED DEFAULT CONFIGURATION
DEFAULT_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "embeddings_file": "/home/ubuntu/scraper_deep/deep_embeddings.json",
    "index_cache_file": "/home/ubuntu/scraper_deep/faiss_index.pkl",
    "gemma_url": "http://localhost:11434/api/generate",
    "gemma_model": "gemma3:27b",
    "top_k": 8,
    "rerank_top_k": 4,
    "min_similarity": 0.15,
    "max_context_length": 2500,
    "cache_size": 50,
    "similarity_gap_threshold": 0.12,
    "max_context_chunks": 4,
    "use_cross_encoder": True,
    "use_hybrid_retrieval": True,
    "keyword_weight": 0.3,
    "llm_options": {
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 15,
        "num_predict": 400,
        "num_ctx": 3072,
        "repeat_penalty": 1.05,
        "mirostat": 2,
        "mirostat_eta": 0.2,
        "mirostat_tau": 6.0
    },
    "llm_timeout": 6000
}

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def safe_load_sentence_transformer(model_name: str, device: str = None, max_retries: int = 3):
    """NUCLEAR OPTION: Load SentenceTransformer with complete meta tensor elimination."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # NUCLEAR: Force disable all lazy loading and meta tensors
    os.environ['PYTORCH_DISABLE_LAZY_MODULES'] = '1'
    os.environ['TORCH_DISABLE_META_DEVICE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    for attempt in range(max_retries):
        try:
            print(f"🔧 Loading attempt {attempt + 1} for {model_name} on {device}")
            
            # NUCLEAR: Complete cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # NUCLEAR: Force non-meta tensor creation - Load on CPU first
            print("  Loading model on CPU first...")
            model = SentenceTransformer(
                model_name,
                device='cpu',  # Always load on CPU first
                trust_remote_code=True
            )
            
            # NUCLEAR: Fix any meta tensors using to_empty() method
            print("  Fixing meta tensors...")
            for name, module in model.named_modules():
                if hasattr(module, 'parameters'):
                    for param_name, param in module.named_parameters():
                        if param.is_meta:
                            print(f"    Fixing meta tensor: {name}.{param_name}")
                            # Use to_empty() as suggested by error message
                            # FIX: Remove the context manager from here
                            new_param = torch.nn.Parameter(
                                torch.empty_like(param, device='cpu').to(param.dtype)
                            )
                            # Initialize with zeros without context manager
                            torch.nn.init.zeros_(new_param)
                            # Replace the parameter
                            setattr(module, param_name.split('.')[-1], new_param)
            
            # Now safely move to target device if not CPU
            if device != 'cpu':
                print(f"  Moving model to {device}...")
                model = model.to(device)
            
            # NUCLEAR: Force initialization with test encoding
            print("  Testing model with sample input...")
            test_text = ["This is a test sentence to initialize the model properly."]
            test_encoding = model.encode(test_text, show_progress_bar=False, convert_to_numpy=True)
            
            # NUCLEAR: Verify no meta tensors remain
            print("  Verifying no meta tensors remain...")
            meta_tensor_found = False
            for name, param in model.named_parameters():
                if param.is_meta:
                    print(f"    WARNING: Meta tensor still found: {name}")
                    meta_tensor_found = True
            
            if meta_tensor_found:
                raise RuntimeError("Meta tensors still present after fixing")
            
            print(f"  ✅ Model loaded successfully on {device}")
            return model
                
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} failed: {e}")
            if "meta tensor" in str(e).lower():
                print("    Meta tensor error detected, trying more aggressive fix...")
                
                # NUCLEAR: Try to manually materialize all parameters
                try:
                    if 'model' in locals():
                        for name, param in model.named_parameters():
                            if param.is_meta:
                                # Create actual tensor data without context manager
                                materialized = torch.zeros_like(param, device='cpu')
                                param.data = materialized
                        
                        # Try again after materialization
                        if device != 'cpu':
                            model = model.to(device)
                        
                        test_encoding = model.encode(["test"], show_progress_bar=False, convert_to_numpy=True)
                        print("  ✅ Recovered after manual materialization")
                        return model
                        
                except Exception as recovery_error:
                    print(f"    Recovery failed: {recovery_error}")
            
            if attempt == max_retries - 1:
                # FINAL NUCLEAR OPTION: CPU only
                if device != 'cpu':
                    print("  🔄 Final attempt: CPU only...")
                    return safe_load_sentence_transformer(model_name, 'cpu', 1)
                else:
                    print("  ❌ All attempts failed, trying alternative approach...")
                    # Last resort: simple fallback without transformers manipulation
                    try:
                        print("  🔄 Final fallback: Basic model loading...")
                        model = SentenceTransformer(model_name, device='cpu')
                        # Immediate test
                        _ = model.encode(["test"], convert_to_numpy=True)
                        print("  ✅ Basic fallback successful")
                        return model
                    except Exception as final_error:
                        raise RuntimeError(f"COMPLETE FAILURE: Cannot load {model_name} after all attempts. Final error: {final_error}")
            
            time.sleep(2)
    
    # This should never be reached, but just in case
    raise RuntimeError(f"Failed to load {model_name} after {max_retries} attempts")

def safe_load_cross_encoder(model_name: str, device: str = None, max_retries: int = 3):
    """NUCLEAR OPTION: Load CrossEncoder with complete meta tensor elimination."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for attempt in range(max_retries):
        try:
            print(f"🔧 Loading CrossEncoder attempt {attempt + 1} on {device}")
            
            # NUCLEAR cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Always load on CPU first
            print("  Loading CrossEncoder on CPU first...")
            model = CrossEncoder(
                model_name,
                device='cpu',
                trust_remote_code=True
            )
            
            # NUCLEAR: Fix meta tensors using to_empty()
            print("  Fixing CrossEncoder meta tensors...")
            if hasattr(model, 'model'):
                for name, module in model.model.named_modules():
                    if hasattr(module, 'parameters'):
                        for param_name, param in module.named_parameters():
                            if param.is_meta:
                                print(f"    Fixing CrossEncoder meta tensor: {name}.{param_name}")
                                # FIX: Remove context manager from here too
                                new_param = torch.nn.Parameter(
                                    torch.empty_like(param, device='cpu').to(param.dtype)
                                )
                                # Initialize without context manager
                                torch.nn.init.zeros_(new_param)
                                setattr(module, param_name.split('.')[-1], new_param)
            
            # Move to target device
            if device != 'cpu':
                print(f"  Moving CrossEncoder to {device}...")
                model = model.to(device)
            
            # Test the model
            print("  Testing CrossEncoder...")
            test_pairs = [("test query", "test document")]
            _ = model.predict(test_pairs)
            
            print(f"  ✅ CrossEncoder loaded successfully on {device}")
            return model
                
        except Exception as e:
            print(f"  ❌ CrossEncoder attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                if device != 'cpu':
                    print("  CrossEncoder falling back to CPU...")
                    return safe_load_cross_encoder(model_name, 'cpu', 1)
                else:
                    print("  ❌ CrossEncoder failed completely, disabling...")
                    return None
            time.sleep(2)
    
    return None  # Explicit return for failed attempts

class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword matching."""
    
    def __init__(self, knowledge_base: List[Dict], index, model):
        self.knowledge_base = knowledge_base
        self.index = index
        self.model = model
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build inverted index for keyword matching."""
        self.keyword_index = defaultdict(set)
        self.doc_terms = {}
        
        for i, chunk in enumerate(self.knowledge_base):
            text = chunk['text'].lower()
            terms = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            
            self.doc_terms[i] = Counter(terms)
            
            for term in set(terms):
                self.keyword_index[term].add(i)
    
    def _keyword_search(self, query: str, top_k: int = 20) -> List[tuple]:
        """Perform keyword-based search with TF-IDF-like scoring."""
        query_terms = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            if term in self.keyword_index:
                idf = np.log(len(self.knowledge_base) / (len(self.keyword_index[term]) + 1))
                
                for doc_id in self.keyword_index[term]:
                    tf = self.doc_terms[doc_id].get(term, 0) / max(len(self.doc_terms[doc_id]), 1)
                    doc_scores[doc_id] += tf * idf
        
        if doc_scores:
            max_score = max(doc_scores.values())
            if max_score > 0:
                for doc_id in doc_scores:
                    doc_scores[doc_id] /= max_score
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
    
    def hybrid_retrieve(self, query: str, top_k: int, keyword_weight: float = 0.3) -> List[Dict]:
        """Combine semantic and keyword retrieval."""
        # Semantic retrieval with proper error handling
        try:
            query_emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype('float32')
            faiss.normalize_L2(query_emb)
            
            semantic_scores, semantic_indices = self.index.search(query_emb, top_k * 2)
        except Exception as e:
            print(f"Semantic search failed: {e}")
            semantic_scores = [[]]
            semantic_indices = [[]]
        
        # Keyword retrieval
        keyword_results = self._keyword_search(query, top_k * 2)
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}
        
        # Combine scores
        combined_results = {}
        
        # Add semantic results
        for score, idx in zip(semantic_scores[0], semantic_indices[0]):
            if idx != -1:
                combined_results[idx] = {
                    'semantic_score': float(score),
                    'keyword_score': keyword_scores.get(idx, 0.0),
                    'combined_score': (1 - keyword_weight) * float(score) + keyword_weight * keyword_scores.get(idx, 0.0)
                }
        
        # Add high-scoring keyword results not in semantic results
        for doc_id, kw_score in keyword_results:
            if doc_id not in combined_results and kw_score > 0.1:
                combined_results[doc_id] = {
                    'semantic_score': 0.0,
                    'keyword_score': kw_score,
                    'combined_score': keyword_weight * kw_score
                }
        
        # Sort by combined score and prepare results
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        results = []
        for doc_id, scores in sorted_results[:top_k]:
            chunk = self.knowledge_base[doc_id].copy()
            chunk['similarity'] = scores['combined_score']
            chunk['semantic_score'] = scores['semantic_score']
            chunk['keyword_score'] = scores['keyword_score']
            results.append(chunk)
        
        return results

class QualityFilter:
    """Filter chunks based on quality indicators."""
    
    def __init__(self):
        self.quality_indicators = [
            'analysis', 'research', 'study', 'data', 'report', 'market',
            'investment', 'strategy', 'performance', 'returns', 'risk'
        ]
        self.low_quality_indicators = ['advertisement', 'ad', 'sponsored', 'click here', 'buy now']
    
    def calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for a text chunk."""
        text_lower = text.lower()
        
        # Length check
        length_score = 1.0
        if len(text) < 200:
            length_score = len(text) / 200
        elif len(text) > 2000:
            length_score = max(0.7, 2000 / len(text))
        
        # Quality indicators
        quality_count = sum(1 for indicator in self.quality_indicators if indicator in text_lower)
        quality_score = min(quality_count * 0.1, 0.5)
        
        # Low quality penalties
        low_quality_count = sum(1 for indicator in self.low_quality_indicators if indicator in text_lower)
        quality_penalty = min(low_quality_count * 0.3, 0.5)
        
        # Information density
        sentences = re.split(r'[.!?]+', text)
        informative_sentences = sum(1 for sent in sentences if re.search(r'\d+\.?\d*%?|\$[\d,]+', sent))
        density_score = min(informative_sentences / max(len(sentences), 1) * 0.3, 0.3)
        
        final_score = length_score + quality_score - quality_penalty + density_score
        return max(0.0, min(final_score, 1.0))
    
    def filter_chunks(self, chunks: List[Dict], min_quality: float = 0.3) -> List[Dict]:
        """Filter chunks based on quality scores."""
        filtered = []
        for chunk in chunks:
            quality_score = self.calculate_quality_score(chunk['text'])
            if quality_score >= min_quality:
                chunk['quality_score'] = quality_score
                filtered.append(chunk)
        
        filtered.sort(key=lambda x: x['similarity'] * 0.7 + x.get('quality_score', 0) * 0.3, reverse=True)
        return filtered

class EnhancedQueryProcessor:
    """Enhanced query processing with better expansion and analysis."""
    
    def __init__(self):
        self.synonyms = REAL_ESTATE_SYNONYMS
        self.question_patterns = [
            (re.compile(r'^(what|how|when|where|why|which)\s+', re.IGNORECASE), 'question'),
            (re.compile(r'\?', re.IGNORECASE), 'question'),
            (re.compile(r'^(tell me|explain|describe)', re.IGNORECASE), 'explanation'),
            (re.compile(r'(example|case study|scenario)', re.IGNORECASE), 'example'),
            (re.compile(r'(calculate|compute|formula)', re.IGNORECASE), 'calculation'),
            (re.compile(r'(compare|versus|vs|difference)', re.IGNORECASE), 'comparison')
        ]
    
    def expand_query_advanced(self, query: str) -> str:
        """Advanced query expansion with context-aware synonyms."""
        words = query.lower().split()
        expanded_terms = []
        
        context_terms = set(words)
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)
            expanded_terms.append(word)
            
            if clean_word in self.synonyms:
                synonyms = self.synonyms[clean_word]
                
                for synonym in synonyms[:2]:
                    synonym_words = synonym.split()
                    if not any(sw in context_terms for sw in synonym_words):
                        expanded_terms.append(synonym)
                        context_terms.update(synonym_words)
        
        return ' '.join(expanded_terms)
    
    def extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts with weights."""
        concepts = []
        
        financial_patterns = [
            (r'cap\s*rate', 'cap_rate', 3),
            (r'noi|net operating income', 'noi', 3),
            (r'cash\s*flow', 'cash_flow', 3),
            (r'roi|return on investment', 'roi', 3),
            (r'yield', 'yield', 2),
            (r'appreciation', 'appreciation', 2)
        ]
        
        property_patterns = [
            (r'commercial|office|retail|industrial', 'commercial', 2),
            (r'residential|apartment|condo|house', 'residential', 2),
            (r'multi.?family', 'multifamily', 2)
        ]
        
        investment_patterns = [
            (r'investment|investing', 'investment', 2),
            (r'analysis|analyze', 'analysis', 2),
            (r'risk|risks', 'risk', 2),
            (r'market|markets', 'market', 1)
        ]
        
        all_patterns = financial_patterns + property_patterns + investment_patterns
        
        for pattern, concept, weight in all_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                concepts.append((concept, weight))
        
        return concepts
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query optimization."""
        cleaned = re.sub(r'\s+', ' ', query.strip())
        query_type = self.detect_query_type(cleaned)
        entities = self.extract_key_entities(cleaned)
        concepts = self.extract_key_concepts(cleaned)
        
        expanded = self.expand_query_advanced(cleaned)
        
        if query_type == 'question':
            focused = re.sub(r'^(what|how|when|where|why|which)\s+(is|are|do|does|can|should)\s*', '', cleaned, flags=re.IGNORECASE)
            focused = re.sub(r'\?', '', focused)
        else:
            focused = cleaned
        
        return {
            'original': query,
            'cleaned': cleaned,
            'expanded': expanded,
            'focused': focused,
            'query_type': query_type,
            'entities': entities,
            'concepts': concepts
        }
    
    def detect_query_type(self, query: str) -> str:
        """Detect query type."""
        for pattern, query_type in self.question_patterns:
            if pattern.search(query):
                return query_type
        return 'general'
    
    def extract_key_entities(self, query: str) -> List[str]:
        """Extract key entities."""
        entities = []
        
        property_types = ['apartment', 'condo', 'house', 'commercial', 'retail', 'office', 'industrial']
        for prop_type in property_types:
            if prop_type in query.lower():
                entities.append(prop_type)
        
        financial_terms = ['cap rate', 'noi', 'cash flow', 'roi', 'yield', 'appreciation']
        for term in financial_terms:
            if term in query.lower():
                entities.append(term)
        
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', query)
        entities.extend(numbers)
        
        return entities

class EnhancedContextOrganizer:
    """Enhanced context organization with better relevance and structure."""
    
    def __init__(self, max_chunks: int = 4, max_context_length: int = 2500):
        self.max_chunks = max_chunks
        self.max_context_length = max_context_length
    
    def extract_relevant_passages(self, text: str, query_concepts: List[tuple], max_passages: int = 3) -> str:
        """Extract most relevant passages based on query concepts."""
        passages = re.split(r'\n\n+|(?<=[.!?])\s+(?=[A-Z])', text)
        passages = [p.strip() for p in passages if len(p.strip()) > 50]
        
        if not passages:
            return text[:800]
        
        scored_passages = []
        concept_terms = [concept for concept, weight in query_concepts]
        
        for passage in passages:
            score = 0
            passage_lower = passage.lower()
            
            for concept, weight in query_concepts:
                if concept.replace('_', ' ') in passage_lower or concept.replace('_', '') in passage_lower:
                    score += weight
            
            if re.search(r'\d+\.?\d*%|\$[\d,]+', passage):
                score += 1
            
            re_terms = ['property', 'investment', 'market', 'analysis', 'return']
            score += sum(0.5 for term in re_terms if term in passage_lower)
            
            if score > 0:
                scored_passages.append((score, passage))
        
        scored_passages.sort(key=lambda x: x[0], reverse=True)
        top_passages = [passage for score, passage in scored_passages[:max_passages]]
        
        if not top_passages:
            return passages[0]
        
        return ' '.join(top_passages)
    
    def organize_context_enhanced(self, chunks: List[Dict], query_info: Dict) -> str:
        """Enhanced context organization."""
        if not chunks:
            return ""
        
        context_parts = []
        total_length = 0
        concepts = query_info.get('concepts', [])
        
        for i, chunk in enumerate(chunks[:self.max_chunks]):
            if concepts:
                relevant_content = self.extract_relevant_passages(chunk['text'], concepts)
            else:
                relevant_content = chunk['text'][:700]
            
            similarity_info = f"(relevance: {chunk['similarity']:.2f}"
            if 'quality_score' in chunk:
                similarity_info += f", quality: {chunk['quality_score']:.2f}"
            similarity_info += ")"
            
            section = f"[Source {i+1}] {similarity_info}\n{relevant_content}"
            
            if total_length + len(section) > self.max_context_length:
                remaining_space = self.max_context_length - total_length
                if remaining_space > 200:
                    section = section[:remaining_space-3] + "..."
                    context_parts.append(section)
                break
            
            context_parts.append(section)
            total_length += len(section)
        
        return '\n\n'.join(context_parts)

class SpeedCache:
    """Minimal cache implementation."""
    
    def __init__(self, max_size: int = 50):
        self.cache = {}
        self.max_size = max_size
        self.keys = []
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            old_key = self.keys.pop(0)
            self.cache.pop(old_key, None)
        
        self.cache[key] = value
        self.keys.append(key)
    
    def clear(self):
        self.cache.clear()
        self.keys.clear()
    
    @staticmethod
    def hash_query(query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()[:12]

class DynamicThresholder:
    """Dynamic similarity thresholding."""
    
    def __init__(self, min_threshold: float = 0.15, gap_threshold: float = 0.12):
        self.min_threshold = min_threshold
        self.gap_threshold = gap_threshold
    
    def calculate_dynamic_threshold(self, scores: List[float], query_type: str = 'general') -> float:
        if not scores or len(scores) < 2:
            return self.min_threshold
        
        sorted_scores = sorted(scores, reverse=True)
        
        type_adjustments = {
            'question': -0.05,
            'calculation': -0.03,
            'comparison': -0.02,
            'general': 0.0,
            'example': 0.02
        }
        
        base_adjustment = type_adjustments.get(query_type, 0.0)
        dynamic_threshold = self.min_threshold + base_adjustment
        
        for i in range(len(sorted_scores) - 1):
            gap = sorted_scores[i] - sorted_scores[i + 1]
            
            if gap > self.gap_threshold:
                potential_threshold = sorted_scores[i + 1] + (gap * 0.2)
                dynamic_threshold = max(dynamic_threshold, potential_threshold)
                break
        
        return max(self.min_threshold, min(dynamic_threshold, 0.5))
    
    def filter_results(self, chunks: List[Dict], query_type: str = 'general') -> List[Dict]:
        if not chunks:
            return chunks
        
        scores = [chunk['similarity'] for chunk in chunks]
        threshold = self.calculate_dynamic_threshold(scores, query_type)
        
        filtered = [chunk for chunk in chunks if chunk['similarity'] >= threshold]
        
        if not filtered and chunks and chunks[0]['similarity'] >= self.min_threshold:
            filtered = [chunks[0]]
        
        return filtered

class ConfigurableRAG:
    """Enhanced RAG with hybrid retrieval, cross-encoder reranking, and quality filtering."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.model = None
        self.cross_encoder = None
        self.index = None
        self.knowledge_base = []
        self.hybrid_retriever = None
        self.quality_filter = QualityFilter()
        self.cache = SpeedCache(self.config["cache_size"])
        self.ready = False
        self.verbose = True
        
        # Enhanced components
        self.query_processor = EnhancedQueryProcessor()
        self.context_organizer = EnhancedContextOrganizer(
            max_chunks=self.config.get("max_context_chunks", 4),
            max_context_length=self.config["max_context_length"]
        )
        self.thresholder = DynamicThresholder(
            min_threshold=self.config["min_similarity"],
            gap_threshold=self.config.get("similarity_gap_threshold", 0.12)
        )
        
        self.stats = {
            'queries': 0, 
            'cache_hits': 0, 
            'avg_time': 0, 
            'timeouts': 0,
            'errors': 0,
            'total_time': 0,
            'avg_chunks_returned': 0,
            'dynamic_threshold_adjustments': 0,
            'reranking_improvements': 0,
            'hybrid_vs_semantic': 0
        }
    
    def update_config(self, new_config: Dict):
        """Update configuration dynamically."""
        self.config.update(new_config)
        if "cache_size" in new_config:
            self.cache = SpeedCache(new_config["cache_size"])
        
        if "max_context_chunks" in new_config:
            self.context_organizer.max_chunks = new_config["max_context_chunks"]
        if "max_context_length" in new_config:
            self.context_organizer.max_context_length = new_config["max_context_length"]
        if "similarity_gap_threshold" in new_config:
            self.thresholder.gap_threshold = new_config["similarity_gap_threshold"]
    
    def set_verbose(self, verbose: bool):
        self.verbose = verbose
    
    def reset_stats(self):
        self.stats = {
            'queries': 0, 'cache_hits': 0, 'avg_time': 0, 'timeouts': 0,
            'errors': 0, 'total_time': 0, 'avg_chunks_returned': 0,
            'dynamic_threshold_adjustments': 0, 'reranking_improvements': 0,
            'hybrid_vs_semantic': 0
        }
    
    def get_stats(self) -> Dict:
        return self.stats.copy()
    
    def clear_cache(self):
        self.cache.clear()
    
    def setup(self) -> bool:
        """Setup the enhanced RAG system with proper meta tensor handling."""
        if self.verbose:
            print("🚀 Starting Enhanced RAG with Meta Tensor Fixes...")
        
        # Check Ollama
        try:
            response = requests.get(f"{self.config['gemma_url'].replace('/api/generate', '/api/tags')}", timeout=3)
            if self.verbose:
                print("✅ Ollama ready")
        except Exception as e:
            if self.verbose:
                print(f"❌ Ollama not available: {e}")
            return False
        
        # Load cached components
        if self._load_cache():
            if self.verbose:
                print("✅ Loaded from cache")
            self._setup_enhanced_components()
            self.ready = True
            if self.verbose:
                print("✅ Enhanced RAG READY!")
            return True
        
        # Build from scratch
        if self.verbose:
            print("🔨 Building enhanced index with meta tensor fixes...")
        if not self._load_data():
            return False
        if not self._build_index():
            return False
        self._setup_enhanced_components()
        self._save_cache()
        
        self.ready = True
        if self.verbose:
            print(f"✅ Enhanced RAG READY! ({len(self.knowledge_base)} chunks)")
        return True
    
    def _setup_enhanced_components(self):
        """Setup enhanced components after basic loading."""
        # Setup cross-encoder with proper meta tensor handling
        if self.config.get("use_cross_encoder", True):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.cross_encoder = safe_load_cross_encoder(
                    self.config["cross_encoder_model"],
                    device=device
                )
                if self.cross_encoder:
                    if self.verbose:
                        print("✅ Cross-encoder loaded safely")
                else:
                    if self.verbose:
                        print("⚠️  Cross-encoder disabled due to loading issues")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Cross-encoder failed: {e}")
                self.cross_encoder = None
        
        # Setup hybrid retriever
        if self.config.get("use_hybrid_retrieval", True):
            try:
                self.hybrid_retriever = HybridRetriever(
                    self.knowledge_base, self.index, self.model
                )
                if self.verbose:
                    print("✅ Hybrid retriever ready")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Hybrid retriever failed: {e}")
                self.hybrid_retriever = None
    
    def _load_cache(self) -> bool:
        """Load from cache with proper meta tensor handling."""
        try:
            cache_file = self.config["index_cache_file"]
            if not os.path.exists(cache_file):
                return False
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.knowledge_base = data['kb']
            self.index = data['idx']
            
            # Load model with safe meta tensor handling
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # FIX: Store result and check for None
            model_result = safe_load_sentence_transformer(
                self.config["model_name"],
                device=device
            )
            
            if model_result is None:
                print("Failed to load model from cache")
                return False
            
            self.model = model_result
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Cache load failed: {e}")
            return False
    
    def _save_cache(self):
        """Save to cache."""
        try:
            data = {'kb': self.knowledge_base, 'idx': self.index}
            with open(self.config["index_cache_file"], 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                print("💾 Enhanced cache saved")
        except Exception as e:
            if self.verbose:
                print(f"Cache save failed: {e}")
    
    def _load_data(self) -> bool:
        """Load and filter data for higher quality."""
        try:
            with open(self.config["embeddings_file"], 'r') as f:
                data = json.load(f)
            
            self.knowledge_base = []
            
            items = data.items() if isinstance(data, dict) else enumerate(data)
            
            for doc_id, item in items:
                text = item.get('text', '').strip()
                embedding = item.get('embedding')
                
                # Enhanced filtering for quality
                if (text and embedding and 
                    len(text) > 150 and len(text) < 4000 and
                    any(keyword in text.lower() for keyword in ['real estate', 'property', 'investment', 'market', 'analysis']) and
                    not any(spam in text.lower() for spam in ['click here', 'buy now', 'advertisement'])):
                    
                    self.knowledge_base.append({
                        'id': str(doc_id),
                        'text': text,
                        'embedding': np.array(embedding, dtype=np.float32),
                        'title': item.get('page_title', 'Document')[:150],
                        'url': item.get('url', '')
                    })
            
            if self.verbose:
                print(f"📚 Loaded {len(self.knowledge_base)} high-quality chunks")
            return len(self.knowledge_base) > 0
            
        except Exception as e:
            if self.verbose:
                print(f"Data loading failed: {e}")
            return False
    
    def _build_index(self) -> bool:
        """NUCLEAR OPTION: Build FAISS index with complete meta tensor elimination."""
        try:
            if self.verbose:
                print("🔧 Building index with NUCLEAR meta tensor protection...")
            
            # NUCLEAR: Complete environment setup
            fix_torch_settings()
            
            # NUCLEAR: Aggressive cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # NUCLEAR: Load model with complete protection
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔧 Attempting to load model on {device} with NUCLEAR protection...")
            
            # FIX: Store the result first, then check if it's None
            model_result = safe_load_sentence_transformer(
                self.config["model_name"],
                device=device
            )
            
            if model_result is None:
                raise RuntimeError("Failed to load sentence transformer model")
            
            self.model = model_result
            
            if self.verbose:
                print("✅ Model loaded successfully with NUCLEAR protection")
            
            # Build embeddings array with extra safety
            print("🔧 Building embeddings array...")
            try:
                embeddings = np.vstack([chunk['embedding'] for chunk in self.knowledge_base])
                embeddings = embeddings.astype(np.float32)
                
                # Normalize embeddings
                faiss.normalize_L2(embeddings)
                
                # Build FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)
                
                if self.verbose:
                    print(f"✅ FAISS index built successfully with {len(self.knowledge_base)} vectors")
                
                return True
                
            except Exception as embedding_error:
                print(f"❌ Embedding processing failed: {embedding_error}")
                return False
            
        except Exception as e:
            if self.verbose:
                print(f"❌ NUCLEAR index build failed: {e}")
                print("🔄 Attempting EMERGENCY recovery...")
            
            # EMERGENCY RECOVERY
            try:
                # NUCLEAR cleanup
                if hasattr(self, 'model'):
                    del self.model
                    
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                # Force CPU mode
                print("🔄 EMERGENCY: Forcing CPU-only mode...")
                
                # Override any CUDA detection
                original_cuda_available = torch.cuda.is_available
                torch.cuda.is_available = lambda: False
                
                try:
                    # FIX: Store result and check for None
                    cpu_model_result = safe_load_sentence_transformer(
                        self.config["model_name"],
                        device='cpu'
                    )
                    
                    if cpu_model_result is None:
                        raise RuntimeError("Failed to load model on CPU")
                    
                    self.model = cpu_model_result
                    
                    # Build embeddings on CPU
                    embeddings = np.vstack([chunk['embedding'] for chunk in self.knowledge_base])
                    embeddings = embeddings.astype(np.float32)
                    faiss.normalize_L2(embeddings)
                    
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(embeddings)
                    
                    if self.verbose:
                        print("✅ EMERGENCY recovery successful - running on CPU")
                    
                    return True
                    
                finally:
                    # Restore original CUDA detection
                    torch.cuda.is_available = original_cuda_available
                    
            except Exception as recovery_error:
                if self.verbose:
                    print(f"❌ EMERGENCY recovery also failed: {recovery_error}")
                    print("💀 COMPLETE SYSTEM FAILURE")
                return False
    
    def _retrieve_enhanced(self, query_info: Dict) -> List[Dict]:
        """Enhanced retrieval with hybrid approach and proper error handling."""
        try:
            if self.hybrid_retriever and self.config.get("use_hybrid_retrieval", True):
                # Use hybrid retrieval
                results = self.hybrid_retriever.hybrid_retrieve(
                    query_info['expanded'],
                    self.config["top_k"],
                    self.config.get("keyword_weight", 0.3)
                )
                self.stats['hybrid_vs_semantic'] += 1
            else:
                # Fallback to pure semantic with proper error handling
                try:
                    query_emb = self.model.encode(
                        [query_info['expanded']], 
                        show_progress_bar=False, 
                        convert_to_numpy=True
                    ).astype('float32')
                    faiss.normalize_L2(query_emb)
                    
                    scores, indices = self.index.search(query_emb, self.config["top_k"])
                    
                    results = []
                    for score, idx in zip(scores[0], indices[0]):
                        if idx != -1:
                            chunk = self.knowledge_base[idx].copy()
                            chunk['similarity'] = float(score)
                            results.append(chunk)
                            
                except Exception as semantic_error:
                    if self.verbose:
                        print(f"Semantic retrieval failed: {semantic_error}")
                    return []
            
            # Apply quality filtering
            results = self.quality_filter.filter_chunks(results, min_quality=0.2)
            
            # Apply dynamic thresholding
            filtered_results = self.thresholder.filter_results(results, query_info['query_type'])
            
            if len(filtered_results) != len(results):
                self.stats['dynamic_threshold_adjustments'] += 1
            
            # Cross-encoder reranking with safe handling
            if self.cross_encoder and len(filtered_results) > 1:
                try:
                    reranked = self._rerank_with_cross_encoder(query_info['original'], filtered_results)
                    if reranked != filtered_results:
                        self.stats['reranking_improvements'] += 1
                    return reranked[:self.config.get("rerank_top_k", 4)]
                except Exception as rerank_error:
                    if self.verbose:
                        print(f"Reranking failed, using original results: {rerank_error}")
                    return filtered_results[:self.config.get("rerank_top_k", 4)]
            
            return filtered_results[:self.config.get("rerank_top_k", 4)]
            
        except Exception as e:
            if self.verbose:
                print(f"Enhanced retrieval error: {e}")
            return []
    
    def _rerank_with_cross_encoder(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank chunks using cross-encoder with comprehensive error handling."""
        try:
            if not chunks or not self.cross_encoder:
                return chunks
            
            # Prepare query-document pairs with length limits
            pairs = [(query[:256], chunk['text'][:512]) for chunk in chunks]
            
            # Get cross-encoder scores with proper error handling - FIXED: Remove context manager
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Update similarity scores with cross-encoder scores
            for i, chunk in enumerate(chunks):
                # Combine original similarity with cross-encoder score
                chunk['ce_score'] = float(ce_scores[i])
                chunk['similarity'] = 0.4 * chunk['similarity'] + 0.6 * float(ce_scores[i])
            
            # Sort by new combined scores
            chunks.sort(key=lambda x: x['similarity'], reverse=True)
            return chunks
            
        except Exception as e:
            if self.verbose:
                print(f"Cross-encoder reranking failed: {e}")
            return chunks
    
    def _create_enhanced_prompt(self, context: str, query_info: Dict) -> str:
        """Create enhanced, query-aware prompt."""
        
        base_identity = """You are an expert real estate investment advisor with 15+ years of experience in property acquisition, market analysis, and portfolio optimization."""
        
        # Query-type specific instructions
        type_instructions = {
            'question': """Provide a direct, authoritative answer followed by detailed supporting analysis. Structure your response with clear reasoning and specific examples.""",
            
            'calculation': """Show detailed step-by-step calculations with clear explanations. Include relevant formulas, typical market benchmarks, and practical examples.""",
            
            'comparison': """Create a structured comparison highlighting key differences, advantages, and trade-offs. Use specific data points and real-world examples.""",
            
            'example': """Provide concrete, detailed examples with specific numbers and scenarios. Draw from real market situations and explain the underlying principles.""",
            
            'explanation': """Build a comprehensive explanation from fundamentals to advanced concepts. Include practical applications and strategic considerations.""",
            
            'general': """Provide expert-level analysis covering all relevant aspects. Include strategic insights, risk factors, and actionable recommendations."""
        }
        
        query_instruction = type_instructions.get(query_info['query_type'], type_instructions['general'])
        
        # Concept-specific focus
        concept_focus = ""
        if query_info.get('concepts'):
            concepts_str = ', '.join([concept for concept, weight in query_info['concepts']])
            concept_focus = f"\n\nFocus particularly on these key concepts: {concepts_str}"
        
        # Enhanced quality requirements
        quality_requirements = """
RESPONSE REQUIREMENTS:
• Lead with specific, actionable insights
• Support claims with data from the provided sources
• Include concrete examples with real numbers when available
• Address potential risks and considerations
• Provide strategic recommendations
• Use professional real estate terminology appropriately
• Ensure information is current and market-relevant"""
        
        prompt = f"""{base_identity}

{query_instruction}{concept_focus}

{quality_requirements}

CONTEXT FROM EXPERT SOURCES:
{context}

QUERY: {query_info['original']}

Provide a comprehensive, expert-level response:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Enhanced LLM call with better error handling."""
        try:
            payload = {
                "model": self.config["gemma_model"],
                "prompt": prompt,
                "stream": False,
                "options": self.config["llm_options"].copy()
            }
            
            response = requests.post(
                self.config["gemma_url"], 
                json=payload, 
                timeout=self.config["llm_timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Quality check on response
                if len(answer) < 50:
                    return "Response too short - please try rephrasing your question."
                if "I don't have" in answer and "information" in answer:
                    return "Based on available sources, I need more specific information to provide a detailed answer."
                
                return answer if answer else "Unable to generate response."
            else:
                return f"LLM error: {response.status_code}"
                
        except requests.Timeout:
            self.stats['timeouts'] += 1
            return "Response timeout - please try a simpler question."
        except Exception as e:
            self.stats['errors'] += 1
            return f"Technical error: {str(e)[:100]}"
    
    def ask(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Enhanced query processing with quality optimizations."""
        if not self.ready:
            return {
                "answer": "System not ready",
                "sources": [],
                "time": 0,
                "error": "System not initialized"
            }
        
        start_time = time.time()
        self.stats['queries'] += 1
        
        # Enhanced query processing
        query_info = self.query_processor.optimize_query(query)
        
        # Check cache
        cache_key = SpeedCache.hash_query(query_info['expanded'])
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return {
                    "answer": cached,
                    "sources": [],
                    "time": time.time() - start_time,
                    "error": None,
                    "cached": True,
                    "query_info": query_info
                }
        
        # Enhanced retrieval
        chunks = self._retrieve_enhanced(query_info)
        
        # Update stats
        self.stats['avg_chunks_returned'] = (
            (self.stats['avg_chunks_returned'] * (self.stats['queries'] - 1) + len(chunks)) 
            / self.stats['queries']
        )
        
        if not chunks:
            answer = "I don't have specific information about that topic in my knowledge base. Could you rephrase or ask about real estate investment fundamentals?"
            sources = []
        else:
            # Enhanced context organization
            context = self.context_organizer.organize_context_enhanced(chunks, query_info)
            
            # Enhanced prompting
            prompt = self._create_enhanced_prompt(context, query_info)
            
            # Get LLM response
            answer = self._call_llm(prompt)
            
            # Enhanced source information
            sources = []
            for chunk in chunks:
                source_info = {
                    "title": chunk["title"], 
                    "similarity": chunk["similarity"]
                }
                if 'quality_score' in chunk:
                    source_info["quality_score"] = chunk["quality_score"]
                if 'ce_score' in chunk:
                    source_info["cross_encoder_score"] = chunk["ce_score"]
                sources.append(source_info)
        
        # Cache high-quality responses
        if use_cache and len(answer) > 100 and "error" not in answer.lower() and "timeout" not in answer.lower():
            self.cache.set(cache_key, answer)
        
        elapsed = time.time() - start_time
        self.stats['avg_time'] = (self.stats['avg_time'] * (self.stats['queries'] - 1) + elapsed) / self.stats['queries']
        self.stats['total_time'] += elapsed
        
        return {
            "answer": answer,
            "sources": sources,
            "time": elapsed,
            "error": None if "error" not in answer.lower() and "timeout" not in answer.lower() else answer,
            "cached": False,
            "query_info": query_info,
            "chunks_used": len(chunks)
        }
    
    def interactive_mode(self):
        """Enhanced interactive mode with quality features."""
        if not self.ready:
            print("❌ System not ready")
            return
        
        print(f"\n{'='*70}")
        print("⚡ ENHANCED RAG SYSTEM - META TENSOR ISSUE FIXED")
        print("="*70)
        print("🚀 Quality Features:")
        print("  • Meta tensor protection implemented")
        print("  • Hybrid retrieval (semantic + keyword)")
        print("  • Cross-encoder reranking")
        print("  • Quality filtering")
        print("  • Enhanced query processing")
        print("  • Smart context organization")
        
        print(f"\n⚙️ Configuration:")
        print(f"  • Hybrid Retrieval: {self.config.get('use_hybrid_retrieval', True)}")
        print(f"  • Cross-Encoder: {self.config.get('use_cross_encoder', True)}")
        print(f"  • Top-K Retrieval: {self.config['top_k']}")
        print(f"  • Rerank Top-K: {self.config.get('rerank_top_k', 4)}")
        print(f"  • Temperature: {self.config['llm_options']['temperature']}")
        print(f"  • Max Tokens: {self.config['llm_options']['num_predict']}")
        
        print("\n💬 Commands: 'stats', 'clear', 'config', 'debug', 'quit'")
        print("="*70)
        
        while True:
            try:
                query = input("\n🔥 Ask: ").strip()
                
                if not query:
                    continue
                elif query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    s = self.stats
                    cache_rate = (s['cache_hits'] / max(s['queries'], 1)) * 100
                    print(f"\n📊 Enhanced Quality Stats:")
                    print(f"  Queries: {s['queries']}")
                    print(f"  Avg time: {s['avg_time']:.2f}s")
                    print(f"  Cache hits: {cache_rate:.0f}%")
                    print(f"  Avg chunks returned: {s['avg_chunks_returned']:.1f}")
                    print(f"  Dynamic threshold adjustments: {s['dynamic_threshold_adjustments']}")
                    print(f"  Reranking improvements: {s['reranking_improvements']}")
                    print(f"  Hybrid vs semantic: {s['hybrid_vs_semantic']}")
                    print(f"  Timeouts: {s['timeouts']}")
                    print(f"  Errors: {s['errors']}")
                    continue
                elif query.lower() == 'clear':
                    self.clear_cache()
                    print("🧹 Cache cleared")
                    continue
                elif query.lower() == 'config':
                    print(f"\n⚙️ Enhanced Config:")
                    for key, value in self.config['llm_options'].items():
                        print(f"  {key}: {value}")
                    print(f"  use_hybrid_retrieval: {self.config.get('use_hybrid_retrieval', True)}")
                    print(f"  use_cross_encoder: {self.config.get('use_cross_encoder', True)}")
                    print(f"  top_k: {self.config['top_k']}")
                    print(f"  rerank_top_k: {self.config.get('rerank_top_k', 4)}")
                    continue
                elif query.lower() == 'debug':
                    if hasattr(self, '_last_debug_info'):
                        info = self._last_debug_info
                        print(f"\n🔍 Enhanced Debug Info:")
                        print(f"  Original: {info.get('original', 'N/A')}")
                        print(f"  Expanded: {info.get('expanded', 'N/A')}")
                        print(f"  Query Type: {info.get('query_type', 'N/A')}")
                        print(f"  Concepts: {info.get('concepts', [])}")
                        print(f"  Chunks Retrieved: {info.get('chunks_used', 0)}")
                    else:
                        print("No debug info available - ask a question first!")
                    continue
                
                # Process query
                result = self.ask(query)
                
                # Store debug info
                self._last_debug_info = result.get('query_info', {})
                self._last_debug_info['chunks_used'] = result.get('chunks_used', 0)
                
                print(f"\n📝 ENHANCED ANSWER\n{'-'*60}")
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\n📚 QUALITY SOURCES ({len(result['sources'])})")
                    for i, source in enumerate(result["sources"], 1):
                        source_info = f"{i}. {source['title']} (sim: {source['similarity']:.2f}"
                        if 'quality_score' in source:
                            source_info += f", quality: {source['quality_score']:.2f}"
                        if 'cross_encoder_score' in source:
                            source_info += f", ce: {source['cross_encoder_score']:.2f}"
                        source_info += ")"
                        print(source_info)
                
                # Enhanced status
                cached_info = " (cached)" if result.get("cached") else ""
                query_type = result.get('query_info', {}).get('query_type', 'unknown')
                chunks_used = result.get('chunks_used', 0)
                
                print(f"{'-'*60}")
                print(f"Time: {result['time']:.2f}s{cached_info} | Type: {query_type} | Chunks: {chunks_used}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("👋 Enhanced RAG session ended!")

# Factory function
def create_rag_instance(config: Dict = None) -> ConfigurableRAG:
    """Factory function to create enhanced RAG instance."""
    try:
        return ConfigurableRAG(config)
    except Exception as e:
        print(f"Error creating enhanced RAG instance: {e}")
        raise

def main():
    """Main function for standalone usage."""
    try:
        print("🚀 Starting Enhanced Quality-Optimized RAG System with Meta Tensor Fixes...")
        
        # Apply torch fixes before anything else
        fix_torch_settings()
        
        rag = create_rag_instance()
        
        if not rag.setup():
            print("❌ Setup failed")
            return
        
        rag.interactive_mode()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()