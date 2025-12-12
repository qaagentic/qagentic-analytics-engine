"""Service for analyzing test failure text and extracting meaningful patterns."""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

logger = logging.getLogger(__name__)


class TextAnalysisService:
    """Service for analyzing test failure text and extracting patterns."""

    def __init__(self):
        """Initialize the text analysis service."""
        logger.info("Initializing text analysis service")
        try:
            # Ensure NLTK resources are downloaded
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"Could not initialize NLTK components: {str(e)}")
            # Set defaults if NLTK fails to download
            self.stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it'])
            self.lemmatizer = None
            
    async def extract_error_patterns(self, error_messages: List[str]) -> List[Dict[str, Any]]:
        """
        Extract common patterns from a list of error messages.
        
        Args:
            error_messages: List of error messages to analyze
            
        Returns:
            List of identified patterns with descriptions and examples
        """
        if not error_messages:
            return []
            
        # Extract common keywords
        keyword_patterns = await self._extract_keywords(error_messages)
        
        # Extract regex patterns for common formats like:
        # - Exception types
        # - Error codes
        # - File paths and line numbers
        regex_patterns = await self._extract_regex_patterns(error_messages)
        
        return keyword_patterns + regex_patterns
        
    async def _extract_keywords(self, error_messages: List[str]) -> List[Dict[str, Any]]:
        """Extract common keywords from error messages."""
        # Tokenize and normalize all messages
        all_tokens = []
        
        for msg in error_messages:
            tokens = word_tokenize(msg.lower()) if self.lemmatizer else msg.lower().split()
            # Remove stop words and punctuation
            tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
            # Lemmatize tokens
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            all_tokens.extend(tokens)
            
        # Count token frequency
        token_counts = Counter(all_tokens)
        
        # Find most common tokens (appearing in at least 30% of messages)
        min_count = max(2, len(error_messages) * 0.3)
        common_tokens = [token for token, count in token_counts.items() if count >= min_count]
        
        if not common_tokens:
            return []
            
        # Create a pattern from the common tokens
        pattern_desc = " ".join(common_tokens[:5])  # Use top 5 tokens for description
        
        # Find examples that contain the pattern
        examples = []
        for msg in error_messages:
            msg_lower = msg.lower()
            if all(token in msg_lower for token in common_tokens[:3]):  # Check if top 3 tokens are in message
                examples.append(msg)
                if len(examples) >= 3:  # Limit to 3 examples
                    break
                    
        return [{
            "pattern_type": "keyword",
            "description": f"Common keywords: {pattern_desc}",
            "confidence": min(0.9, max(0.5, len(common_tokens) / 10)),
            "examples": examples,
        }]
        
    async def _extract_regex_patterns(self, error_messages: List[str]) -> List[Dict[str, Any]]:
        """Extract common regex patterns from error messages."""
        patterns = []
        
        # Look for exception types
        exception_pattern = r'([A-Z][a-zA-Z]+(?:Error|Exception|Failure))'
        exceptions = []
        for msg in error_messages:
            matches = re.findall(exception_pattern, msg)
            exceptions.extend(matches)
            
        exception_counts = Counter(exceptions)
        common_exceptions = [ex for ex, count in exception_counts.items() 
                            if count >= max(2, len(error_messages) * 0.3)]
                            
        if common_exceptions:
            # Find examples with the common exception
            examples = []
            for ex in common_exceptions[:2]:  # Use top 2 exceptions
                for msg in error_messages:
                    if ex in msg:
                        examples.append(msg)
                        if len(examples) >= 3:  # Limit to 3 examples
                            break
                            
            patterns.append({
                "pattern_type": "exception",
                "description": f"Common exceptions: {', '.join(common_exceptions)}",
                "confidence": min(0.9, max(0.6, len(common_exceptions) / len(error_messages))),
                "examples": examples[:3],  # Limit to 3 examples
            })
            
        # Look for file paths
        file_pattern = r'(/[\w\-./]+\.\w+)'
        files = []
        for msg in error_messages:
            matches = re.findall(file_pattern, msg)
            files.extend(matches)
            
        file_counts = Counter(files)
        common_files = [f for f, count in file_counts.items() 
                       if count >= max(2, len(error_messages) * 0.3)]
                       
        if common_files:
            # Find examples with the common file path
            examples = []
            for file in common_files[:2]:  # Use top 2 file paths
                for msg in error_messages:
                    if file in msg:
                        examples.append(msg)
                        if len(examples) >= 3:  # Limit to 3 examples
                            break
                            
            patterns.append({
                "pattern_type": "file_path",
                "description": f"Common file paths: {', '.join(common_files)}",
                "confidence": min(0.9, max(0.6, len(common_files) / len(error_messages))),
                "examples": examples[:3],  # Limit to 3 examples
            })
            
        return patterns
        
    async def extract_locations(self, stack_traces: List[str]) -> List[Dict[str, Any]]:
        """
        Extract file paths and line numbers from stack traces.
        
        Args:
            stack_traces: List of stack traces to analyze
            
        Returns:
            List of identified file locations
        """
        if not stack_traces:
            return []
            
        # Match common stack trace patterns like:
        # - at ClassX.methodY(File.java:123)
        # - File "path/to/file.py", line 456, in function_name
        # - /path/to/file.js:789:12
        java_pattern = r'at\s+([\w.]+)\(([\w./]+):(\d+)\)'
        python_pattern = r'File\s+"([\w./]+)",\s+line\s+(\d+),\s+in\s+([\w_]+)'
        js_pattern = r'([\w./]+):(\d+)(?::(\d+))?'
        
        locations = []
        location_set = set()  # To avoid duplicates
        
        for trace in stack_traces:
            # Try Java pattern
            for match in re.finditer(java_pattern, trace):
                class_method, file_path, line_number = match.groups()
                class_name = class_method.split('.')[-2] if '.' in class_method else None
                method_name = class_method.split('.')[-1] if '.' in class_method else class_method
                
                location = (file_path, line_number, method_name, class_name)
                if location not in location_set:
                    location_set.add(location)
                    locations.append({
                        "file_path": file_path,
                        "line_number": int(line_number),
                        "method_name": method_name,
                        "class_name": class_name,
                    })
                    
            # Try Python pattern
            for match in re.finditer(python_pattern, trace):
                file_path, line_number, function_name = match.groups()
                
                location = (file_path, line_number, function_name, None)
                if location not in location_set:
                    location_set.add(location)
                    locations.append({
                        "file_path": file_path,
                        "line_number": int(line_number),
                        "method_name": function_name,
                        "class_name": None,
                    })
                    
            # Try JS pattern
            for match in re.finditer(js_pattern, trace):
                file_path, line_number, column = match.groups()
                
                location = (file_path, line_number, None, None)
                if location not in location_set:
                    location_set.add(location)
                    locations.append({
                        "file_path": file_path,
                        "line_number": int(line_number),
                        "method_name": None,
                        "class_name": None,
                    })
                    
        # Sort locations by frequency of appearance
        location_counts = Counter([loc["file_path"] for loc in locations])
        locations.sort(key=lambda x: location_counts.get(x["file_path"], 0), reverse=True)
        
        return locations[:5]  # Return the top 5 locations
