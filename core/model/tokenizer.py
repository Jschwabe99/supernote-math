"""Tokenizer for LaTeX math expressions."""

from typing import List, Dict, Optional
from pathlib import Path

from config import SPECIAL_TOKENS, DEFAULT_VOCAB_PATH


class MathTokenizer:
    """Tokenizer for LaTeX math expressions."""
    
    def __init__(self, vocab_file: Optional[Path] = None):
        """Initialize tokenizer with vocabulary.
        
        Args:
            vocab_file: Path to vocabulary file, one token per line
        """
        # Default vocabulary includes ASCII chars, LaTeX commands, and special tokens
        if vocab_file:
            with open(vocab_file, 'r') as f:
                self.vocab = f.read().splitlines()
        else:
            # Basic vocabulary - would be expanded in a real implementation
            self.vocab = SPECIAL_TOKENS + [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '+', '-', '=', '\\times', '\\div', '\\frac', '\\sqrt',
                '(', ')', '[', ']', '{', '}', '^', '_',
                'a', 'b', 'c', 'x', 'y', 'z', '\\alpha', '\\beta', '\\gamma',
                '\\sin', '\\cos', '\\tan', '\\log', '\\lim', '\\sum', '\\int',
                '\\infty', '\\pi', 'e'
            ]
            
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        # Special token IDs
        self.pad_id = self.token_to_id[SPECIAL_TOKENS[0]]  # <PAD>
        self.sos_id = self.token_to_id[SPECIAL_TOKENS[1]]  # <SOS>
        self.eos_id = self.token_to_id[SPECIAL_TOKENS[2]]  # <EOS>
        self.unk_id = self.token_to_id[SPECIAL_TOKENS[3]]  # <UNK>
    
    def tokenize(self, latex_string: str) -> List[str]:
        """Tokenize a LaTeX string into tokens.
        
        Args:
            latex_string: LaTeX string to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization approach - in practice would need more sophisticated parsing
        tokens = []
        i = 0
        while i < len(latex_string):
            # Check for LaTeX commands
            if latex_string[i] == '\\':
                # Find the end of the command
                j = i + 1
                while j < len(latex_string) and latex_string[j].isalpha():
                    j += 1
                command = latex_string[i:j]
                if command in self.token_to_id:
                    tokens.append(command)
                else:
                    tokens.append(SPECIAL_TOKENS[3])  # <UNK>
                i = j
            else:
                # Single character token
                if latex_string[i] in self.token_to_id:
                    tokens.append(latex_string[i])
                else:
                    tokens.append(SPECIAL_TOKENS[3])  # <UNK>
                i += 1
        
        return tokens
    
    def encode(self, latex_string: str) -> List[int]:
        """Encode a LaTeX string into token IDs.
        
        Args:
            latex_string: LaTeX string to encode
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(latex_string)
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a LaTeX string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            LaTeX string
        """
        return ''.join([self.id_to_token.get(id, SPECIAL_TOKENS[3]) for id in token_ids])