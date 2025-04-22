"""SymPy-based solver for mathematical expressions."""

import re
import sympy
from sympy.parsing.latex import parse_latex
from typing import Optional, Union, Dict, Any, Tuple

from config import MAX_SOLVING_TIME


class LatexParser:
    """Parser for LaTeX expressions to SymPy format."""
    
    def __init__(self):
        """Initialize LaTeX parser."""
        pass
    
    def parse_to_sympy(self, latex_string: str) -> Optional[sympy.Expr]:
        """Parse LaTeX string to SymPy expression.
        
        Args:
            latex_string: LaTeX string to parse
            
        Returns:
            SymPy expression or None if parsing failed
        """
        try:
            # Remove unnecessary LaTeX formatting
            cleaned_latex = self._clean_latex(latex_string)
            # Parse LaTeX to SymPy expression
            expr = parse_latex(cleaned_latex)
            return expr
        except Exception as e:
            print(f"Error parsing LaTeX: {e}")
            return None
    
    def _clean_latex(self, latex_string: str) -> str:
        """Clean LaTeX string for parsing.
        
        Args:
            latex_string: Original LaTeX string
            
        Returns:
            Cleaned LaTeX string
        """
        # Remove display math delimiters
        latex_string = latex_string.replace('\\[', '').replace('\\]', '')
        latex_string = latex_string.replace('\\begin{equation}', '').replace('\\end{equation}', '')
        
        # Add other cleaning rules as needed
        return latex_string


class SymPySolver:
    """Solver for mathematical expressions using SymPy."""
    
    def __init__(self, timeout: float = MAX_SOLVING_TIME):
        """Initialize solver with timeout.
        
        Args:
            timeout: Maximum time in seconds to spend on solving
        """
        self.parser = LatexParser()
        self.timeout = timeout
    
    def solve_latex(self, latex_string: str) -> Dict[str, Any]:
        """Solve a LaTeX mathematical expression.
        
        This method detects the type of expression (equation, calculation, etc.)
        and applies the appropriate solving method.
        
        Args:
            latex_string: LaTeX string to solve
            
        Returns:
            Dictionary with solution information
        """
        result = {
            'input': latex_string,
            'parsed': False,
            'solved': False,
            'result': None,
            'result_latex': None,
            'error': None,
            'solution_type': None,
        }
        
        # Parse LaTeX to SymPy expression
        try:
            expr = self.parser.parse_to_sympy(latex_string)
            if expr is None:
                result['error'] = "Failed to parse LaTeX"
                return result
                
            result['parsed'] = True
            
            # Detect expression type and solve accordingly
            if '=' in latex_string:
                # Equation solving
                result['solution_type'] = 'equation'
                solution = self._solve_equation(expr)
                result['result'] = solution
                result['solved'] = True
            else:
                # Expression evaluation
                result['solution_type'] = 'evaluation'
                value = self._evaluate_expression(expr)
                result['result'] = value
                result['solved'] = True
            
            # Convert result back to LaTeX
            if result['result'] is not None:
                result['result_latex'] = sympy.latex(result['result'])
                
        except Exception as e:
            result['error'] = f"Error solving expression: {str(e)}"
        
        return result
    
    def _solve_equation(self, expr: sympy.Expr) -> Union[sympy.Expr, Dict]:
        """Solve an equation for unknown variables.
        
        Args:
            expr: SymPy equation expression
            
        Returns:
            Solution as SymPy expression or dictionary
        """
        # This is a placeholder implementation
        # In reality, would use sympy.solve() with proper equation handling
        if isinstance(expr, sympy.Equality):
            # Extract LHS and RHS
            lhs = expr.lhs
            rhs = expr.rhs
            
            # Find free symbols (variables)
            symbols = expr.free_symbols
            if len(symbols) == 0:
                # No variables, just check if equation is true
                return sympy.sympify(lhs == rhs)
            elif len(symbols) == 1:
                # One variable, solve for it
                symbol = list(symbols)[0]
                return sympy.solve(lhs - rhs, symbol)
            else:
                # Multiple variables, solve for all
                return {str(symbol): sympy.solve(lhs - rhs, symbol) for symbol in symbols}
        else:
            # Not an equality, can't solve as equation
            return expr
    
    def _evaluate_expression(self, expr: sympy.Expr) -> sympy.Expr:
        """Evaluate a mathematical expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            Evaluated expression
        """
        # Check if expression is numeric (no free variables)
        if len(expr.free_symbols) == 0:
            # Fully numeric, evaluate to float
            return float(expr.evalf())
        else:
            # Contains variables, simplify expression
            return expr.simplify()
    
    def format_result(self, result: Dict[str, Any], fmt: str = 'latex') -> str:
        """Format the solution result in the requested format.
        
        Args:
            result: Result dictionary from solve_latex
            fmt: Output format ('latex', 'text', 'mathml')
            
        Returns:
            Formatted result string
        """
        if not result['solved']:
            return f"Error: {result['error']}" if result['error'] else "Could not solve expression"
        
        if fmt == 'latex':
            return result['result_latex'] if result['result_latex'] else sympy.latex(result['result'])
        elif fmt == 'text':
            return str(result['result'])
        elif fmt == 'mathml':
            return sympy.printing.mathml(result['result'])
        else:
            return str(result['result'])


def detect_question_in_latex(latex_string: str) -> Tuple[str, bool]:
    """Detect if LaTeX string contains a question to solve.
    
    Args:
        latex_string: LaTeX string to analyze
        
    Returns:
        Tuple of (processed_latex, is_question)
    """
    # Check for question marks
    has_question_mark = '?' in latex_string
    
    # Check for equals signs with no right-hand side
    has_empty_rhs = bool(re.search(r'=\s*$', latex_string))
    
    # Replace question marks with appropriate SymPy variables
    processed_latex = latex_string.replace('?', 'x')
    
    return processed_latex, (has_question_mark or has_empty_rhs)