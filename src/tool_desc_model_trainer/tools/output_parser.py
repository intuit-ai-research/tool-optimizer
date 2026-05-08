#!/usr/bin/env python3
"""
Utility functions for parsing model outputs with structured tags or JSON.

Supports extracting:
- JSON format outputs (recommended for v7+ prompts)
- Chain-of-thought reasoning from <think>...</think> tags
- Final API descriptions wrapped in configurable start/end tokens
- Handling various edge cases (missing tags, malformed output, etc.)
"""

import json
import re
import os
from typing import Dict, Optional, Tuple

# Import LLM calling infrastructure
try:
    import sys
    # Add project root to path (go up 4 levels: tools/ -> policy_learn/ -> DRAFT/ -> FunctionWrapper_PL/)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from utils.llm import llm_call
    LLM_AVAILABLE = True
except ImportError as e:
    raise ImportError("LLM calling infrastructure (utils.llm.llm_call) is required for output parser") from e


DEFAULT_START_TOKEN = "<|extra_0|>"
DEFAULT_END_TOKEN = "<|extra_1|>"


def extract_json_description_and_reasoning(model_output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract description and reasoning from JSON-formatted model output.
    
    Handles cases where model output may contain text before/after JSON,
    or multiple JSON objects (takes the last valid one).
    
    Args:
        model_output: Raw model output potentially containing JSON
    
    Returns:
        Tuple of (description, reasoning) where either may be None
    """
    if not model_output:
        return None, None
    
    # Strategy 1: Try to find JSON object with "description" field
    # Look for all potential JSON objects in the output
    json_candidates = []
    
    # Find all { ... } patterns (non-greedy for nested handling)
    brace_depth = 0
    start_idx = None
    
    for i, char in enumerate(model_output):
        if char == '{':
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                json_candidates.append(model_output[start_idx:i+1])
                start_idx = None
    
    # Try each candidate, prefer later ones (model's actual output vs echoed prompt)
    for json_str in reversed(json_candidates):
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "description" in parsed:
                desc = parsed.get("description")
                if desc and len(str(desc).strip()) >= 10:
                    reasoning = parsed.get("reasoning")
                    reasoning_text = str(reasoning).strip() if reasoning else None
                    return str(desc).strip(), reasoning_text or None
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Simple regex fallback for malformed JSON
    # Match "description": "..." or "description": '...'
    desc_pattern = r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"|\'description\'\s*:\s*\'((?:[^\'\\]|\\.)*)\''
    matches = list(re.finditer(desc_pattern, model_output, re.DOTALL))
    if matches:
        # Take the last match
        match = matches[-1]
        desc = match.group(1) or match.group(2)
        if desc:
            # Unescape JSON string escapes
            desc = desc.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            if len(desc.strip()) >= 10:
                return desc.strip(), None
    
    return None, None


def extract_json_description(model_output: str) -> Optional[str]:
    """
    Extract description from JSON-formatted model output.
    
    Handles cases where model output may contain text before/after JSON,
    or multiple JSON objects (takes the last valid one).
    
    Args:
        model_output: Raw model output potentially containing JSON
    
    Returns:
        Extracted description string, or None if JSON parsing fails
    """
    description, _ = extract_json_description_and_reasoning(model_output)
    return description


def extract_think_and_output(
    model_output: str,
    strip_output_header: bool = True,
    start_token: str = DEFAULT_START_TOKEN,
    end_token: str = DEFAULT_END_TOKEN,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract chain-of-thought reasoning and final output from model response.
    
    Args:
        model_output: Raw model output potentially containing <think> and <output> tags
        strip_output_header: If True, remove common headers like "**Improved API Description:**"
        start_token: Token that marks the start of the final description
        end_token: Token that marks the end of the final description
    
    Returns:
        Tuple of (chain_of_thought, improved_description)
        - chain_of_thought: Content from <think>...</think> or None
        - improved_description: Content between start/end tokens or <output> tags, or None
    """
    chain_of_thought = None
    improved_description = None
    
    # Extract chain-of-thought from <think> tags (use LAST occurrence to handle prompt echoing)
    think_matches = list(re.finditer(r"<think>(.*?)</think>", model_output, re.DOTALL | re.IGNORECASE))
    if think_matches:
        chain_of_thought = think_matches[-1].group(1).strip()
    
    # Extract improved description using start/end tokens first
    token_pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
    token_match = re.search(token_pattern, model_output, re.DOTALL | re.IGNORECASE)
    if token_match:
        improved_description = token_match.group(1).strip()
    else:
        # Fallback: Extract from legacy <output> tags
        # Strategy: Find all </output> tags, then look for content after the placeholder
        output_closing_tags = list(re.finditer(r'</output>\s*>*', model_output, re.IGNORECASE))
        
        if len(output_closing_tags) >= 2:
            # If we have 2+ closing tags, the real content is likely between the 2nd and last tag
            # or after the 2nd tag (model echoes prompt, then generates)
            second_close_pos = output_closing_tags[1].end()
            
            # Extract everything after the second </output> tag
            content_after_second = model_output[second_close_pos:].strip()
            
            # Look for the final </output> to bound the content
            final_close_match = re.search(r'</output>', content_after_second, re.IGNORECASE)
            if final_close_match:
                # Content is between 2nd and final </output>
                improved_description = content_after_second[:final_close_match.start()].strip()
            else:
                # No final tag, use everything after 2nd tag
                improved_description = content_after_second
            
            # Clean up trailing artifacts
            improved_description = re.sub(r'>+\}*$', '', improved_description).strip()
            
        elif len(output_closing_tags) == 1:
            # Only one closing tag - try traditional extraction
            output_match = re.search(r"<output>(.*?)</output>", model_output, re.DOTALL | re.IGNORECASE)
            if output_match:
                improved_description = output_match.group(1).strip()
    
    # Strip headers if we got content
    if strip_output_header and improved_description:
        header_patterns = [
            r"^\*\*Improved API Description[^:]*:\*\*\s*",
            r"^\*\*API Description[^:]*:\*\*\s*",
            r"^Improved API Description[^:]*:\s*",
            r"^API Description[^:]*:\s*",
        ]
        for pattern in header_patterns:
            improved_description = re.sub(pattern, "", improved_description, flags=re.IGNORECASE)
            improved_description = improved_description.strip()
    
    # Strip start/end tokens if still present
    if improved_description:
        improved_description = _strip_tokens(improved_description, start_token, end_token)
    
    # Final check: Is this placeholder text?
    is_placeholder = (
        not improved_description or 
        improved_description in ['...', ''] or
        improved_description.startswith('[Your improved') or
        improved_description.startswith('[Analyze:') or
        improved_description.startswith('tags.') or
        len(improved_description) < 50
    )
    
    if is_placeholder:
        improved_description = None
    
    return chain_of_thought, improved_description


def extract_improved_description(
    model_output: str,
    fallback_to_raw: bool = True,
    strip_output_header: bool = True,
    use_llm_extraction: bool = True,
    extraction_model: str = "gpt-41-2025-04-14",
    start_token: str = DEFAULT_START_TOKEN,
    end_token: str = DEFAULT_END_TOKEN,
    prefer_json: bool = True,
) -> str:
    """
    Extract improved API description from model output.
    
    Tries extraction methods in order:
    1. JSON format (if prefer_json=True) - fastest, most reliable for v7+ prompts
    2. Start/end token format - for v1-v6 prompts
    3. LLM-based extraction (if use_llm_extraction=True) - fallback for malformed output

    Args:
        model_output: Raw model output
        fallback_to_raw: If True and extraction fails, return raw output
        strip_output_header: If True, remove common headers
        use_llm_extraction: If True, use LLM to extract as fallback
        extraction_model: Model to use for LLM extraction
        start_token: Start delimiter token
        end_token: End delimiter token
        prefer_json: If True, try JSON extraction first (recommended for v7+ prompts)

    Returns:
        Extracted improved description
    """
    
    # Strategy 1: Try JSON extraction first (fast, no API calls)
    if prefer_json:
        json_result = extract_json_description(model_output)
        if json_result:
            return json_result
    
    # Strategy 2: Try token-based extraction
    if start_token and end_token:
        token_pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
        token_match = re.search(token_pattern, model_output, re.DOTALL | re.IGNORECASE)
        
        if token_match:
            extracted = token_match.group(1).strip()
            # Strip tokens in case of nested/malformed output
            result = _strip_tokens(extracted, start_token, end_token)
            if result and len(result.strip()) >= 10:
                return result
    
    # Strategy 3: LLM-based extraction as fallback
    if use_llm_extraction:
        return extract_improved_desc_with_llm(
            model_output,
            extraction_model,
            strip_output_header,
            start_token=start_token,
            end_token=end_token,
        )
    
    # Final fallback: return raw output if allowed
    if fallback_to_raw:
        return model_output.strip()
    
    return None

def extract_improved_desc_with_llm(
    model_output: str, 
    extraction_model: str = "gpt-41-2025-04-14", 
    strip_output_header: bool = True,
    start_token: str = DEFAULT_START_TOKEN,
    end_token: str = DEFAULT_END_TOKEN,
) -> str:

    # Load extraction prompt
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'prompts',
        'extraction_prompt_v0.txt'
    )
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Extraction prompt not found: {prompt_path}")

    with open(prompt_path, 'r') as f:
        extraction_prompt_template = f.read()
    
    # Insert configurable tokens, keep model_output placeholder for later substitution
    extraction_prompt_template = extraction_prompt_template.format(
        start_token=start_token,
        end_token=end_token,
    )
    
    # Format prompt with model output
    extraction_prompt = extraction_prompt_template.replace('{{model_output}}', model_output)
    
    # Call LLM to extract
    response_obj = llm_call(
        model=extraction_model,
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0,  # Use low temperature for consistent extraction
        top_p=1.0,
        max_tokens=2000
    )
    
    # Extract content from response
    if hasattr(response_obj, 'choices') and len(response_obj.choices) > 0:
        extracted = response_obj.choices[0].message.content
    elif isinstance(response_obj, str):
        extracted = response_obj
    elif isinstance(response_obj, dict) and 'choices' in response_obj:
        extracted = response_obj['choices'][0]['message']['content']
    else:
        raise ValueError(f"Unexpected LLM response format: {type(response_obj)}")
    
    if not extracted:
        raise ValueError("LLM extraction returned empty result")
    
    # Clean up the extracted description
    extracted = extracted.strip()
    if strip_output_header:
        extracted = _strip_common_headers(extracted)
    
    # Strip start/end tokens if they're still present in the extracted text
    extracted = _strip_tokens(extracted, start_token, end_token)
    
    return extracted

def _strip_common_headers(text: str) -> str:
    """Strip common headers from extracted description."""
    header_patterns = [
        r"^\*\*Improved API Description[^:]*:\*\*\s*",
        r"^\*\*API Description[^:]*:\*\*\s*",
        r"^Improved API Description[^:]*:\s*",
        r"^API Description[^:]*:\s*",
    ]
    for pattern in header_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = text.strip()
    return text


def _strip_tokens(text: str, start_token: str, end_token: str) -> str:
    """Strip start/end tokens from text if present."""
    # Remove tokens from anywhere in the text
    text = text.replace(start_token, "")
    text = text.replace(end_token, "")
    return text.strip()


def extract_chain_of_thought(model_output: str) -> Optional[str]:
    """
    Extract only the chain-of-thought reasoning from model output.
    
    Args:
        model_output: Raw model output
    
    Returns:
        Chain-of-thought text or None if not found
    """
    chain_of_thought, _ = extract_think_and_output(model_output, strip_output_header=False)
    return chain_of_thought


def parse_model_output(
    model_output: str,
    include_metadata: bool = False,
    start_token: str = DEFAULT_START_TOKEN,
    end_token: str = DEFAULT_END_TOKEN,
) -> Dict[str, any]:
    """
    Parse model output into structured dictionary.
    
    Args:
        model_output: Raw model output
        include_metadata: If True, include parsing metadata
    
    Returns:
        Dictionary with:
        - improved_description: The extracted final description
        - chain_of_thought: The reasoning (if present)
        - has_think_tags: Boolean indicating if <think> tags were found
        - has_output_tags: Boolean indicating if start/end tokens or legacy <output> tags were found
        - raw_output: Original model output (if include_metadata=True)
    """
    chain_of_thought, improved_description = extract_think_and_output(
        model_output,
        strip_output_header=False,
        start_token=start_token,
        end_token=end_token,
    )
    
    result = {
        "improved_description": improved_description
        or extract_improved_description(
            model_output,
            start_token=start_token,
            end_token=end_token,
        ),
        "chain_of_thought": chain_of_thought,
        "has_think_tags": chain_of_thought is not None,
        "has_output_tags": (
            improved_description is not None
            or (start_token in model_output and end_token in model_output)
            or "<output>" in model_output.lower()
        ),
    }
    
    if include_metadata:
        result["raw_output"] = model_output
        result["output_length"] = len(result["improved_description"])
        result["think_length"] = len(chain_of_thought) if chain_of_thought else 0
    
    return result


def clean_description_for_evaluation(description: str) -> str:
    """
    Clean a description for use in evaluation/reward computation.
    
    Removes any lingering tags, extra whitespace, and normalizes formatting.
    
    Args:
        description: API description (potentially with tags or formatting issues)
    
    Returns:
        Cleaned description ready for evaluation
    """
    # Remove any remaining XML-like tags
    cleaned = re.sub(r"<[^>]+>", "", description)
    
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    
    # Remove common artifacts
    artifacts = [
        r"^\[final improved API description text here\]",
        r"^\(your final description\)",
    ]
    for artifact in artifacts:
        cleaned = re.sub(artifact, "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
    
    return cleaned


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Output Parser")
    print("=" * 80)
    
    # Test case 1: Complete output with start/end tokens and think tags
    test_output_1 = """
    <think>
    This API is used for discovering movies based on various filters.
    The example queries show filtering by language, year, and ratings.
    I need to document all the filter parameters clearly.
    </think>
    
    <|extra_0|>
    **Improved API Description (Your Response):**
    
    Use the `discover_movies` API to search and filter movies based on multiple criteria.
    
    **Required Parameters:**
    - None (all parameters are optional)
    
    **Optional Parameters:**
    - `language` (string, ISO 639-1): Filter by language code (e.g., "en")
    - `primary_release_year` (integer): Filter by release year
    - `vote_average.gte` (float): Minimum vote average (0-10)
    
    **Returns:** List of movie objects with details.
    <|extra_1|>
    """
    
    print("\n📝 Test 1: Complete output with both tags")
    result = parse_model_output(test_output_1, include_metadata=True)
    print(f"  Has <think> tags: {result['has_think_tags']}")
    print(f"  Has <output> tags: {result['has_output_tags']}")
    print(f"  Think length: {result['think_length']} chars")
    print(f"  Output length: {result['output_length']} chars")
    print(f"\n  Chain of thought preview:")
    print(f"  {result['chain_of_thought'][:100]}...")
    print(f"\n  Description preview:")
    print(f"  {result['improved_description'][:100]}...")
    
    # Test case 2: Output only (no think tags) using start/end tokens
    test_output_2 = """
    <|extra_0|>
    The discover_movies API allows searching for movies with optional filters.
    <|extra_1|>
    """
    
    print("\n📝 Test 2: Output only (no think tags)")
    result2 = parse_model_output(test_output_2)
    print(f"  Has <think> tags: {result2['has_think_tags']}")
    print(f"  Has <output> tags: {result2['has_output_tags']}")
    print(f"  Description: {result2['improved_description']}")
    
    # Test case 3: Raw output (no tags)
    test_output_3 = """
    The discover_movies API is used for filtering movies by various criteria
    like language, year, and ratings.
    """
    
    print("\n📝 Test 3: Raw output (no tags - fallback)")
    result3 = parse_model_output(test_output_3)
    print(f"  Has <think> tags: {result3['has_think_tags']}")
    print(f"  Has <output> tags: {result3['has_output_tags']}")
    print(f"  Description: {result3['improved_description']}")
    
    # Test case 4: Malformed output (missing end token)
    test_output_4 = """
    <think>
    Some reasoning here
    <|extra_0|>
    Incomplete description without closing tags
    """
    
    print("\n📝 Test 4: Malformed output (missing closing tags)")
    result4 = parse_model_output(test_output_4)
    print(f"  Has <think> tags: {result4['has_think_tags']}")
    print(f"  Has <output> tags: {result4['has_output_tags']}")
    print(f"  Description: {result4['improved_description'][:80]}...")
    
    # Test case 5: JSON format output (v7+ prompts)
    test_output_5 = '''{"description": "The discover_movies API searches for movies based on filters like language, year, and vote average. All parameters are optional. Use this when you need to browse movies by criteria rather than searching for a specific title."}'''
    
    print("\n📝 Test 5: JSON format output (v7+ prompts)")
    result5 = extract_json_description(test_output_5)
    print(f"  Extracted: {result5[:80]}..." if result5 else "  Extracted: None")
    
    # Test case 6: JSON with surrounding text
    test_output_6 = '''Here is the improved description:
    
{"description": "Use get_movie_details to retrieve comprehensive information about a specific movie by its ID. Required: movie_id (integer). Returns title, overview, release date, runtime, genres, and more."}

I hope this helps!'''
    
    print("\n📝 Test 6: JSON with surrounding text")
    result6 = extract_json_description(test_output_6)
    print(f"  Extracted: {result6[:80]}..." if result6 else "  Extracted: None")
    
    # Test case 7: extract_improved_description with JSON (prefer_json=True)
    print("\n📝 Test 7: extract_improved_description with JSON")
    result7 = extract_improved_description(test_output_5, use_llm_extraction=False, prefer_json=True)
    print(f"  Extracted: {result7[:80]}..." if result7 else "  Extracted: None")
    
    # Test case 8: Malformed JSON fallback to token
    test_output_8 = '''<|extra_0|>
This API gets movie details by ID.
<|extra_1|>'''
    
    print("\n📝 Test 8: Token format (JSON fails, falls back to tokens)")
    result8 = extract_improved_description(test_output_8, use_llm_extraction=False, prefer_json=True)
    print(f"  Extracted: {result8}" if result8 else "  Extracted: None")
    
    print("\n" + "=" * 80)
    print("✅ Output parser tests complete!")

