"""
Advanced Chunking Strategies for Production RAG Systems
Moving beyond basic character-based chunking to semantic-aware methods
"""

import re
from typing import List, Dict, Any
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize models
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
# For token counting (GPT-4 tokenizer as reference)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens using GPT-4 tokenizer"""
    return len(tokenizer.encode(text))

class ChunkingStrategy:
    """Base class for different chunking strategies"""
    
    def chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap (what you've been using)"""
    
    def chunk(self, text: str, max_chars: int = 500, overlap: int = 80) -> List[Dict[str, Any]]:
        text = " ".join(text.split())
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "strategy": "fixed_size",
                "start_char": start,
                "end_char": end,
                "token_count": count_tokens(chunk_text)
            })
            
            chunk_id += 1
            if end == len(text):
                break
            start = end - overlap
            
        return chunks

class TokenBasedChunking(ChunkingStrategy):
    """Token-based chunking - better for LLM context windows"""
    
    def chunk(self, text: str, max_tokens: int = 200, overlap_tokens: int = 50) -> List[Dict[str, Any]]:
        tokens = tokenizer.encode(text)
        if not tokens:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "strategy": "token_based",
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens)
            })
            
            chunk_id += 1
            if end == len(tokens):
                break
            start = end - overlap_tokens
            
        return chunks

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking - splits at semantic boundaries"""
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, text: str, max_tokens: int = 300) -> List[Dict[str, Any]]:
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            # Check if adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "strategy": "semantic",
                    "sentence_count": len(current_chunk),
                    "token_count": current_tokens
                })
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "strategy": "semantic",
                "sentence_count": len(current_chunk),
                "token_count": current_tokens
            })
        
        return chunks

class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking - tries different separators"""
    
    def __init__(self, separators: List[str] = None):
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str, max_tokens: int = 200, overlap_tokens: int = 50) -> List[Dict[str, Any]]:
        return self._recursive_split(text, self.separators, max_tokens, overlap_tokens)
    
    def _recursive_split(self, text: str, separators: List[str], max_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
        if not separators:
            # Fallback to character-based splitting
            return FixedSizeChunking().chunk(text, max_chars=max_tokens*4, overlap=overlap_tokens*4)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        sections = text.split(separator)
        
        chunks = []
        current_section = ""
        chunk_id = 0
        
        for section in sections:
            test_section = current_section + separator + section if current_section else section
            
            if count_tokens(test_section) <= max_tokens:
                current_section = test_section
            else:
                # Current section is too big, process it
                if current_section:
                    if count_tokens(current_section) > max_tokens:
                        # Recursively split with next separator
                        sub_chunks = self._recursive_split(current_section, remaining_separators, max_tokens, overlap_tokens)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({
                            "text": current_section,
                            "chunk_id": chunk_id,
                            "strategy": "recursive",
                            "separator": separator,
                            "token_count": count_tokens(current_section)
                        })
                        chunk_id += 1
                
                current_section = section
        
        # Handle the last section
        if current_section:
            if count_tokens(current_section) > max_tokens:
                sub_chunks = self._recursive_split(current_section, remaining_separators, max_tokens, overlap_tokens)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "text": current_section,
                    "chunk_id": chunk_id,
                    "strategy": "recursive",
                    "separator": separator,
                    "token_count": count_tokens(current_section)
                })
        
        return chunks

def compare_chunking_strategies(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Compare different chunking strategies on the same text"""
    strategies = {
        "fixed_size": FixedSizeChunking(),
        "token_based": TokenBasedChunking(),
        "semantic": SemanticChunking(),
        "recursive": RecursiveChunking()
    }
    
    results = {}
    for name, strategy in strategies.items():
        chunks = strategy.chunk(text)
        results[name] = chunks
        
        print(f"\n=== {name.upper()} CHUNKING ===")
        print(f"Number of chunks: {len(chunks)}")
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per chunk: {total_tokens/len(chunks):.1f}")
        
        # Show first few chunks
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk["text"][:100].replace('\n', ' ')
            print(f"Chunk {i}: [{chunk['token_count']} tokens] {preview}...")
    
    return results

def analyze_chunk_quality(chunks: List[Dict[str, Any]], query: str) -> Dict[str, float]:
    """Analyze chunk quality for a given query"""
    if not chunks:
        return {"avg_similarity": 0.0, "max_similarity": 0.0, "coverage": 0.0}
    
    query_embedding = model.encode(query, normalize_embeddings=True)
    similarities = []
    
    for chunk in chunks:
        chunk_embedding = model.encode(chunk["text"], normalize_embeddings=True)
        similarity = np.dot(chunk_embedding, query_embedding)
        similarities.append(similarity)
    
    return {
        "avg_similarity": np.mean(similarities),
        "max_similarity": np.max(similarities),
        "coverage": len([s for s in similarities if s > 0.5]) / len(similarities)
    }

if __name__ == "__main__":
    # HR Policies document for realistic comparison
    hr_policies = """
    # Employee Handbook - TechCorp Solutions
    
    ## 1. Working Hours and Attendance
    
    ### 1.1 Standard Working Hours
    Our standard working hours are Monday through Friday, 9:00 AM to 6:00 PM, with a one-hour lunch break. 
    All employees are expected to be available during core hours of 10:00 AM to 4:00 PM for meetings and collaboration.
    
    ### 1.2 Remote Work Policy
    TechCorp offers hybrid work arrangements. Employees may work remotely up to 3 days per week with manager approval.
    Remote work requires proper home office setup and reliable internet connection. All remote employees must attend
    important meetings in person when requested by management.
    
    ### 1.3 Overtime Policy
    Overtime work requires prior approval from the department head. Overtime is compensated at 1.5x the regular hourly rate
    for exempt employees. Non-exempt employees may receive compensatory time off instead of monetary compensation.
    
    ## 2. Leave Policies
    
    ### 2.1 Annual Leave
    Full-time employees are entitled to 21 days of annual leave per year. Leave accrues at 1.75 days per month.
    Employees must request leave at least 2 weeks in advance, except in emergency situations. Leave cannot be carried 
    forward beyond 6 months.
    
    ### 2.2 Sick Leave
    Employees are entitled to 12 sick days per year. Sick leave can be used for personal illness or to care for 
    immediate family members. Doctor's certificate required for sick leave exceeding 3 consecutive days.
    
    ### 2.3 Maternity and Paternity Leave
    Female employees are entitled to 26 weeks of maternity leave. Male employees are entitled to 2 weeks of paternity leave.
    Both parents can take additional 6 weeks of parental leave that can be shared between them.
    
    ## 3. Benefits and Compensation
    
    ### 3.1 Health Insurance
    TechCorp provides comprehensive health insurance covering medical, dental, and vision expenses.
    The company covers 80% of premium costs for employees and 60% for dependents. Coverage includes pre-existing conditions
    after a waiting period of 12 months.
    
    ### 3.2 Retirement Benefits
    Employees are eligible for the company's 401(k) plan after 6 months of service. The company matches 50% of 
    employee contributions up to 6% of salary. Vesting schedule is 4 years with 25% vesting per year.
    
    ### 3.3 Performance Bonuses
    Annual performance bonuses are based on individual and company performance. Bonus range is typically 10-30% 
    of base salary. Exceptional performers may receive up to 50% bonus based on management discretion.
    
    ## 4. Code of Conduct
    
    ### 4.1 Professional Behavior
    All employees must maintain professional behavior in the workplace. Harassment, discrimination, or bullying 
    will result in disciplinary action up to and including termination. Respect for diversity and inclusion is mandatory.
    
    ### 4.2 Confidentiality
    Employees must maintain confidentiality of company information, trade secrets, and client data. 
    Non-disclosure agreements must be signed by all employees. Violation of confidentiality may result in legal action.
    
    ### 4.3 Social Media Policy
    Employees should not share confidential company information on social media platforms. All social media posts 
    should reflect positively on the company brand. Personal opinions should be clearly distinguished from company positions.
    """
    
    print("🏢 CHUNKING STRATEGY COMPARISON - HR POLICIES")
    print("=" * 60)
    
    # Test questions that highlight differences
    test_questions = [
        "What is the company policy on remote work?",
        "How many sick days are employees entitled to?",
        # "What are the maternity leave benefits?",
        # "Does the company provide health insurance?",
        # "What is the overtime compensation policy?"
    ]
    
    # Compare strategies
    results = compare_chunking_strategies(hr_policies)
    
    print(f"\n📋 ANSWER QUALITY COMPARISON")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n❓ QUESTION: {question}")
        print("-" * 50)
        
        for strategy_name, chunks in results.items():
            # Find most relevant chunk
            best_chunk = None
            best_score = 0
            
            for chunk in chunks:
                chunk_embedding = model.encode(chunk["text"], normalize_embeddings=True)
                query_embedding = model.encode(question, normalize_embeddings=True)
                similarity = np.dot(chunk_embedding, query_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_chunk = chunk
            
            print(f"\n📊 {strategy_name.upper()}:")
            print(f"   Best Match Score: {best_score:.3f}")
            if best_chunk:
                # Show relevant portion of the chunk
                chunk_text = best_chunk["text"]
                if len(chunk_text) > 200:
                    chunk_text = chunk_text[:200] + "..."
                print(f"   Relevant Content: {chunk_text}")
                
                # Quality assessment
                if best_score > 0.7:
                    print(f"   ✅ High Quality - Direct answer likely")
                elif best_score > 0.5:
                    print(f"   ⚠️  Medium Quality - Partial answer likely")
                else:
                    print(f"   ❌ Low Quality - May not answer question")
            else:
                print(f"   ❌ No relevant content found")
    
    print(f"\n📈 STRATEGY PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for strategy_name, chunks in results.items():
        total_chunks = len(chunks)
        avg_tokens = sum(chunk["token_count"] for chunk in chunks) / total_chunks
        max_tokens = max(chunk["token_count"] for chunk in chunks)
        min_tokens = min(chunk["token_count"] for chunk in chunks)
        
        print(f"\n{strategy_name.upper()}:")
        print(f"   Total Chunks: {total_chunks}")
        print(f"   Avg Tokens: {avg_tokens:.1f}")
        print(f"   Token Range: {min_tokens} - {max_tokens}")
        print(f"   Strategy Type: {chunks[0]['strategy'] if chunks else 'N/A'}")
        
        # Analyze content preservation
        if strategy_name == "fixed_size":
            print(f"   ⚠️  May cut sentences/sections mid-way")
        elif strategy_name == "semantic":
            print(f"   ✅ Preserves sentence boundaries")
        elif strategy_name == "recursive":
            print(f"   ✅ Respects document structure (headers, paragraphs)")
        elif strategy_name == "token_based":
            print(f"   ✅ Optimized for LLM context windows")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Use SEMANTIC chunking for better question answering")
    print("2. Use RECURSIVE chunking for structured documents")
    print("3. Use TOKEN-BASED for cost optimization")
    print("4. Avoid FIXED-SIZE for complex documents like HR policies")