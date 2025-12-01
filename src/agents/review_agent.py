"""ReviewAgent - Evaluates draft against criteria and suggests improvements."""

from typing import Dict, Any
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.utils.prompts import REVIEW_AGENT_SYSTEM_PROMPT, get_review_prompt


def review_agent(state: EssayState, ollama_client: OllamaClient) -> Dict[str, Any]:
    """
    ReviewAgent evaluates the essay draft against criteria and provides feedback.
    
    Args:
        state: Current essay state
        ollama_client: Ollama client instance
        
    Returns:
        Updated state dictionary with review feedback and score
    """
    print("🔎 ReviewAgent: Evaluating essay...")
    
    if not state.sections:
        print("  Warning: No sections available for review")
        return {"review_feedback": [], "review_score": 0.0}
    
    try:
        # Combine sections into essay content
        essay_content = "\n\n".join([
            f"## {name}\n\n{content}"
            for name, content in state.sections.items()
        ])
        
        # Generate prompt
        prompt = get_review_prompt(state.topic, state.criteria, essay_content)
        
        # Get structured response
        review_data = ollama_client.generate_structured(
            prompt=prompt,
            system=REVIEW_AGENT_SYSTEM_PROMPT,
            temperature=0.4
        )
        
        score = review_data.get("score", 0.0)
        feedback = review_data.get("feedback", [])
        strengths = review_data.get("strengths", [])
        weaknesses = review_data.get("weaknesses", [])
        
        # Combine all feedback
        all_feedback = []
        if strengths:
            all_feedback.append("Strengths:")
            all_feedback.extend([f"  - {s}" for s in strengths])
        if weaknesses:
            all_feedback.append("Weaknesses:")
            all_feedback.extend([f"  - {w}" for w in weaknesses])
        if feedback:
            all_feedback.append("General Feedback:")
            all_feedback.extend([f"  - {f}" for f in feedback])
        
        print(f"  ✓ Review score: {score:.2f}/1.0")
        print(f"  ✓ Generated {len(all_feedback)} feedback points")
        
        return {
            "review_feedback": all_feedback,
            "review_score": score
        }
    
    except Exception as e:
        print(f"  ✗ Error in ReviewAgent: {str(e)}")
        return {"review_feedback": [], "review_score": 0.0}

