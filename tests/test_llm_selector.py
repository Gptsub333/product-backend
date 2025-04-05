import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.llm_selector import run

def test_llm_selector():
    input_data = {
        "llm": "claude_3_5_sonnet",
        "text": "What are the benefits of edge computing?"
    }

    result = run(input_data)
    print("LLM Response:", result.get("response", result))

if __name__ == "__main__":
    test_llm_selector()
