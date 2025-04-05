from modules.prompt_generator import run

def test_prompt_generator():
    input_data = {
        "text": "Explain the importance of data privacy in modern web applications."
    }

    result = run(input_data)
    print("Generated Prompt:", result.get("prompt", result))

if __name__ == "__main__":
    test_prompt_generator()