from transformers import pipeline
summarizer = pipeline("summarization")

def run(input_data: dict):
    prompt = input_data["prompt"]
    summary = summarizer(prompt, max_length=50, min_length=25, do_sample=False)
    return {"summary": summary[0]['summary_text']}
