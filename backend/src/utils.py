import os

def save_llm_output(text: str, filename: str):
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"LLM output saved to: {file_path}")
    return file_path
