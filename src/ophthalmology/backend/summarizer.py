from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from .config import CLASS_MAP  # Import CLASS_MAP for future label mapping if needed

# Load model and tokenizer only once
_tokenizer = T5Tokenizer.from_pretrained("t5-base")
_model = T5ForConditionalGeneration.from_pretrained("t5-base")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _model.to(_device)

# Clean and prepare text for summarization
def clean_text(text):
    text = text.strip().replace("\n", " ")
    return "summarize: " + text

# Summarize a clinical note
def summarize_note(note, max_input_len=512, max_output_len=64):
    _model.eval()
    input_text = clean_text(note)
    input_ids = _tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    ).to(_device)
    # Generate summary
    summary_ids = _model.generate(
        input_ids,
        max_length=max_output_len,
        num_beams=4,
        early_stopping=True
    )
    output = _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

# Optional: test block for direct script usage
if __name__ == "__main__":
    note = """
    The patient is a 65-year-old male with a 10-year history of type 2 diabetes.\nHe reports blurred vision and floaters in the right eye. Fundus exam shows scattered microaneurysms and cotton wool spots. HbA1c levels are poorly controlled.
    """
    summary = summarize_note(note)
    print("Summary:", summary)
