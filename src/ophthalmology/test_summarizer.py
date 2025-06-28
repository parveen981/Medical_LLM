from summarizer import summarize_note

sample_note = """
The patient is a 65-year-old male with a 12-year history of poorly controlled type 2 diabetes.
Complains of progressive blurred vision, particularly in the right eye.
Fundus examination reveals multiple microaneurysms, hard exudates, and cotton wool spots.
No history of prior eye surgery. Blood pressure is also elevated.
Recommend urgent referral to ophthalmology and improved glycemic control.
"""

print("\U0001F4DD Original Clinical Note:")
print(sample_note)
print("\n\U0001F50D Summary:")
print(summarize_note(sample_note))
