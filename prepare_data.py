from datasets import load_dataset
import pandas as pd

print("Downloading questions and documents...")
# Load the two separate parts of the dataset
questions_ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions", split="test")
documents_ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents", split="test")

# Convert to pandas for easier merging
df_questions = questions_ds.to_pandas()
df_documents = documents_ds.to_pandas()

formatted_data = []

print("Merging data for fine-tuning...")
# Loop through the first 500 questions (perfect for a small fine-tuning run)
for index, row in df_questions.head(500).iterrows():
    question = row['question']
    best_answer = row['gold_answer']
    
    # Get the ID of the document that contains the answer
    doc_ids = row['expected_doc_ids']
    
    if len(doc_ids) > 0:
        target_doc_id = doc_ids[0] # Grab the primary document
        
        # Find the actual messy text of that document
        matching_doc = df_documents[df_documents['doc_id'] == target_doc_id]
        
        if not matching_doc.empty:
            messy_context = matching_doc.iloc[0]['content']
            
            # Format exactly how Qwen or Llama expects it for Instruction Tuning
            formatted_prompt = {
                "instruction": f"You are an enterprise AI. Answer the question using ONLY the provided company context.\n\nContext:\n{messy_context}\n\nQuestion:\n{question}",
                "response": best_answer
            }
            formatted_data.append(formatted_prompt)

# Save to a JSONL file ready for TRL SFTTrainer
df_final = pd.DataFrame(formatted_data)
df_final.to_json("enterprise_rag_finetune.jsonl", orient="records", lines=True)

print(f"Success! Saved {len(formatted_data)} formatted examples to enterprise_rag_finetune.jsonl")
