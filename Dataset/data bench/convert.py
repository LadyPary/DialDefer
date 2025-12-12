import pandas as pd

# Load your CSV
df = pd.read_csv('truthfulQA_new.csv')

# 1. Create custom ID
df['id'] = 'truthfulqa-' + df['ID'].astype(str)

# 2. Add dataset field
df['dataset'] = 'truthfulqa'

# 3. Rename existing columns to lowercase
df = df.rename(columns={
    'Type': 'type', 
    'Category': 'category', 
    'Question': 'question'
})

# 4. Transform answers into lists
df['correct_answers'] = df['Best Answer'].apply(lambda x: [x])
df['incorrect_answers'] = df['Best Incorrect Answer'].apply(lambda x: [x])

# 5. Select and reorder the final columns
final_df = df[['id', 'dataset', 'type', 'category', 'question', 'correct_answers', 'incorrect_answers']]

# Save as JSONL
final_df.to_json('truthfulQA_custom.jsonl', orient='records', lines=True, force_ascii=False)

print("Custom conversion complete!")