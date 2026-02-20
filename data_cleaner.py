import pandas as pd
import re

raw_csv_file = "./data/supreme_court_cases_part1.csv"
cleaned_file = "./data/data.txt"

df = pd.read_csv(raw_csv_file)
print(f"Successfully loaded {len(df)} cases")

def clean_html_text(text):
    text = re.sub(r"<.*?>", " ", str(text))   
    text = re.sub(r"\s+", " ", text).strip()  
    return text

df["cleaned_text"] = df["full_text"].apply(clean_html_text)

df["cleaned_text"].to_csv(cleaned_file, index=False, header=False)

print(f"Successfully saved {len(df)} cleaned cases to {cleaned_file}.")
