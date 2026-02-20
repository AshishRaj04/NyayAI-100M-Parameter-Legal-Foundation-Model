import pandas as pd
import re

raw_csv_file = "delhi_hc_contract_law_cases.csv"
cleaned_csv_file = "cleaned_legal_data.csv"

df = pd.read_csv(raw_csv_file)
print(f"successfully loaded {len(df)} cases")

def clean_html_text(html_text):
    if not isinstance(html_text,str):
        return ""

    text = re.sub(r'\s+', ' ',html_text)
    text = re.sub(r'\* IN THE HIGH COURT OF DELHI AT NEW DELHI.*HON\'BLE MR\. JUSTICE \w+\s?\w+\.?', '', text, flags=re.IGNORECASE)
    return text

df["cleaned_text"] = df['full_text'].apply(clean_html_text)

clean_df = df[['title' , 'cleaned_text']]

clean_df.to_csv(cleaned_csv_file , index=False , encoding = 'utf-8')
print(f"Successfully saved {len(clean_df)} cleaned cases.")