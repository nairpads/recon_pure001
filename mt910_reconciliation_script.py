
import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- MT910 Parser --------
def parse_mt910_messages(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    messages = re.split(r'(?=^:20:)', content, flags=re.MULTILINE)
    data = []

    for msg in messages:
        trx_ref = re.search(r':20:(.*)', msg)
        account = re.search(r':25:(.*)', msg)
        details = re.search(r':32A:(\d{6})([A-Z]{3})([\d,\.]+)', msg)
        narration = re.search(r':86:(.*)', msg)

        if details:
            date_str = details.group(1)
            currency = details.group(2)
            amount = details.group(3).replace(',', '.')
            try:
                date = datetime.strptime(date_str, '%y%m%d').date()
            except ValueError:
                date = None
        else:
            date, currency, amount = None, None, None

        data.append({
            'reference': trx_ref.group(1).strip() if trx_ref else '',
            'account': account.group(1).strip() if account else '',
            'date': date,
            'currency': currency,
            'amount': float(amount) if amount else None,
            'narration': narration.group(1).strip() if narration else ''
        })

    return pd.DataFrame(data)

# -------- Application Entry Reader --------
def load_application_entries(file_path):
    df = pd.read_csv(file_path)
    df['narration'] = df['narration'].astype(str).str.lower()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    return df

# -------- Reconciliation Logic --------
def reconcile(mt_df, app_df):
    mt_df['narration'] = mt_df['narration'].str.lower()
    vectorizer = TfidfVectorizer().fit(mt_df['narration'].tolist() + app_df['narration'].tolist())
    mt_vec = vectorizer.transform(mt_df['narration'])
    app_vec = vectorizer.transform(app_df['narration'])

    sim_matrix = cosine_similarity(mt_vec, app_vec)

    matched_index = []
    sim_scores = []
    matched_amount = []
    matched_date = []
    matched_narration = []

    for i, row in mt_df.iterrows():
        best_score = 0
        best_idx = -1
        for j, app_row in app_df.iterrows():
            if pd.isnull(row['date']) or pd.isnull(app_row['date']) or pd.isnull(row['amount']) or pd.isnull(app_row['amount']):
                continue

            score = sim_matrix[i, j]
            try:
                amount_match = abs(row['amount'] - app_row['amount']) <= 0.1
                date_match = abs((row['date'] - app_row['date']).days) <= 1
            except Exception:
                continue

            if score > best_score and amount_match and date_match:
                best_score = score
                best_idx = j

        if best_idx != -1:
            matched_index.append(best_idx)
            sim_scores.append(best_score)
            matched_amount.append(app_df.loc[best_idx, 'amount'])
            matched_date.append(app_df.loc[best_idx, 'date'])
            matched_narration.append(app_df.loc[best_idx, 'narration'])
        else:
            matched_index.append(None)
            sim_scores.append(0)
            matched_amount.append(None)
            matched_date.append(None)
            matched_narration.append(None)

    mt_df['matched_index'] = matched_index
    mt_df['similarity_score'] = sim_scores
    mt_df['matched_amount'] = matched_amount
    mt_df['matched_date'] = matched_date
    mt_df['matched_narration'] = matched_narration
    return mt_df

# -------- Run Reconciliation --------
def run_reconciliation(mt_file, app_file):
    mt_df = parse_mt910_messages(mt_file)
    app_df = load_application_entries(app_file)
    result_df = reconcile(mt_df, app_df)
    result_df.to_csv('mt910_reconciliation_result.csv', index=False)
    print("✅ Reconciliation complete. Output saved to mt910_reconciliation_result.csv")
