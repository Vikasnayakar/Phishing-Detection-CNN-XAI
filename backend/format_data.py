import pandas as pd
import os
import logging

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def format_and_merge_data():
    """
    Scans the data folder, standardizes multiple CSV formats, 
    cleans the content, and exports a master 'emails.csv'.
    """
    data_folder = "../data/"
    
    if not os.path.exists(data_folder):
        logging.error(f"Directory {data_folder} not found! Please create it.")
        return

    all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    combined_list = []

    print(f"\n{'='*40}")
    print(f"📊 DATA MERGE START: Found {len(all_files)} source files.")
    print(f"{'='*40}")

    for file in all_files:
        # Avoid recursive merging (don't merge the output file into itself)
        if file == "emails.csv":
            continue
            
        try:
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            
            # 1. Standardize column names
            # Handles different Kaggle datasets (e.g., 'Body', 'Email Text', 'Message')
            df = df.rename(columns={
                "Email Text": "text",
                "Email Type": "label",
                "Body": "text",
                "text_combined": "text",
                "Message": "text",
                "Class": "label"
            })
            
            # 2. Label Encoding (Convert "Phishing/Safe" strings to 1/0)
            if 'label' in df.columns and df['label'].dtype == object:
                df['label'] = df['label'].apply(
                    lambda x: 1 if any(word in str(x).lower() for word in ['phishing', 'spam', 'fraud']) else 0
                )
            
            # 3. Filter only relevant columns
            if 'text' in df.columns and 'label' in df.columns:
                # Basic cleaning: Remove emails that are too short to be useful
                df = df[df['text'].str.len() > 10] 
                combined_list.append(df[['text', 'label']])
                logging.info(f"✅ Processed {file}: Added {len(df)} rows.")
            else:
                logging.warning(f"⚠️ Skipped {file}: Missing 'text' or 'label' columns.")

        except Exception as e:
            logging.error(f"❌ Error processing {file}: {e}")

    if not combined_list:
        logging.error("No valid data found to merge!")
        return

    # 4. Merge Everything
    final_df = pd.concat(combined_list, ignore_index=True)

    # 5. REMOVE DUPLICATES (Critical for AI Accuracy)
    # If the same email appears 10 times, the AI overfits.
    original_count = len(final_df)
    final_df.drop_duplicates(subset=['text'], inplace=True)
    final_df.dropna(inplace=True)
    
    # 6. Final Export
    output_path = os.path.join(data_folder, "emails.csv")
    final_df.to_csv(output_path, index=False)

    print(f"\n{'='*40}")
    print(f"🎊 SUCCESS! DATASET BUILT.")
    print(f"---")
    print(f"Total Emails Collected: {original_count}")
    print(f"Unique Emails Kept:     {len(final_df)}")
    print(f"Duplicates Removed:     {original_count - len(final_df)}")
    print(f"---")
    print(f"Phishing (1): {len(final_df[final_df['label'] == 1])}")
    print(f"Safe (0):     {len(final_df[final_df['label'] == 0])}")
    print(f"---")
    print(f"Final master file: {output_path}")
    print(f"Next step: Run 'python train_model.py'")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    format_and_merge_data()