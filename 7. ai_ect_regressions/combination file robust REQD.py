import pandas as pd
import os
import re
from pathlib import Path

# Define the working directory relative to this repository.
wd = Path(__file__).resolve().parent / "outputs_multiple_robustness"

# Define the valid group names (only these 8 folders)
VALID_GROUPS = [
    'capital structure',
    'cash flow', 
    'efficiency',
    'employee and per share',
    'growth',
    'liquidity',
    'market valuation',
    'profitability'
]

# Function to add significance stars to p-values (modifies in place)
def add_significance_stars(p_value):
    """
    Add significance stars to p-values:
    * for p < 0.1
    ** for p < 0.05
    *** for p < 0.01
    """
    if pd.isna(p_value):
        return p_value
    
    # If it's already a string with stars, extract numeric
    if isinstance(p_value, str):
        # Remove existing stars
        p_value_clean = re.sub(r'[\*]+', '', p_value)
        try:
            p_value_num = float(p_value_clean)
        except:
            return p_value
    else:
        p_value_num = p_value
    
    # Format with scientific notation for very small numbers
    if p_value_num < 0.0001:
        formatted_p = f"{p_value_num:.2e}"
    else:
        formatted_p = f"{p_value_num:.4f}"
    
    if p_value_num < 0.01:
        return f"{formatted_p}***"
    elif p_value_num < 0.05:
        return f"{formatted_p}**"
    elif p_value_num < 0.1:
        return f"{formatted_p}*"
    else:
        return formatted_p

# Function to determine highlight color based on p-values and available columns
def get_highlight_color(row, has_sentiment=False):
    """Return color based on p-value thresholds"""
    
    # Extract numeric value for count p-value
    p_count_str = row.get('my_lex_p_count')
    p_count_num = None
    
    if p_count_str is not None:
        if isinstance(p_count_str, str):
            p_count_clean = re.sub(r'[\*]+', '', p_count_str)
            try:
                p_count_num = float(p_count_clean)
            except:
                pass
        elif isinstance(p_count_str, (int, float)):
            p_count_num = p_count_str
    
    # If no sentiment columns (frequency-only file)
    if not has_sentiment:
        if p_count_num is not None and p_count_num < 0.1:
            return 'yellow'
        return None
    
    # If sentiment columns exist (sentiment file)
    p_sent_str = row.get('my_lex_p_sent')
    p_sent_num = None
    
    if p_sent_str is not None:
        if isinstance(p_sent_str, str):
            p_sent_clean = re.sub(r'[\*]+', '', p_sent_str)
            try:
                p_sent_num = float(p_sent_clean)
            except:
                pass
        elif isinstance(p_sent_str, (int, float)):
            p_sent_num = p_sent_str
    
    # Check conditions for sentiment file
    if p_count_num is not None and p_sent_num is not None:
        if p_count_num < 0.1 and p_sent_num < 0.1:
            return 'orange'
        elif p_count_num < 0.1:
            return 'yellow'
    
    return None

# Function to create highlighted Excel file with conditional coloring
def create_highlighted_excel(df, output_path, sheet_name):
    """Create Excel file with highlighting based on available columns"""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill
        
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        
        # Define fills
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        orange_fill = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")
        
        # Check if this is a sentiment file (has my_lex_p_sent column)
        has_sentiment = 'my_lex_p_sent' in df.columns
        
        # Write headers
        for col_idx, col_name in enumerate(df.columns, 1):
            ws.cell(row=1, column=col_idx, value=col_name)
        
        # Write data and highlight rows
        for row_idx, row in enumerate(df.iterrows(), 2):
            row_data = row[1]
            
            # Determine highlight color based on file type
            highlight_color = get_highlight_color(row_data, has_sentiment)
            
            # Write each cell
            for col_idx, value in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                if highlight_color == 'yellow':
                    cell.fill = yellow_fill
                elif highlight_color == 'orange':
                    cell.fill = orange_fill
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        wb.save(output_path)
        return True
    except Exception as e:
        print(f"  Warning: Could not create Excel highlighting: {e}")
        return False

# Collect all CSV files from subdirectories
all_data = []
print("Collecting data from all folders...")

# Walk through all subdirectories
for root, dirs, files in os.walk(wd):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            # Get the immediate parent folder name and normalize
            folder_name = os.path.basename(root).strip().lower()
            
            # Only process if folder_name is in VALID_GROUPS
            if folder_name not in VALID_GROUPS:
                print(f"  Skipping invalid group: {folder_name} (from {file_path})")
                continue
            
            try:
                df_temp = pd.read_csv(file_path)
                
                # Filter to only include FE_robust models (exclude contemporary_only and predictive_only)
                if 'model_choice' in df_temp.columns:
                    df_temp = df_temp[df_temp['model_choice'].isin(['contemporary_FE_robust', 'predictive_FE_robust'])]
                    
                    if len(df_temp) == 0:
                        print(f"  Skipping {folder_name}/{file}: No FE_robust models found")
                        continue
                
                # Add folder name as group column (keep original case for display)
                df_temp.insert(0, 'group', folder_name.title())
                all_data.append(df_temp)
                print(f"  Loaded: {folder_name}/{file} ({len(df_temp)} FE_robust rows)")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

if not all_data:
    print("\nNo valid CSV files found in the 8 required folders!")
    print(f"Expected folders: {VALID_GROUPS}")
    exit()

# Combine all data
all_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal rows collected (FE_robust only): {len(all_df)}")
print(f"Unique model choices found: {all_df['model_choice'].unique()}")

# ============================================
# 1. Create contemporary_fe_robust.csv (frequency only - model_choice == 'contemporary_FE_robust')
# ============================================
contemp_data = all_df[all_df['model_choice'] == 'contemporary_FE_robust'].copy()

if len(contemp_data) > 0:
    # Separate by lexicon_choice for frequency
    my_lex_data = contemp_data[contemp_data['lexicon_choice'] == 'my'][['group', 'financial_metric', 'beta_count', 'p_count']]
    gpt_lex_data = contemp_data[contemp_data['lexicon_choice'] == 'gpt'][['group', 'financial_metric', 'beta_count', 'p_count']]
    
    # Rename columns for frequency
    my_lex_data = my_lex_data.rename(columns={
        'beta_count': 'my_lex_beta_count',
        'p_count': 'my_lex_p_count'
    })
    
    gpt_lex_data = gpt_lex_data.rename(columns={
        'beta_count': 'gpt_lex_beta_count',
        'p_count': 'gpt_lex_p_count'
    })
    
    # Merge on group and financial_metric
    contemp_df = pd.merge(my_lex_data, gpt_lex_data, on=['group', 'financial_metric'], how='outer')
    
    # Reorder columns
    contemp_df = contemp_df[['group', 'financial_metric', 'my_lex_beta_count', 'my_lex_p_count', 
                            'gpt_lex_beta_count', 'gpt_lex_p_count']]
    
    # Add significance stars to p-value columns
    contemp_df['my_lex_p_count'] = contemp_df['my_lex_p_count'].apply(add_significance_stars)
    contemp_df['gpt_lex_p_count'] = contemp_df['gpt_lex_p_count'].apply(add_significance_stars)
    
    # Save contemporary_fe_robust.csv
    contemp_csv_path = os.path.join(wd, "contemporary_fe_robust.csv")
    contemp_df.to_csv(contemp_csv_path, index=False)
    print(f"\n✓ Created: contemporary_fe_robust.csv ({len(contemp_df)} rows)")
    
    # Create highlighted Excel for contemporary_fe_robust.csv
    contemp_excel_path = os.path.join(wd, "contemporary_fe_robust.xlsx")
    if create_highlighted_excel(contemp_df, contemp_excel_path, "Contemporary FE Robust"):
        print(f"✓ Created: contemporary_fe_robust.xlsx (yellow highlighting for my_lex_p_count < 0.1)")
else:
    print(f"\n✗ No contemporary_FE_robust model data found")
    contemp_df = pd.DataFrame()

# ============================================
# 2. Create contemporary_fe_robust_sent.csv (with sentiment - model_choice == 'contemporary_FE_robust')
# ============================================
if len(contemp_data) > 0:
    # Separate by lexicon_choice for sentiment
    my_lex_sent = contemp_data[contemp_data['lexicon_choice'] == 'my'][['group', 'financial_metric', 'beta_sent', 'p_sent']]
    gpt_lex_sent = contemp_data[contemp_data['lexicon_choice'] == 'gpt'][['group', 'financial_metric', 'beta_sent', 'p_sent']]
    
    # Rename columns for sentiment
    my_lex_sent = my_lex_sent.rename(columns={
        'beta_sent': 'my_lex_beta_sent',
        'p_sent': 'my_lex_p_sent'
    })
    
    gpt_lex_sent = gpt_lex_sent.rename(columns={
        'beta_sent': 'gpt_lex_beta_sent',
        'p_sent': 'gpt_lex_p_sent'
    })
    
    # Merge sentiment on group and financial_metric
    contemp_sent_df = pd.merge(my_lex_sent, gpt_lex_sent, on=['group', 'financial_metric'], how='outer')
    
    # Merge frequency and sentiment data
    contemp_full_df = pd.merge(contemp_df, contemp_sent_df, on=['group', 'financial_metric'], how='outer')
    
    # Reorder columns appropriately
    contemp_full_df = contemp_full_df[[
        'group', 'financial_metric',
        'my_lex_beta_count', 'my_lex_p_count',
        'gpt_lex_beta_count', 'gpt_lex_p_count',
        'my_lex_beta_sent', 'my_lex_p_sent',
        'gpt_lex_beta_sent', 'gpt_lex_p_sent'
    ]]
    
    # Add significance stars to sentiment p-value columns
    contemp_full_df['my_lex_p_sent'] = contemp_full_df['my_lex_p_sent'].apply(add_significance_stars)
    contemp_full_df['gpt_lex_p_sent'] = contemp_full_df['gpt_lex_p_sent'].apply(add_significance_stars)
    
    # Save contemporary_fe_robust_sent.csv
    contemp_sent_csv_path = os.path.join(wd, "contemporary_fe_robust_sent.csv")
    contemp_full_df.to_csv(contemp_sent_csv_path, index=False)
    print(f"✓ Created: contemporary_fe_robust_sent.csv ({len(contemp_full_df)} rows)")
    
    # Create highlighted Excel for contemporary_fe_robust_sent.csv
    contemp_sent_excel_path = os.path.join(wd, "contemporary_fe_robust_sent.xlsx")
    if create_highlighted_excel(contemp_full_df, contemp_sent_excel_path, "Contemporary FE Robust Sentiment"):
        print(f"✓ Created: contemporary_fe_robust_sent.xlsx (yellow: count p<0.1, orange: count AND sentiment p<0.1)")
else:
    print(f"\n✗ No contemporary_FE_robust model data for sentiment file")

# ============================================
# 3. Create predictive_fe_robust.csv (frequency only - model_choice == 'predictive_FE_robust')
# ============================================
predictive_data = all_df[all_df['model_choice'] == 'predictive_FE_robust'].copy()

if len(predictive_data) > 0:
    # Separate by lexicon_choice for frequency
    my_lex_data_pred = predictive_data[predictive_data['lexicon_choice'] == 'my'][['group', 'financial_metric', 'beta_count', 'p_count']]
    gpt_lex_data_pred = predictive_data[predictive_data['lexicon_choice'] == 'gpt'][['group', 'financial_metric', 'beta_count', 'p_count']]
    
    # Rename columns for frequency
    my_lex_data_pred = my_lex_data_pred.rename(columns={
        'beta_count': 'my_lex_beta_score',
        'p_count': 'my_lex_p_score'
    })
    
    gpt_lex_data_pred = gpt_lex_data_pred.rename(columns={
        'beta_count': 'gpt_lex_beta_score',
        'p_count': 'gpt_lex_p_score'
    })
    
    # Merge on group and financial_metric
    predictive_df = pd.merge(my_lex_data_pred, gpt_lex_data_pred, on=['group', 'financial_metric'], how='outer')
    
    # Reorder columns
    predictive_df = predictive_df[['group', 'financial_metric', 'my_lex_beta_score', 'my_lex_p_score', 
                                  'gpt_lex_beta_score', 'gpt_lex_p_score']]
    
    # Add significance stars to p-value columns
    predictive_df['my_lex_p_score'] = predictive_df['my_lex_p_score'].apply(add_significance_stars)
    predictive_df['gpt_lex_p_score'] = predictive_df['gpt_lex_p_score'].apply(add_significance_stars)
    
    # Save predictive_fe_robust.csv
    predictive_csv_path = os.path.join(wd, "predictive_fe_robust.csv")
    predictive_df.to_csv(predictive_csv_path, index=False)
    print(f"✓ Created: predictive_fe_robust.csv ({len(predictive_df)} rows)")
    
    # Create highlighted Excel for predictive_fe_robust.csv
    predictive_excel_path = os.path.join(wd, "predictive_fe_robust.xlsx")
    if create_highlighted_excel(predictive_df, predictive_excel_path, "Predictive FE Robust"):
        print(f"✓ Created: predictive_fe_robust.xlsx (yellow highlighting for my_lex_p_score < 0.1)")
else:
    print(f"\n✗ No predictive_FE_robust model data found")
    predictive_df = pd.DataFrame()

# ============================================
# 4. Create predictive_fe_robust_sent.csv (with sentiment - model_choice == 'predictive_FE_robust')
# ============================================
if len(predictive_data) > 0:
    # Separate by lexicon_choice for sentiment
    my_lex_sent_pred = predictive_data[predictive_data['lexicon_choice'] == 'my'][['group', 'financial_metric', 'beta_sent', 'p_sent']]
    gpt_lex_sent_pred = predictive_data[predictive_data['lexicon_choice'] == 'gpt'][['group', 'financial_metric', 'beta_sent', 'p_sent']]
    
    # Rename columns for sentiment
    my_lex_sent_pred = my_lex_sent_pred.rename(columns={
        'beta_sent': 'my_lex_beta_sent',
        'p_sent': 'my_lex_p_sent'
    })
    
    gpt_lex_sent_pred = gpt_lex_sent_pred.rename(columns={
        'beta_sent': 'gpt_lex_beta_sent',
        'p_sent': 'gpt_lex_p_sent'
    })
    
    # Merge sentiment on group and financial_metric
    predictive_sent_df = pd.merge(my_lex_sent_pred, gpt_lex_sent_pred, on=['group', 'financial_metric'], how='outer')
    
    # Merge frequency and sentiment data
    predictive_full_df = pd.merge(predictive_df, predictive_sent_df, on=['group', 'financial_metric'], how='outer')
    
    # Reorder columns appropriately
    predictive_full_df = predictive_full_df[[
        'group', 'financial_metric',
        'my_lex_beta_score', 'my_lex_p_score',
        'gpt_lex_beta_score', 'gpt_lex_p_score',
        'my_lex_beta_sent', 'my_lex_p_sent',
        'gpt_lex_beta_sent', 'gpt_lex_p_sent'
    ]]
    
    # Add significance stars to sentiment p-value columns
    predictive_full_df['my_lex_p_sent'] = predictive_full_df['my_lex_p_sent'].apply(add_significance_stars)
    predictive_full_df['gpt_lex_p_sent'] = predictive_full_df['gpt_lex_p_sent'].apply(add_significance_stars)
    
    # Save predictive_fe_robust_sent.csv
    predictive_sent_csv_path = os.path.join(wd, "predictive_fe_robust_sent.csv")
    predictive_full_df.to_csv(predictive_sent_csv_path, index=False)
    print(f"✓ Created: predictive_fe_robust_sent.csv ({len(predictive_full_df)} rows)")
    
    # Create highlighted Excel for predictive_fe_robust_sent.csv
    predictive_sent_excel_path = os.path.join(wd, "predictive_fe_robust_sent.xlsx")
    if create_highlighted_excel(predictive_full_df, predictive_sent_excel_path, "Predictive FE Robust Sentiment"):
        print(f"✓ Created: predictive_fe_robust_sent.xlsx (yellow: score p<0.1, orange: score AND sentiment p<0.1)")
else:
    print(f"\n✗ No predictive_FE_robust model data for sentiment file")

# Print summary
print("\n" + "="*60)
print("SUMMARY - ROBUSTNESS CHECKS (FE_ROBUST MODELS ONLY)")
print("="*60)
print(f"✓ contemporary_fe_robust.csv + .xlsx: {len(contemp_df)} rows (frequency only, yellow highlighting)")
print(f"✓ contemporary_fe_robust_sent.csv + .xlsx: {len(contemp_full_df)} rows (frequency + sentiment, yellow/orange highlighting)")
print(f"✓ predictive_fe_robust.csv + .xlsx: {len(predictive_df)} rows (frequency only, yellow highlighting)")
print(f"✓ predictive_fe_robust_sent.csv + .xlsx: {len(predictive_full_df)} rows (frequency + sentiment, yellow/orange highlighting)")

# Show model counts
print("\nModel counts in source data:")
print(f"  contemporary_FE_robust: {len(contemp_data)} rows")
print(f"  predictive_FE_robust: {len(predictive_data)} rows")

print("\n✅ All robustness files created successfully!")
print("\nNote: Significance stars: * p<0.1, ** p<0.05, *** p<0.01")
print("      Excel highlighting:")
print("        - Yellow: my_lex_p_score < 0.1 only (all files)")
print("        - Orange: my_lex_p_score < 0.1 AND my_lex_p_sent < 0.1 (sentiment files only)")
print("\nNote: Only models with 'contemporary_FE_robust' and 'predictive_FE_robust' are included")
print("      Models 'contemporary_only' and 'predictive_only' are excluded as requested")
