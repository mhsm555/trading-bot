import pandas as pd
import os
import sys

def debug_excel_file():
    input_file = 'data/btc_daily_2010_2025_processed.xlsx'
    output_file = 'data/btc_debug_report.xlsx'
    
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return

    print(f"--- 🕵️‍♂️ INSPECTING {input_file} ---")
    df = pd.read_excel(input_file)
    
    # 1. CLEANING (Same as training script)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.dropna(inplace=True)
    
    # 2. CREATE VERIFICATION COLUMNS
    # We bring "Tomorrow's Close" into the current row so you can see them side-by-side
    df['close_today'] = df['close']
    df['close_tomorrow'] = df['close'].shift(-1)
    
    # Calculate the actual % move to the next day
    df['actual_future_move_%'] = ((df['close_tomorrow'] - df['close_today']) / df['close_today']) * 100
    
    # 3. CHECK FOR ERRORS
    # Error A: Target is 1 (Buy), but Price actually went DOWN or stayed flat
    # (Note: We allow a tiny tolerance for floating point math)
    errors = df[ (df['target'] == 1) & (df['actual_future_move_%'] <= 0) ]
    
    if not errors.empty:
        print(f"⚠️ WARNING: Found {len(errors)} LOGIC ERRORS!")
        print("Rows where Target=1 but price did NOT go up:")
        print(errors[['timestamp', 'close_today', 'close_tomorrow', 'actual_future_move_%', 'target']].head())
    else:
        print("✅ Logic Check Passed: Every 'Target=1' row is followed by a price increase.")

    # 4. SAVE READABLE REPORT
    # We rearrange columns to make it easy to read in Excel
    cols = [
        'timestamp', 
        'close_today', 
        'close_tomorrow', 
        'actual_future_move_%', 
        'target',
        'RSI' # Adding RSI for context
    ]
    
    # Save only valid rows (drop the last row which has no 'tomorrow')
    report_df = df[cols].dropna()
    
    print(f"\n💾 Saving Audit Report to {output_file}...")
    report_df.to_excel(output_file, index=False)
    print("Done! Open this file to see side-by-side comparisons.")

if __name__ == "__main__":
    debug_excel_file()