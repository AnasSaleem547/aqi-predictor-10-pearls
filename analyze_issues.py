import pandas as pd
import numpy as np

df = pd.read_csv('retrieved_karachi_aqi_features.csv')

print('=== CRITICAL ISSUES FOUND ===')
print()

print('1. DATE RANGE ANOMALY:')
print(f'   Data spans from {df["datetime"].min()} to {df["datetime"].max()}')
print('   ❌ This includes FUTURE dates (2025)! This should not happen.')
print()

print('2. SUSPICIOUS AQI JUMPS:')
# Look at the first few rows where AQI jumps dramatically
print('First 10 rows AQI values:')
for i in range(min(10, len(df))):
    print(f'   Row {i}: {df.iloc[i]["datetime"]} -> AQI: {df.iloc[i]["aqi_epa_calc"]}')
print()
print('   ❌ AQI jumps from 9 to 170 with identical PM values!')
print()

print('3. IDENTICAL PM VALUES:')
# Check if PM values are identical in first few rows
pm_cols = ['pm2_5_nowcast', 'pm10_nowcast']
for col in pm_cols:
    first_vals = df[col].head(5).tolist()
    print(f'   {col} first 5 values: {first_vals}')
    if len(set(first_vals)) == 1:
        print(f'   ❌ All values are identical! This suggests NowCast calculation issue.')
print()

print('4. MISSING ENGINEERED FEATURES:')
# Check which rows have missing engineered features
missing_ratio_rows = df[df['pm2_5_pm10_ratio'].isnull()]
if not missing_ratio_rows.empty:
    print(f'   Rows with missing PM ratio: {len(missing_ratio_rows)}')
    print(f'   These rows: {missing_ratio_rows.index.tolist()}')
print()

print('5. HIGH AQI VALUES:')
high_aqi = df[df['aqi_epa_calc'] > 300]
print(f'   Records with AQI > 300: {len(high_aqi)} out of {len(df)} ({len(high_aqi)/len(df)*100:.1f}%)')
if len(high_aqi) > 0:
    print(f'   Max AQI: {df["aqi_epa_calc"].max()}')
    print('   Sample high AQI rows:')
    sample_high = high_aqi.head(3)
    for _, row in sample_high.iterrows():
        print(f'     {row["datetime"]}: AQI={row["aqi_epa_calc"]}, PM2.5={row["pm2_5_nowcast"]:.1f}, PM10={row["pm10_nowcast"]:.1f}')

print()
print('6. NOWCAST CALCULATION VERIFICATION:')
# Check if NowCast values make sense
print('   Checking for repeated NowCast values (should vary):')
pm25_unique = df['pm2_5_nowcast'].nunique()
pm10_unique = df['pm10_nowcast'].nunique()
print(f'   PM2.5 NowCast unique values: {pm25_unique} out of {len(df)} records')
print(f'   PM10 NowCast unique values: {pm10_unique} out of {len(df)} records')

# Check for impossible ratios
print()
print('7. RATIO VALIDATION:')
valid_ratios = df[df['pm2_5_pm10_ratio'].notna()]
impossible_ratios = valid_ratios[valid_ratios['pm2_5_pm10_ratio'] > 1.0]
print(f'   PM2.5/PM10 ratios > 1.0: {len(impossible_ratios)} (should be 0)')
if len(impossible_ratios) > 0:
    print('   ❌ Found impossible ratios where PM2.5 > PM10!')