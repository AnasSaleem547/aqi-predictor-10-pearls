# Unified AQI Hopsworks Pipeline

A comprehensive air quality index (AQI) pipeline that fetches data from OpenWeather API, calculates EPA-style AQI, performs advanced feature engineering, and manages data in Hopsworks Feature Store for Karachi, Pakistan.

## 🚀 Features

- **Dual Pipeline Modes**: Backfill (12 months) and Hourly updates
- **Advanced Feature Engineering**: 14 production features + 1 target variable
- **EPA AQI Calculation**: NowCast algorithm for PM pollutants
- **Data Quality Validation**: Missing data thresholds, imputation, outlier handling
- **Hopsworks Integration**: Feature store management with explicit schema
- **Timezone Preservation**: Pakistan Standard Time (PKT) maintained throughout
- **GitHub Actions Ready**: Designed for automated execution

## 📊 Feature Set

### Core Pollutants (5 features)
- `pm2_5_nowcast`: PM2.5 with NowCast algorithm (µg/m³)
- `pm10_nowcast`: PM10 with NowCast algorithm (µg/m³)
- `co_ppm_8hr_avg`: Carbon monoxide 8-hour average (ppm)
- `no2_ppb`: Nitrogen dioxide (ppb)
- `so2_ppb`: Sulfur dioxide (ppb)

### Engineered Features (4 features)
- `pm2_5_pm10_ratio`: PM2.5 to PM10 ratio
- `traffic_index`: Combined NO2 and CO indicator
- `total_pm`: Sum of PM2.5 and PM10
- `pm_weighted`: Weighted PM (PM2.5 × 0.7 + PM10 × 0.3)

### Temporal Features (4 features)
- `hour`: Hour of day (0-23)
- `is_rush_hour`: Rush hour indicator (7-9 AM, 6-8 PM)
- `is_weekend`: Weekend indicator (0/1)
- `season`: Season based on Karachi climate (0-3)

### Lag Features (2 features)
- `pm2_5_nowcast_lag_1h`: PM2.5 from 1 hour ago
- `no2_ppb_lag_1h`: NO2 from 1 hour ago

### Target Variable
- `aqi_epa_calc`: EPA-calculated AQI (maximum of all pollutant AQIs)

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- OpenWeather API key
- Hopsworks account and API key

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file with required credentials:
   ```env
   OPENWEATHER_API_KEY=your_openweather_api_key
   HOPSWORKS_API_KEY=your_hopsworks_api_key
   HOPSWORKS_PROJECT_NAME=your_hopsworks_project_name
   ```

## 🚀 Usage

### Command Line Interface

#### Backfill Pipeline (12 months of historical data)
```bash
python unified_aqi_hopsworks_pipeline.py backfill
```

#### Hourly Pipeline (latest data update)
```bash
python unified_aqi_hopsworks_pipeline.py hourly
```

#### Help/Information
```bash
python unified_aqi_hopsworks_pipeline.py
```

### Pipeline Behavior

#### Backfill Pipeline
- Fetches 12 months of historical data
- Creates new Hopsworks feature group (overwrites if exists)
- Processes all data with feature engineering
- Uploads complete dataset to Hopsworks

#### Hourly Pipeline
- Checks if feature group exists (runs backfill if not)
- Fetches last 3 hours of data for lag feature calculation
- Processes only the most recent hour
- Appends new data to existing feature group

## 🤖 GitHub Actions Integration

### Workflow Examples

#### Hourly Updates
```yaml
name: Hourly AQI Update
on:
  schedule:
    - cron: '0 * * * *'  # Every hour
  workflow_dispatch:

jobs:
  update-aqi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Run hourly pipeline
        env:
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          HOPSWORKS_PROJECT_NAME: ${{ secrets.HOPSWORKS_PROJECT_NAME }}
        run: python unified_aqi_hopsworks_pipeline.py hourly
```

#### Weekly Backfill (Maintenance)
```yaml
name: Weekly AQI Backfill
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM
  workflow_dispatch:

jobs:
  backfill-aqi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Run backfill pipeline
        env:
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          HOPSWORKS_PROJECT_NAME: ${{ secrets.HOPSWORKS_PROJECT_NAME }}
        run: python unified_aqi_hopsworks_pipeline.py backfill
```

### Required GitHub Secrets
Add these secrets to your GitHub repository:
- `OPENWEATHER_API_KEY`: Your OpenWeather API key
- `HOPSWORKS_API_KEY`: Your Hopsworks API key  
- `HOPSWORKS_PROJECT_NAME`: Your Hopsworks project name

## 🔧 Configuration

### Data Quality Settings
```python
MISSING_DATA_THRESHOLD = 0.30  # 30% missing data tolerance
API_RETRY_ATTEMPTS = 3         # Number of API retry attempts
API_RETRY_DELAY = 5           # Seconds between retries
```

### Feature Group Settings
```python
FEATURE_GROUP_NAME = "karachifeatures"
FEATURE_GROUP_VERSION = 1
```

### Location Settings
```python
LAT, LON = 24.8546842, 67.0207055  # Karachi coordinates
PKT = zoneinfo.ZoneInfo("Asia/Karachi")  # Pakistan timezone
```

## 📈 Data Quality Checks

### Validation Rules
- **Missing Data**: Max 30% missing for core pollutants
- **Target Variable**: Max 50% missing for AQI target
- **API Retry**: 3 attempts with 5-second delays
- **Imputation**: Zero values imputed using historical Hopsworks data
- **Outliers**: Handled automatically in EPA AQI calculations

### Imputation Strategy
1. Fetch historical data from Hopsworks (last 7 days)
2. Use median of historical values for imputation
3. Fallback to forward/backward fill if no historical data
4. Zero values treated as missing and imputed

## 🧪 Testing

### Feature Engineering Tests
```bash
python test_feature_engineering.py
```

### Data Processing Tests
```bash
python test_data_processing.py
```

## 📁 File Structure

```
├── unified_aqi_hopsworks_pipeline.py  # Main pipeline script
├── test_feature_engineering.py       # Feature engineering tests
├── test_data_processing.py           # Data processing tests
├── requirements.txt                  # Python dependencies
├── README_unified_pipeline.md        # This documentation
└── .env                              # Environment variables (create this)
```

## 🔍 Monitoring & Debugging

### Exit Codes
- `0`: Success
- `1`: Failure (check logs for details)

### Logging
The pipeline provides detailed console output with:
- ✅ Success indicators
- ❌ Error messages
- ⚠️ Warnings
- 📊 Data statistics
- 🕐 Timestamps

### Common Issues
1. **API Rate Limits**: Handled with retry logic
2. **Missing Credentials**: Check `.env` file
3. **Hopsworks Connection**: Verify API key and project name
4. **Data Quality**: Check missing data thresholds

## 🌍 Timezone Handling

All timestamps are maintained in Pakistan Standard Time (PKT, UTC+5):
- API data converted from UTC to PKT
- Hopsworks storage preserves timezone information
- Feature engineering uses local time context

## 📊 Hopsworks Schema

The pipeline creates a feature group with explicit data types:
- `datetime`: timestamp (primary key, event time)
- Pollutant features: float
- Temporal features: int
- Target variable: int

## 🚨 Error Handling

### API Failures
- Automatic retry with exponential backoff
- Graceful degradation for partial data
- Detailed error logging

### Data Quality Issues
- Validation before upload
- Imputation for missing values
- Rejection of poor-quality batches

### Hopsworks Issues
- Connection retry logic
- Feature group existence checks
- Schema validation

## 📞 Support

For issues or questions:
1. Check the console output for detailed error messages
2. Verify environment variables are set correctly
3. Test individual components using the test scripts
4. Review GitHub Actions logs for automation issues

## 🔄 Version History

- **v1.0**: Initial unified pipeline with 14 features
- Feature group version: 1
- Compatible with Hopsworks 3.4+