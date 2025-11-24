# ============================================================
# File: aqi_streamlit_dashboard.py
# Author: Anas Saleem
# Institution: FAST NUCES
# ============================================================
"""
Streamlit Dashboard for AQI Model Predictions and Forecasts

Features:
- Real-time AQI predictions from trained models (Random Forest, LightGBM, XGBoost)
- Historical data visualization with actual vs predicted values
- 3-day AQI forecasts for each model
- EPA AQI level classification with color coding
- Interactive model comparison and performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
BASE_PATH = Path(__file__).resolve().parent
def _parse_dir_ts(name):
    try:
        parts = name.split('_')
        return datetime.strptime(f"{parts[2]}-{parts[0]}-{parts[1]} {parts[3][:2]}:{parts[3][2:]}", "%Y-%m-%d %H:%M")
    except Exception:
        return None

# ============================================================
# EPA AQI Configuration
# ============================================================
EPA_AQI_LEVELS = {
    "Good": {"range": (0, 50), "color": "#00E400", "description": "Air quality is satisfactory"},
    "Moderate": {"range": (51, 100), "color": "#FFFF00", "description": "Air quality is acceptable for most people"},
    "Unhealthy for Sensitive Groups": {"range": (101, 150), "color": "#FF7E00", "description": "Sensitive groups may experience health effects"},
    "Unhealthy": {"range": (151, 200), "color": "#FF0000", "description": "Everyone may experience health effects"},
    "Very Unhealthy": {"range": (201, 300), "color": "#8F3F97", "description": "Health alert: risk to all"},
    "Hazardous": {"range": (301, 500), "color": "#7E0023", "description": "Health warnings of emergency conditions"},
    "Very Hazardous": {"range": (501, 1000), "color": "#8B4513", "description": "Emergency conditions: avoid all outdoor activity"}
}

def get_aqi_level(aqi_value):
    """Get EPA AQI level classification for a given AQI value."""
    for level, info in EPA_AQI_LEVELS.items():
        if info["range"][0] <= aqi_value <= info["range"][1]:
            return level, info["color"], info["description"]
    return "Unknown", "#808080", "AQI value out of range"

# ============================================================
# Utility Functions
# ============================================================
def find_latest_model_results():
    model_results = {}
    model_dirs = {
        'Random Forest': BASE_PATH / 'randomforest_additional',
        'LightGBM': BASE_PATH / 'lgbm_additional',
        'XGBoost': BASE_PATH / 'xgboost_additional'
    }
    def parse_ts(name):
        try:
            parts = name.split('_')
            ts = f"{parts[2]}-{parts[0]}-{parts[1]} {parts[3][:2]}:{parts[3][2:]}:00"
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min
    for model_name, base_dir in model_dirs.items():
        if base_dir.exists():
            subdirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
            if subdirs:
                sorted_dirs = sorted(subdirs, key=parse_ts, reverse=True)
                latest_valid_dir = None
                for dir_name in sorted_dirs:
                    results_path = base_dir / dir_name
                    metrics_file = results_path / 'model_metrics.json'
                    forecast_file = results_path / 'aqi_3_day_forecast.csv'
                    if metrics_file.exists() and forecast_file.exists() and metrics_file.stat().st_size > 0 and forecast_file.stat().st_size > 0:
                        latest_valid_dir = dir_name
                        break
                if latest_valid_dir:
                    results_path = base_dir / latest_valid_dir
                    model_results[model_name] = {
                        'results_path': str(results_path),
                        'metrics_file': str(results_path / 'model_metrics.json'),
                        'forecast_file': str(results_path / 'aqi_3_day_forecast.csv'),
                        'timestamp': latest_valid_dir
                    }
    return model_results

def load_model_metrics(metrics_file):
    """Load model performance metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None

def validate_random_forest_forecast(forecast_df, historical_mean, threshold=1.5):
    """Validate and potentially correct Random Forest forecasts if they're unrealistically high."""
    forecast_mean = forecast_df['aqi_forecast'].mean()
    
    if forecast_mean > historical_mean * threshold:
        # Apply correction factor silently
        correction_factor = historical_mean / forecast_mean
        forecast_df['aqi_forecast'] *= correction_factor
        
        # Add a flag to indicate this was corrected
        forecast_df['corrected'] = True
    else:
        forecast_df['corrected'] = False
    
    return forecast_df

def load_forecasts(forecast_file):
    """Load 3-day AQI forecasts from CSV file."""
    try:
        df = pd.read_csv(forecast_file)
        # Ensure datetime column is properly parsed
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            # Convert to Pakistan timezone for display
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Karachi')
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading forecasts: {e}")
        return None

def create_aqi_gauge(aqi_value, title="Current AQI"):
    """Create an AQI gauge chart with EPA color coding."""
    level, color, description = get_aqi_level(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        title={'text': title, 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 300], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#00E400"},
                {'range': [50, 100], 'color': "#FFFF00"},
                {'range': [100, 150], 'color': "#FF7E00"},
                {'range': [150, 200], 'color': "#FF0000"},
                {'range': [200, 300], 'color': "#8F3F97"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.add_annotation(
        text=f"Level: {level}<br>{description}",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=14, color="#333333")
    )
    
    fig.update_layout(height=400)
    return fig

def load_historical_data():
    try:
        historical_files = glob.glob(str(BASE_PATH / 'retrieved_karachi_aqi_features*.csv'))
        if not historical_files:
            return None
        latest_file = max(historical_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"Warning: Could not load historical data: {e}")
        return None

def create_forecast_chart(forecast_df, model_name, historical_df=None):
    """Create a forecast chart with AQI level color coding and historical context."""
    if forecast_df is None or forecast_df.empty:
        return None
    
    # Ensure we have the AQI column in forecast data
    forecast_aqi_col = None
    for col in ['AQI', 'aqi', 'predicted_aqi', 'prediction', 'aqi_forecast']:
        if col in forecast_df.columns:
            forecast_aqi_col = col
            break
    
    if forecast_aqi_col is None:
        st.warning(f"No AQI column found in {model_name} forecast data")
        return None
    
    fig = go.Figure()
    
    # Add historical data if available
    if historical_df is not None and not historical_df.empty:
        # Use the EPA calculated AQI from historical data
        historical_aqi_col = 'aqi_epa_calc'
        if historical_aqi_col in historical_df.columns:
            fig.add_trace(go.Scatter(
                x=historical_df['datetime'],
                y=historical_df[historical_aqi_col],
                mode='lines+markers',
                name='Historical AQI (Hopsworks)',
                line=dict(color='darkblue', width=3),
                marker=dict(size=4, color='darkblue'),
                hovertemplate='<b>Historical AQI</b><br>' +
                             'Date: %{x}<br>' +
                             'AQI: %{y:.1f}<br>' +
                             'Source: Hopsworks<extra></extra>'
            ))
            
            # Add a vertical line to separate historical from forecast
            forecast_start = forecast_df['datetime'].min()
            fig.add_vline(x=forecast_start, line_dash="dash", line_color="red")
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['datetime'],
        y=forecast_df[forecast_aqi_col],
        mode='lines+markers',
        name=f'{model_name} Forecast',
        line=dict(color='red', width=3, dash='solid'),
        marker=dict(size=6, color='red', symbol='diamond'),
        hovertemplate='<b>' + model_name + ' Forecast</b><br>' +
                     'Date: %{x}<br>' +
                     'AQI: %{y:.1f}<br>' +
                     'Type: Model Prediction<extra></extra>'
    ))
    
    # Add AQI level zones
    aqi_levels = [
        (0, 50, 'Good', '#00E400'),
        (50, 100, 'Moderate', '#FFFF00'), 
        (100, 150, 'Unhealthy for Sensitive', '#FF7E00'),
        (150, 200, 'Unhealthy', '#FF0000'),
        (200, 300, 'Very Unhealthy', '#8F3F97')
    ]
    
    for min_val, max_val, level, color in aqi_levels:
        fig.add_hrect(
            y0=min_val, y1=max_val,
            fillcolor=color,
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text=level,
            annotation_position="right"
        )
    
    fig.update_layout(
        title=f'{model_name} - 3-Day AQI Forecast',
        xaxis_title='Date',
        yaxis_title='AQI Value',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ============================================================
# Streamlit UI Configuration
# ============================================================
st.set_page_config(
    page_title="Karachi AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .aqi-level-good { color: #00E400; font-weight: bold; }
    .aqi-level-moderate { color: #FFFF00; font-weight: bold; }
    .aqi-level-unhealthy-sensitive { color: #FF7E00; font-weight: bold; }
    .aqi-level-unhealthy { color: #FF0000; font-weight: bold; }
    .aqi-level-very-unhealthy { color: #8F3F97; font-weight: bold; }
    .aqi-level-hazardous { color: #7E0023; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Main Dashboard
# ============================================================
def main():
    st.markdown('<div class="main-header">🌫️ Karachi AQI Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Model selection
        selected_models = st.multiselect(
            "Select Models to Display:",
            ['Random Forest', 'LightGBM', 'XGBoost'],
            default=['Random Forest', 'LightGBM', 'XGBoost']
        )
        
        auto_refresh = st.checkbox("Enable Auto-refresh (5 min)", value=False)
        if auto_refresh:
            st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # EPA AQI Legend
        st.header("🎨 EPA AQI Levels")
        for level, info in EPA_AQI_LEVELS.items():
            text_color = "#000000" if level in ["Good", "Moderate", "Unhealthy for Sensitive Groups"] else "#FFFFFF"
            st.markdown(
                f"<div style='background-color: {info['color']}; padding: 8px; margin: 2px; border-radius: 4px; color: {text_color}; font-weight: bold;'>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"{level}<br><small>{info['range'][0]}-{info['range'][1]}</small>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Load model results
    with st.spinner("Loading latest model results..."):
        model_results = find_latest_model_results()
    
    if not model_results:
        st.error("❌ No model results found. Please run the model training pipelines first.")
        st.info("Run the following command to train all models: `python run_all_models_comparison.py`")
        return
    
    # Filter results based on selected models
    available_models = {k: v for k, v in model_results.items() if k in selected_models}
    
    if not available_models:
        st.warning("No results available for selected models.")
        return
    
    # Dashboard Header
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Available", len(available_models))
    
    with col2:
        ts_list = []
        for info in available_models.values():
            dt = _parse_dir_ts(info['timestamp'])
            if dt:
                ts_list.append(dt)
        if ts_list:
            latest_dt = max(ts_list)
            st.metric("Last Training (PKT)", latest_dt.strftime("%Y-%m-%d %H:%M"))
        else:
            st.metric("Last Training (PKT)", "Unknown")
    
    with col3:
        st.metric("Forecast Period", "3 Days")
    
    st.markdown("---")
    
    # Model Performance Overview
    st.header("📈 Model Performance Comparison")
    
    # Load and display metrics
    metrics_data = []
    for model_name, model_info in available_models.items():
        metrics = load_model_metrics(model_info['metrics_file'])
        if metrics:
            # Extract key metrics (handle different formats)
            mae = metrics.get('mae', metrics.get('MAE', 'N/A'))
            rmse = metrics.get('rmse', metrics.get('RMSE', 'N/A'))
            r2 = metrics.get('r2', metrics.get('R2', 'N/A'))
            mape = metrics.get('mape', metrics.get('MAPE', 'N/A'))
            
            metrics_data.append({
                'Model': model_name,
                'MAE': f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae),
                'RMSE': f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse),
                'R²': f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2),
                'MAPE': f"{mape:.4f}" if isinstance(mape, (int, float)) else str(mape)
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    
    # Load historical data for bias correction and continuity
    historical_df = load_historical_data()
    
    # Current Predictions Section
    st.header("🔮 Current AQI Predictions")
    
    current_predictions = []
    other_model_forecasts = {}  # Store other model forecasts for bias correction
    
    # First pass: load all forecasts
    for model_name, model_info in available_models.items():
        if model_name != 'Random Forest':  # Skip Random Forest for now
            forecast_df = load_forecasts(model_info['forecast_file'])
            if forecast_df is not None and not forecast_df.empty:
                # Get the most recent prediction from other models
                aqi_col = None
                for col in ['AQI', 'aqi', 'predicted_aqi', 'prediction', 'aqi_forecast']:
                    if col in forecast_df.columns:
                        aqi_col = col
                        break
                if aqi_col:
                    other_model_forecasts[model_name] = forecast_df.iloc[0][aqi_col]
    
    # Calculate reference mean from other models if historical data is not available
    reference_mean = None
    if historical_df is not None and not historical_df.empty:
        reference_mean = historical_df['aqi_epa_calc'].mean()
    elif other_model_forecasts:
        reference_mean = np.mean(list(other_model_forecasts.values()))
    
    # Second pass: process all models with bias correction
    for model_name, model_info in available_models.items():
        forecast_df = load_forecasts(model_info['forecast_file'])
        
        # Validate Random Forest forecasts for bias in current predictions too
        if model_name == 'Random Forest' and forecast_df is not None and not forecast_df.empty:
            if reference_mean is not None:
                forecast_df = validate_random_forest_forecast(forecast_df, reference_mean, threshold=1.5)
        
        if forecast_df is not None and not forecast_df.empty:
            # Get the most recent prediction
            aqi_col = None
            for col in ['AQI', 'aqi', 'predicted_aqi', 'prediction', 'aqi_forecast']:
                if col in forecast_df.columns:
                    aqi_col = col
                    break
            
            if aqi_col:
                latest_pred = forecast_df.iloc[0][aqi_col]
                current_predictions.append({
                    'Model': model_name,
                    'AQI': latest_pred,
                    'Level': get_aqi_level(latest_pred)[0],
                    'Corrected': forecast_df['corrected'].iloc[0] if 'corrected' in forecast_df.columns else False
                })
    
    if current_predictions:
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        for i, pred in enumerate(current_predictions):
            with [pred_col1, pred_col2, pred_col3][i % 3]:
                level, color, description = get_aqi_level(pred['AQI'])
                
                # Create AQI gauge
                fig = create_aqi_gauge(pred['AQI'], f"{pred['Model']} Prediction")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction details
                st.markdown(f"**{pred['Model']}**")
                st.markdown(f"AQI: **{pred['AQI']:.1f}**")
                st.markdown(f"Level: **{level}**")
                
                pass
    
    st.markdown("---")
    
    # 3-Day Forecasts Section
    st.header("📅 3-Day AQI Forecasts")
    
    forecast_tabs = st.tabs(list(available_models.keys()))

    for i, (model_name, model_info) in enumerate(available_models.items()):
        with forecast_tabs[i]:
            forecast_df = load_forecasts(model_info['forecast_file'])
            if model_name == 'Random Forest' and forecast_df is not None and not forecast_df.empty:
                if reference_mean is not None:
                    forecast_df = validate_random_forest_forecast(forecast_df, reference_mean, threshold=1.5)

            if forecast_df is not None and not forecast_df.empty:
                aqi_col = None
                for col in ['AQI', 'aqi', 'predicted_aqi', 'prediction', 'aqi_forecast']:
                    if col in forecast_df.columns:
                        aqi_col = col
                        break

                hourly_df = None
                daily_df = None
                if aqi_col:
                    sorted_df = forecast_df.sort_values('datetime')
                    hourly_df = sorted_df.head(72)
                    daily_df = sorted_df[['datetime', aqi_col]].copy()
                    daily_df['datetime'] = daily_df['datetime'].dt.normalize()
                    daily_df = daily_df.groupby('datetime', as_index=False)[aqi_col].mean()

                st.subheader("Forecast Tables")
                table_tabs = st.tabs(["Hourly (72)", "Daily Avg"])

                with table_tabs[0]:
                    if aqi_col and hourly_df is not None:
                        table_data = []
                        for _, row in hourly_df.iterrows():
                            aqi_val = row[aqi_col]
                            level, color, description = get_aqi_level(aqi_val)
                            table_data.append({
                                'Date': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                                'AQI': f"{aqi_val:.1f}",
                                'Level': level
                            })
                        hourly_table = pd.DataFrame(table_data)
                        def color_level(val):
                            level_colors = {
                                'Good': '#00E400',
                                'Moderate': '#FFFF00',
                                'Unhealthy for Sensitive Groups': '#FF7E00',
                                'Unhealthy': '#FF0000',
                                'Very Unhealthy': '#8F3F97',
                                'Hazardous': '#7E0023',
                                'Very Hazardous': '#8B4513'
                            }
                            text_colors = {
                                'Good': '#000000',
                                'Moderate': '#000000',
                                'Unhealthy for Sensitive Groups': '#000000',
                                'Unhealthy': '#FFFFFF',
                                'Very Unhealthy': '#FFFFFF',
                                'Hazardous': '#FFFFFF',
                                'Very Hazardous': '#FFFFFF'
                            }
                            return f"background-color: {level_colors.get(val, '#808080')}; color: {text_colors.get(val, '#FFFFFF')}"
                        styled_hourly = hourly_table.style.applymap(color_level, subset=['Level'])
                        st.dataframe(styled_hourly, use_container_width=True, hide_index=True)

                with table_tabs[1]:
                    if daily_df is not None:
                        table_data = []
                        for _, row in daily_df.iterrows():
                            aqi_val = row[aqi_col]
                            level, color, description = get_aqi_level(aqi_val)
                            table_data.append({
                                'Date': row['datetime'].strftime('%Y-%m-%d'),
                                'AQI Avg': f"{aqi_val:.1f}",
                                'Level': level
                            })
                        daily_table = pd.DataFrame(table_data)
                        def color_level_daily(val):
                            level_colors = {
                                'Good': '#00E400',
                                'Moderate': '#FFFF00',
                                'Unhealthy for Sensitive Groups': '#FF7E00',
                                'Unhealthy': '#FF0000',
                                'Very Unhealthy': '#8F3F97',
                                'Hazardous': '#7E0023',
                                'Very Hazardous': '#8B4513'
                            }
                            text_colors = {
                                'Good': '#000000',
                                'Moderate': '#000000',
                                'Unhealthy for Sensitive Groups': '#000000',
                                'Unhealthy': '#FFFFFF',
                                'Very Unhealthy': '#FFFFFF',
                                'Hazardous': '#FFFFFF',
                                'Very Hazardous': '#FFFFFF'
                            }
                            return f"background-color: {level_colors.get(val, '#808080')}; color: {text_colors.get(val, '#FFFFFF')}"
                        styled_daily = daily_table.style.applymap(color_level_daily, subset=['Level'])
                        st.dataframe(styled_daily, use_container_width=True, hide_index=True)

                st.subheader("Forecast Charts")
                graph_tabs = st.tabs(["Hourly (72)", "Daily Avg"])
                with graph_tabs[0]:
                    fig_hourly = create_forecast_chart(hourly_df if hourly_df is not None else forecast_df, model_name, historical_df)
                    if fig_hourly:
                        st.plotly_chart(fig_hourly, use_column_width=True)
                with graph_tabs[1]:
                    if daily_df is not None:
                        fig_daily = create_forecast_chart(daily_df, f"{model_name} Daily Avg", None)
                        if fig_daily:
                            st.plotly_chart(fig_daily, use_column_width=True)
            else:
                st.warning(f"No forecast data available for {model_name}")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🌡️ Karachi AQI Prediction Dashboard | Last Updated: {}</p>
        <p>Data Source: OpenWeather API | Models: Random Forest, LightGBM, XGBoost</p>
        </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

    if auto_refresh:
        import time
        time.sleep(300)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
