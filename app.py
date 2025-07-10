import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from supabase import create_client, Client

# Configure page
st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide"
)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    url = "https://kkbezilvomgcuugjtukh.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrYmV6aWx2b21nY3V1Z2p0dWtoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwNjU3MzIsImV4cCI6MjA2NzY0MTczMn0.dNU34oBodTuHmcTiYIir14csqp-SyifFWHDQW8VgyeY"
    return create_client(url, key)

# Range Filter Functions
def range_filter_buy_signals(df, date_col='date', source_col='close', sampling_period=100, range_multiplier=3.0):
    """
    Convert Pine Script Range Filter Buy Signals to Python
    Exact replication of Pine Script logic
    """
    
    # Create a copy and ensure proper date handling
    data = df.copy()
    
    # Convert date column to datetime if it's not already
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        # Sort by date to ensure chronological order
        data = data.sort_values(by=date_col).reset_index(drop=True)
    else:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    src = data[source_col].values
    n = len(src)
    
    # Pine Script EMA implementation
    def pine_ema(values, period):
        alpha = 2.0 / (period + 1.0)
        result = np.full(len(values), np.nan)
        result[0] = values[0]
        
        for i in range(1, len(values)):
            if not np.isnan(values[i]):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
        return result
    
    # Smooth Average Range function - exact Pine Script logic
    def smooth_average_range(x, t, m):
        # Calculate absolute difference with previous value
        abs_diff = np.full(len(x), np.nan)
        abs_diff[0] = 0  # First value difference is 0
        for i in range(1, len(x)):
            abs_diff[i] = abs(x[i] - x[i-1])
        
        # Calculate EMA of absolute differences
        wper = t * 2 - 1
        avrng = pine_ema(abs_diff, t)
        smoothrng = pine_ema(avrng, wper) * m
        return smoothrng
    
    # Calculate smooth range
    smrng = smooth_average_range(src, sampling_period, range_multiplier)
    
    # Range Filter function - exact Pine Script logic
    def range_filter(x, r):
        rngfilt = np.full(len(x), np.nan)
        rngfilt[0] = x[0]
        
        for i in range(1, len(x)):
            prev_filt = rngfilt[i-1] if not np.isnan(rngfilt[i-1]) else x[i]
            current_x = x[i]
            current_r = r[i]
            
            if current_x > prev_filt:
                rngfilt[i] = max(prev_filt, current_x - current_r)
            else:
                rngfilt[i] = min(prev_filt, current_x + current_r)
        
        return rngfilt
    
    # Apply range filter
    filt = range_filter(src, smrng)
    
    # Calculate filter direction - exact Pine Script logic
    upward = np.full(n, 0.0)
    downward = np.full(n, 0.0)
    
    for i in range(1, n):
        # Upward direction
        if filt[i] > filt[i-1]:
            upward[i] = upward[i-1] + 1
        elif filt[i] < filt[i-1]:
            upward[i] = 0
        else:
            upward[i] = upward[i-1]
        
        # Downward direction
        if filt[i] < filt[i-1]:
            downward[i] = downward[i-1] + 1
        elif filt[i] > filt[i-1]:
            downward[i] = 0
        else:
            downward[i] = downward[i-1]
    
    # Calculate target bands
    hband = filt + smrng
    lband = filt - smrng
    
    # Break Out conditions - exact Pine Script logic
    longCond = np.full(n, False)
    shortCond = np.full(n, False)
    
    for i in range(1, n):
        # Long condition - exact Pine Script logic
        longCond[i] = ((src[i] > filt[i] and src[i] > src[i-1] and upward[i] > 0) or
                       (src[i] > filt[i] and src[i] < src[i-1] and upward[i] > 0))
        
        # Short condition - exact Pine Script logic  
        shortCond[i] = ((src[i] < filt[i] and src[i] < src[i-1] and downward[i] > 0) or
                        (src[i] < filt[i] and src[i] > src[i-1] and downward[i] > 0))
    
    # State tracking - exact Pine Script logic
    CondIni = np.full(n, 0)
    
    for i in range(1, n):
        if longCond[i]:
            CondIni[i] = 1
        elif shortCond[i]:
            CondIni[i] = -1
        else:
            CondIni[i] = CondIni[i-1]
    
    # Generate buy signals - exact Pine Script logic
    longCondition = np.full(n, False)
    
    for i in range(1, n):
        longCondition[i] = longCond[i] and CondIni[i-1] == -1
    
    # Add results to dataframe
    data['range_filter'] = filt
    data['smooth_range'] = smrng
    data['upper_band'] = hband
    data['lower_band'] = lband
    data['upward'] = upward
    data['downward'] = downward
    data['long_condition'] = longCond
    data['short_condition'] = shortCond
    data['condition_state'] = CondIni
    data['buy_signal'] = longCondition
    
    return data

def get_multi_stock_buy_signals(df, stock_col='stock', date_col='date', source_col='close', 
                               sampling_period=100, range_multiplier=3.0, 
                               return_buy_signals_only=True):
    """
    Apply Range Filter Buy Signals to multiple stocks and return buy signals
    """
    
    # Validate required columns
    required_cols = [stock_col, date_col, source_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Apply the range filter function to each stock group
    results = []
    
    for stock_name, stock_data in df.groupby(stock_col):
        try:
            # Apply range filter to this stock
            stock_result = range_filter_buy_signals(
                stock_data, 
                date_col=date_col, 
                source_col=source_col,
                sampling_period=sampling_period, 
                range_multiplier=range_multiplier
            )
            results.append(stock_result)
        except Exception as e:
            st.error(f"Error processing stock {stock_name}: {e}")
            continue
    
    # Combine all results
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        
        # Filter to only buy signals if requested
        if return_buy_signals_only:
            buy_signals_df = combined_df[combined_df['buy_signal'] == True].copy()
            return buy_signals_df
        else:
            return combined_df
    else:
        return pd.DataFrame()
    
def get_all_records(table_name, page_size=1000):
    all_records = []
    offset = 0
    supabase = init_supabase()
    
    while True:
        response = supabase.table(table_name).select('*').range(offset, offset + page_size - 1).execute()
        
        if not response.data:
            break
            
        all_records.extend(response.data)
        offset += page_size
        
        # Break if we got fewer records than requested (last page)
        if len(response.data) < page_size:
            break
    
    return all_records


@st.cache_data
def load_stock_data():
    """Load stock data from Supabase"""
    try:
        supabase = init_supabase()
        # Usage
        data = get_all_records('histstockdata')
        multi_stock_df = pd.DataFrame(data)
        return multi_stock_df
    except Exception as e:
        st.error(f"Error loading data from Supabase: {e}")
        return pd.DataFrame()

def get_data_stats(df):
    """Get statistics about the loaded data"""
    if df.empty:
        return None
    
    stats = {}
    
    # Total count
    stats['total_records'] = len(df)
    
    # Unique stocks
    if 'stock' in df.columns:
        stats['unique_stocks'] = df['stock'].nunique()
    else:
        stats['unique_stocks'] = 0
    
    # Latest date data available
    if 'date' in df.columns:
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        stats['latest_date'] = df_temp['date'].max()
    else:
        stats['latest_date'] = None
    
    # Latest created date (assuming there's a created_at column)
    if 'created_at' in df.columns:
        df_temp = df.copy()
        df_temp['created_at'] = pd.to_datetime(df_temp['created_at'])
        stats['latest_created'] = df_temp['created_at'].max()
    else:
        stats['latest_created'] = None
    
    return stats

def run_screener(selected_date, sampling_period=100, range_multiplier=3.0):
    """
    Run the screener for the selected date
    """
    # Load data
    multi_stock_df = load_stock_data()
    
    if multi_stock_df.empty:
        return []
    
    # Get buy signals for all stocks
    buy_signals = get_multi_stock_buy_signals(
        multi_stock_df,
        stock_col='stock',
        date_col='date',
        source_col='close',
        sampling_period=sampling_period,
        range_multiplier=range_multiplier,
        return_buy_signals_only=True
    )
    
    if buy_signals.empty:
        return []
    
    # Filter by selected date
    buy_signals['date'] = pd.to_datetime(buy_signals['date']).dt.date
    selected_date_signals = buy_signals[buy_signals['date'] == selected_date]
    
    # Format results for display
    results = []
    for _, row in selected_date_signals.iterrows():
        # Calculate signal strength based on range filter values
        upward_strength = row['upward']
        if upward_strength > 5:
            strength = "Strong"
        elif upward_strength > 2:
            strength = "Medium"
        else:
            strength = "Weak"
        
        results.append({
            "Stock": row['stock'],
            "Price": float(row['close']),
            "Signal": "BUY",
            "Strength": strength,
            "Date": row['date'],
            "Range_Filter": float(row['range_filter']),
            "Upper_Band": float(row['upper_band']),
            "Lower_Band": float(row['lower_band'])
        })
    
    return results

# Title and description
st.title("📈 Stock Screener - Range Filter Buy Signals")
st.markdown("Select a date to run the Range Filter screener and identify stocks with buy signals.")

# Load data once for stats
multi_stock_df = load_stock_data()
data_stats = get_data_stats(multi_stock_df)

# Sidebar for date selection and parameters
st.sidebar.header("Screener Configuration")

# Generate date options for last 1 week
today = datetime.now().date()
date_options = []
for i in range(8):  # Today + 7 days back
    date_options.append(today - timedelta(days=i))

# Date selection methods
date_method = st.sidebar.radio(
    "Choose date selection method:",
    ["Select from dropdown", "Enter manually"]
)

selected_date = None

if date_method == "Select from dropdown":
    selected_date = st.sidebar.selectbox(
        "Select Date:",
        date_options,
        format_func=lambda x: x.strftime("%Y-%m-%d (%A)")
    )
else:
    # Manual date input
    date_input = st.sidebar.text_input(
        "Enter Date (YYYY-MM-DD):",
        value=today.strftime("%Y-%m-%d"),
        help="Format: YYYY-MM-DD (e.g., 2024-01-15)"
    )
    
    try:
        selected_date = datetime.strptime(date_input, "%Y-%m-%d").date()
    except ValueError:
        st.sidebar.error("Invalid date format. Please use YYYY-MM-DD format.")
        selected_date = None

# Screener parameters
# st.sidebar.subheader("Range Filter Parameters")
# sampling_period = st.sidebar.slider("Sampling Period", 10, 200, 100)
# range_multiplier = st.sidebar.slider("Range Multiplier", 1.0, 5.0, 3.0, 0.1)

sampling_period=100
range_multiplier=3.0

# Display selected date
if selected_date:
    st.sidebar.success(f"Selected Date: {selected_date.strftime('%Y-%m-%d')}")

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Screener Parameters")
    
    st.write("**Current Settings:**")
    st.info(f"📅 Date: {selected_date.strftime('%Y-%m-%d') if selected_date else 'Not selected'}")
    st.info(f"📊 Sampling Period: {sampling_period}")
    st.info(f"🎯 Range Multiplier: {range_multiplier}")
    
    # Data Statistics Section
    st.subheader("📊 Data Statistics")
    
    if data_stats:
        st.write("**Dataset Overview:**")
        
        # Total records
        st.metric("Total Records", f"{data_stats['total_records']:,}")
        
        # Unique stocks
        st.metric("Unique Stocks", f"{data_stats['unique_stocks']:,}")
        
        # Latest date available
        if data_stats['latest_date']:
            latest_date_str = data_stats['latest_date'].strftime('%Y-%m-%d')
            st.metric("Latest Date", latest_date_str)
        else:
            st.metric("Latest Date", "N/A")
        
        # Latest created date
        if data_stats['latest_created']:
            latest_created_str = data_stats['latest_created'].strftime('%Y-%m-%d %H:%M')
            st.metric("Latest Created", latest_created_str)
        else:
            st.metric("Latest Created", "N/A")
    else:
        st.warning("No data loaded or available")
    
    st.write("**Range Filter Conditions:**")
    st.write("• Price above Range Filter")
    st.write("• Upward trend detected")
    st.write("• Previous state was downward")
    st.write("• Range-based signal confirmation")

with col2:
    st.subheader("Screener Results")
    
    # Submit button
    if st.button("🔍 Run Screener", type="primary", disabled=not selected_date):
        if selected_date:
            # Show loading spinner
            with st.spinner("Running Range Filter screener..."):
                try:
                    # Run the screener
                    screener_results = run_screener(
                        selected_date, 
                        sampling_period=sampling_period,
                        range_multiplier=range_multiplier
                    )
                    
                    if screener_results:
                        st.success(f"✅ Found {len(screener_results)} stocks with buy signals!")
                        
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(screener_results)
                        
                        # Display results with highlighting
                        st.markdown("### 🎯 Stocks with Buy Signals")
                        
                        # Create colored boxes for each stock
                        for idx, stock in enumerate(screener_results):
                            col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
                            
                            with col_a:
                                if stock["Strength"] == "Strong":
                                    st.success(f"**{stock['Stock']}** 🚀")
                                elif stock["Strength"] == "Medium":
                                    st.warning(f"**{stock['Stock']}** ⚡")
                                else:
                                    st.info(f"**{stock['Stock']}** 📊")
                            
                            with col_b:
                                st.write(f"₹{stock['Price']:.2f}")
                            
                            with col_c:
                                st.write(f"{stock['Signal']}")
                            
                            with col_d:
                                st.write(f"{stock['Strength']}")
                        
                        # Detailed results table
                        st.markdown("### 📋 Detailed Results")
                        display_df = df[['Stock', 'Price', 'Signal', 'Strength', 'Range_Filter', 'Upper_Band', 'Lower_Band']].copy()
                        display_df['Price'] = display_df['Price'].round(2)
                        display_df['Range_Filter'] = display_df['Range_Filter'].round(2)
                        display_df['Upper_Band'] = display_df['Upper_Band'].round(2)
                        display_df['Lower_Band'] = display_df['Lower_Band'].round(2)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("### 📊 Summary")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        
                        with col_s1:
                            st.metric("Total Signals", len(screener_results))
                        
                        with col_s2:
                            strong_signals = len([s for s in screener_results if s["Strength"] == "Strong"])
                            st.metric("Strong Signals", strong_signals)
                        
                        with col_s3:
                            avg_price = sum([s["Price"] for s in screener_results]) / len(screener_results)
                            st.metric("Avg Price", f"₹{avg_price:.2f}")
                        
                        # Downloadable results
                        st.markdown("### 📥 Download Results")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"range_filter_signals_{selected_date.strftime('%Y-%m-%d')}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.warning("No stocks found with buy signals for the selected date.")
                        
                except Exception as e:
                    st.error(f"Error running screener: {e}")
        else:
            st.error("Please select a valid date first.")

# Footer
st.markdown("---")
st.markdown("""
**How it works:**
1. Select a date from the dropdown or enter manually
2. Adjust Range Filter parameters if needed
3. Click 'Run Screener' to analyze all stocks
4. View stocks with buy signals based on Range Filter algorithm
5. Download results as CSV if needed

**Range Filter Algorithm:**
- Uses EMA-based smoothing for price filtering
- Detects trend changes and breakout conditions
- Generates buy signals when price crosses above the filter after a downtrend
""")