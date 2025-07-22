import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from supabase import create_client, Client
from fyers_apiv3 import fyersModel
import logging
import os
import packaging.version

# Check Streamlit version compatibility
if packaging.version.parse(st.__version__) < packaging.version.parse("1.18.0"):
    st.error("This app requires Streamlit version 1.18.0 or higher for caching functionality. Please upgrade Streamlit using: `pip install streamlit --upgrade`")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Stock Screener",
    page_icon="📈",
    layout="wide"
)

# Initialize Supabase client (no caching to avoid _thread.RLock issue)
def init_supabase():
    url = "https://kkbezilvomgcuugjtukh.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrYmV6aWx2b21nY3V1Z2p0dWtoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwNjU3MzIsImV4cCI6MjA2NzY0MTczMn0.dNU34oBodTuHmcTiYIir14csqp-SyifFWHDQW8VgyeY"
    return create_client(url, key)

# Data Fetching Functions
def get_all_records(table_name, page_size=1000):
    all_records = []
    offset = 0
    supabase = init_supabase()
    
    while True:
        try:
            response = supabase.table(table_name).select('*').range(offset, offset + page_size - 1).execute()
            if not response.data:
                break
            all_records.extend(response.data)
            offset += page_size
            if len(response.data) < page_size:
                break
        except Exception as e:
            logger.error(f"Error fetching records from {table_name}: {e}")
            st.error(f"Error fetching data from Supabase: {e}")
            break
    return all_records

@st.cache_data
def load_stock_data():
    """Load stock data from Supabase"""
    try:
        data = get_all_records('histstockdata')
        df = pd.DataFrame(data)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Sort by stock, date, and created_at (descending to get latest first)
        df = df.sort_values(['stock', 'date', 'created_at'], ascending=[True, True, False])

        # Keep only the first (latest) row for each stock and date combination
        df = df.drop_duplicates(subset=['stock', 'date'], keep='first')

        return df
    except Exception as e:
        st.error(f"Error loading data from Supabase: {e}")
        return pd.DataFrame()

def upload_to_supabase(df):
    """Append DataFrame to Supabase histstockdata table"""
    try:
        supabase = init_supabase()
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        # Add created_at timestamp
        df['created_at'] = datetime.now().isoformat()
        # Ensure JSON-serializable types
        df['date'] = df['date'].astype(str)
        df['stock'] = df['stock'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        # Prepare data for batch insert
        records = df[['stock', 'date', 'open', 'high', 'low', 'close', 'volume', 'created_at']].to_dict('records')
        # Insert in batches of 1000 with upsert to avoid duplicates
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table('histstockdata').insert(batch, upsert=True).execute()
        logger.info(f"Appended {len(records)} records to Supabase")
        return True
    except Exception as e:
        logger.error(f"Error appending to Supabase: {e}")
        st.error(f"Error appending to Supabase: {e}")
        return False

# Fyers Data Downloader
class FyersNifty500Downloader:
    def __init__(self, app_id, app_secret, access_token=None):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.fyers = None
        if self.access_token:
            self.fyers = fyersModel.FyersModel(client_id=self.app_id, token=self.access_token)

    def generate_auth_url(self):
        session = fyersModel.SessionModel(
            client_id=self.app_id,
            secret_key=self.app_secret,
            redirect_uri="https://trade.fyers.in/api-login/redirect-to-app",
            response_type="code",
            grant_type="authorization_code"
        )
        return session.generate_authcode()

    def generate_access_token(self, auth_code):
        try:
            session = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.app_secret,
                redirect_uri="https://trade.fyers.in/api-login/redirect-to-app",
                response_type="code",
                grant_type="authorization_code"
            )
            session.set_token(auth_code)
            response = session.generate_token()
            if response['s'] == 'ok':
                self.access_token = response['access_token']
                self.fyers = fyersModel.FyersModel(client_id=self.app_id, token=self.access_token)
                return True
            else:
                logger.error(f"Failed to generate access token: {response}")
                return False
        except Exception as e:
            logger.error(f"Error generating access token: {e}")
            return False

    def get_nifty500_symbols(self):
        nifty500_stocks = [
    "ACMESOLAR", "CAMPUS", "UTIAMC", "ABSLAMC", "SAMMAANCAP", "CGCL", "GLENMARK", 
    "KIRLOSENG", "PREMIERENE", "OLECTRA", "LLOYDSME", "LEMONTREE", "SYRMA", 
    "SAGILITY", "JPPOWER", "JBMA", "IIFL", "PRESTIGE", "PAYTM", "BLS", "IREDA", 
    "NUVAMA", "HOMEFIRST", "GODIGIT", "PFC", "AWL", "SWSOLAR", "BALRAMCHIN", 
    "DBREALTY", "JSWENERGY", "PCBL", "NAM-INDIA", "RADICO", "HDFCAMC", "MEDANTA", 
    "SCHNEIDER", "CONCOR", "JUBLPHARMA", "JUBLINGREA", "BBTC", "NYKAA", "RTNINDIA", 
    "EMCURE", "WAAREEENER", "MANAPPURAM", "ANANTRAJ", "KAYNES", "MARUTI", 
    "SIGNATURE", "ADANIPOWER", "RECLTD", "APTUS", "INOXWIND", "MOTILALOFS", 
    "HEG", "MSUMI", "AMBER", "BLUEDART", "RKFORGE", "ELGIEQUIP", "VTL", "PVRINOX", 
    "GICRE", "INDUSINDBK", "ROUTE", "SCHAEFFLER", "TATASTEEL", "ENGINERSIN", 
    "APARINDS", "MAHSEAMLES", "EIHOTEL", "LODHA", "SWIGGY", "BAJFINANCE", 
    "TEJASNET", "APLAPOLLO", "TATATECH", "NMDC", "ACE", "NATIONALUM", "SBFC", 
    "TITAGARH", "GRAPHITE", "JIOFIN", "IDEA", "BPCL", "RHIM", "3MINDIA", 
    "ANANDRATHI", "MFSL", "JYOTICNC", "CCL", "EIDPARRY", "TRITURBINE", "HYUNDAI", 
    "KAJARIACER", "COHANCE", "FIVESTAR", "KIMS", "LICHSGFIN", "CRISIL", 
    "AXISBANK", "SAILIFE", "NSLNISP", "SUNDARMFIN", "NETWEB", "ADANIGREEN", 
    "TORNTPOWER", "IOC", "OBEROIRLTY", "SWANENERGY", "SAIL", "NTPCGREEN", 
    "ALIVUS", "SOBHA", "PIDILITIND", "JSL", "HSCL", "WELSPUNLIV", "CROMPTON", 
    "TRIVENI", "BALKRISIND", "ERIS", "PATANJALI", "HBLENGINE", "INDUSTOWER", 
    "CHALET", "ECLERX", "VOLTAS", "JINDALSTEL", "ATGL", "ITI", "TATACHEM", 
    "SBIN", "ASAHIINDIA", "RITES", "GPIL", "RVNL", "CHOLAHLDNG", "BAJAJFINSV", 
    "ARE&M", "PRAJIND", "ALKYLAMINE", "AEGISLOG", "HONAUT", "BOSCHLTD", "GVT&D", 
    "JUBLFOOD", "VGUARD", "ULTRACEMCO", "CERA", "TATAELXSI", "RPOWER", "AFCONS", 
    "UNITDSPR", "KALYANKJIL", "BANDHANBNK", "ADANIENSOL", "JMFINANCIL", 
    "POONAWALLA", "ASTRAL", "DIXON", "MAPMYINDIA", "ADANIENT", "NH", "TATAMOTORS", 
    "EXIDEIND", "LINDEINDIA", "TRENT", "DLF", "IGIL", "MINDACORP", "JWL", 
    "USHAMART", "GAIL", "KEI", "KPRMILL", "HINDALCO", "ACC", "HUDCO", "GPPL", 
    "TTML", "CAMS", "HAVELLS", "CGPOWER", "POLYCAB", "CEATLTD", "UCOBANK", 
    "LT", "GRASIM", "JUSTDIAL", "DEEPAKNTR", "CDSL", "BHEL", "LTFOODS", 
    "TATAPOWER", "HINDCOPPER", "ONGC", "DEVYANI", "NBCC", "JSWINFRA", "AIIL", 
    "RAYMONDLSL", "POWERGRID", "BRIGADE", "AADHARHFC", "THERMAX", "ANGELONE", 
    "CUMMINSIND", "HDFCBANK", "NIVABUPA", "BATAINDIA", "ZEEL", "JSWSTEEL", 
    "GODFRYPHLP", "TATACONSUM", "GODREJPROP", "TATAINVEST", "INDIGO", "TITAN", 
    "TIINDIA", "VIJAYA", "SKFINDIA", "HEROMOTOCO", "TCS", "INDIAMART", 
    "GODREJIND", "MARICO", "TANLA", "KOTAKBANK", "KNRCON", "IRFC", "DALBHARAT", 
    "IRCTC", "J&KBANK", "SUZLON", "DMART", "SJVN", "IRB", "FINPIPE", "JKCEMENT", 
    "BRITANNIA", "ADANIPORTS", "NTPC", "IRCON", "DRREDDY", "ZENSARTECH", 
    "GODREJAGRO", "MAXHEALTH", "ETERNAL", "BASF", "NESTLEIND", "SBILIFE", 
    "CARBORUNIV", "SBICARD", "BAJAJHFL", "TECHNOE", "NHPC", "POLYMED", 
    "CAPLIPOINT", "COROMANDEL", "PETRONET", "UPL", "ABCAPITAL", "MUTHOOTFIN", 
    "KEC", "IDBI", "UNIONBANK", "RAINBOW", "LAURUSLABS", "BHARTIHEXA", "MANYAVAR", 
    "IEX", "IDFCFIRSTB", "ABBOTINDIA", "CREDITACC", "TVSMOTOR", "ASHOKLEY", 
    "SHYAMMETL", "LTTS", "RELIANCE", "ICICIBANK", "WHIRLPOOL", "PEL", 
    "LATENTVIEW", "RENUKA", "APLLTD", "HAPPSTMNDS", "WELCORP", "RCF", 
    "TORNTPHARM", "DABUR", "HONASA", "IOB", "PAGEIND", "KIRLOSBROS", "BAJAJ-AUTO", 
    "MPHASIS", "RRKABEL", "CENTURYPLY", "BIKAJI", "RAILTEL", "HINDUNILVR", 
    "SAREGAMA", "BANKBARODA", "AUBANK", "PPLPHARMA", "HFCL", "YESBANK", 
    "CHAMBLFERT", "SONACOMS", "GODREJCP", "UNOMINDA", "SAPPHIRE", "BEML", 
    "SUPREMEIND", "FLUOROCHEM", "MAZDOCK", "TATACOMM", "MASTEK", "KPIL", "ITC", 
    "LTIM", "TRIDENT", "FACT", "HINDZINC", "MRPL", "EICHERMOT", "APOLLOTYRE", 
    "APOLLOHOSP", "ASIANPAINT", "INOXINDIA", "RAYMOND", "BSOFT", "360ONE", 
    "JKTYRE", "IFCI", "NETWORK18", "VMM", "MMTC", "BEL", "BLUESTARCO", "CYIENT", 
    "TARIL", "KANSAINER", "CONCORDBIO", "PNBHOUSING", "REDINGTON", "HINDPETRO", 
    "CESC", "CRAFTSMAN", "POLICYBZR", "FINCABLES", "INDIACEM", "LUPIN", 
    "CENTRALBK", "COALINDIA", "GMRAIRPORT", "MAHABANK", "DEEPAKFERT", "JINDALSAW", 
    "IGL", "KFINTECH", "CLEAN", "NLCINDIA", "HCLTECH", "SUNDRMFAST", "UBL", 
    "IKS", "NEULANDLAB", "SRF", "FORTIS", "VEDL", "BAJAJHLDNG", "SUNPHARMA", 
    "AUROPHARMA", "BIOCON", "GNFC", "NCC", "DELHIVERY", "COLPAL", "AAVAS", 
    "JBCHEPHARM", "MOTHERSON", "ZYDUSLIFE", "ABB", "IPCALAB", "INDHOTEL", 
    "SHRIRAMFIN", "OLAELEC", "MANKIND", "SYNGENE", "NAVA", "KPITTECH", "FIRSTCRY", 
    "PNCINFRA", "AFFLE", "CANBK", "M&M", "MGL", "COCHINSHIP", "PNB", "CHOLAFIN", 
    "SUNTV", "BERGEPAINT", "SARDAEN", "FEDERALBNK", "NIACL", "TIMKEN", 
    "PHOENIXLTD", "PERSISTENT", "ABREL", "GMDCLTD", "MCX", "OIL", "KARURVYSYA", 
    "AJANTPHARM", "NAUKRI", "SCI", "LTF", "SUMICHEM", "ESCORTS", "INFY", 
    "ZFCVINDIA", "HAL", "SHREECEM", "STARHEALTH", "VBL", "DOMS", "INDIANB", 
    "AMBUJACEM", "GUJGASLTD", "ABFRL", "SIEMENS", "NATCOPHARM", "CASTROLIND", 
    "AARTIIND", "BSE", "ICICIGI", "TECHM", "GRAVITA", "PGEL", "CHENNPETRO", 
    "M&MFIN", "AIAENG", "GRANULES", "CIPLA", "BANKINDIA", "GSPL", "WIPRO", 
    "ICICIPRULI", "POWERINDIA", "ATUL", "GLAXO", "ALKEM", "WOCKPHARMA", 
    "JYOTHYLAB", "GILLETTE", "ENDURANCE", "TBOTEK", "ASTERDM", "DCMSHRIRAM", 
    "CUB", "SONATSOFTW", "RAMCOCEM", "GESHIP", "INDGN", "GLAND", "DIVISLAB", 
    "LALPATHLAB", "OFSS", "BHARATFORG", "NEWGEN", "NAVINFLUOR", "FSL", "MRF", 
    "EMAMILTD", "LICI", "SOLARINDS", "PTCIL", "ALOKINDS", "AKUMS", "ASTRAZEN", 
    "BHARTIARTL", "WESTLIFE", "COFORGE", "INTELLECT", "HDFCLIFE", "DATAPATTNS", 
    "JSWHL", "CANFINHOME", "ZENTEC", "PIIND", "ELECON", "PFIZER", "RBLBANK", 
    "GRSE", "BAYERCROP", "METROPOLIS", "BDL"
]
        return [f"NSE:{symbol}-EQ" for symbol in nifty500_stocks]

    def get_historical_data(self, symbol, days=2):
        try:
            data = {
                "symbol": symbol,
                "resolution": "D",
                "date_format": "1",
                "range_from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "range_to": datetime.now().strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            response = self.fyers.history(data=data)
            if response.get('s') == 'ok' and 'candles' in response:
                df = pd.DataFrame(response['candles'], 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['date'] = df['datetime'].dt.date
                clean_symbol = symbol.replace("NSE:", "").replace("-EQ", "")
                df['stock'] = clean_symbol
                return df[['stock', 'date', 'open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()

    def download_all_data(self, days=2):
        symbols = self.get_nifty500_symbols()
        consolidated_data = pd.DataFrame()
        for i, symbol in enumerate(symbols):
            df = self.get_historical_data(symbol, days)
            if not df.empty:
                consolidated_data = pd.concat([consolidated_data, df], ignore_index=True)
            time.sleep(1.2)
        return consolidated_data

# Signal Processing Functions
def range_filter_signals(df, date_col='date', source_col='close', sampling_period=100, range_multiplier=3.0):
    """Calculate Range Filter signals (Buy and Sell) using a window of recent data"""
    data = df.copy()
    if date_col not in data.columns:
        raise ValueError(f"Date column '{date_col}' not found")
    
    # Ensure date and created_at are datetime
    data[date_col] = pd.to_datetime(data[date_col])
    if 'created_at' in data.columns:
        data['created_at'] = pd.to_datetime(data['created_at'])
    else:
        st.warning("No 'created_at' column found; using date for sorting")
        data['created_at'] = data[date_col]

    # Filter to last 5 days of data per stock to ensure enough data points for trends
    recent_date = data[date_col].max()
    # window_start = recent_date - timedelta(days=5)
    # data = data[data[date_col] >= window_start]

    # Get latest record per stock based on created_at, keeping all data for trends
    data = data.sort_values('created_at')
    grouped = data.groupby('stock')
    results = []

    for stock_name, stock_data in grouped:
        if len(stock_data) < 2:
            st.warning(f"Insufficient data for {stock_name} (only {len(stock_data)} records); skipping")
            continue

        stock_data = stock_data.sort_values(date_col).reset_index(drop=True)
        src = stock_data[source_col].values
        n = len(src)

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

        def smooth_average_range(x, t, m):
            abs_diff = np.full(len(x), np.nan)
            abs_diff[0] = 0
            for i in range(1, len(x)):
                abs_diff[i] = abs(x[i] - x[i-1])
            wper = t * 2 - 1
            avrng = pine_ema(abs_diff, t)
            return pine_ema(avrng, wper) * m

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

        smrng = smooth_average_range(src, sampling_period, range_multiplier)
        filt = range_filter(src, smrng)
        upward = np.full(n, 0.0)
        downward = np.full(n, 0.0)

        for i in range(1, n):
            if filt[i] > filt[i-1]:
                upward[i] = upward[i-1] + 1
                downward[i] = 0
            elif filt[i] < filt[i-1]:
                downward[i] = downward[i-1] + 1
                upward[i] = 0
            else:
                upward[i] = upward[i-1]
                downward[i] = downward[i-1]

        hband = filt + smrng
        lband = filt - smrng

        longCond = np.full(n, False)
        shortCond = np.full(n, False)
        for i in range(1, n):
            longCond[i] = ((src[i] > filt[i] and src[i] > src[i-1] and upward[i] > 0) or
                           (src[i] > filt[i] and src[i] < src[i-1] and upward[i] > 0))
            shortCond[i] = ((src[i] < filt[i] and src[i] < src[i-1] and downward[i] > 0) or
                            (src[i] < filt[i] and src[i] > src[i-1] and downward[i] > 0))

        CondIni = np.full(n, 0)
        for i in range(1, n):
            if longCond[i]:
                CondIni[i] = 1
            elif shortCond[i]:
                CondIni[i] = -1
            else:
                CondIni[i] = CondIni[i-1]

        longCondition = np.full(n, False)
        shortCondition = np.full(n, False)
        for i in range(1, n):
            longCondition[i] = longCond[i] and CondIni[i-1] == -1
            shortCondition[i] = shortCond[i] and CondIni[i-1] == 1

        stock_data['range_filter'] = filt
        stock_data['smooth_range'] = smrng
        stock_data['upper_band'] = hband
        stock_data['lower_band'] = lband
        stock_data['upward'] = upward
        stock_data['downward'] = downward
        stock_data['long_condition'] = longCond
        stock_data['short_condition'] = shortCond
        stock_data['condition_state'] = CondIni
        stock_data['buy_signal'] = longCondition
        stock_data['sell_signal'] = shortCondition

        # Keep only the latest record for output
        latest_record = stock_data.iloc[-1:].copy()
        results.append(latest_record)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def get_multi_stock_signals(df, stock_col='stock', date_col='date', source_col='close', 
                           sampling_period=100, range_multiplier=3.0, signal_type='Both'):
    """Apply Range Filter signals to multiple stocks"""
    required_cols = [stock_col, date_col, source_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    try:
        signals = range_filter_signals(df, date_col, source_col, sampling_period, range_multiplier)
        if signals.empty:
            return pd.DataFrame()

        if signal_type == 'Buy':
            return signals[signals['buy_signal'] == True].copy()
        elif signal_type == 'Sell':
            return signals[signals['sell_signal'] == True].copy()
        else:
            return signals[signals['buy_signal'] | signals['sell_signal']].copy()
    except Exception as e:
        st.error(f"Error processing signals: {e}")
        return pd.DataFrame()

def get_data_stats(df):
    """Get statistics about the loaded data"""
    if df.empty:
        return None
    stats = {
        'total_records': len(df),
        'unique_stocks': df['stock'].nunique() if 'stock' in df.columns else 0,
        'latest_date': df['date'].max() if 'date' in df.columns else None,
        'latest_created': df['created_at'].max() if 'created_at' in df.columns else None
    }
    return stats

def run_screener(selected_date, sampling_period=100, range_multiplier=3.0, signal_type='Both'):
    """Run the screener for the selected date and signal type"""
    multi_stock_df = load_stock_data()
    if multi_stock_df.empty:
        return []

    signals = get_multi_stock_signals(
        multi_stock_df, sampling_period=sampling_period, range_multiplier=range_multiplier, signal_type=signal_type
    )
    if signals.empty:
        return []

    signals['date'] = pd.to_datetime(signals['date']).dt.date
    selected_date_signals = signals[signals['date'] == selected_date]

    results = []
    for _, row in selected_date_signals.iterrows():
        strength_val = row['upward'] if row['buy_signal'] else row['downward']
        strength = "Strong" if strength_val > 5 else "Medium" if strength_val > 2 else "Weak"
        signal = "BUY" if row['buy_signal'] else "SELL"
        results.append({
            "Stock": row['stock'],
            "Price": float(row['close']),
            "Signal": signal,
            "Strength": strength,
            "Date": row['date'],
            "Range_Filter": float(row['range_filter']),
            "Upper_Band": float(row['upper_band']),
            "Lower_Band": float(row['lower_band'])
        })
    return results

# UI Components
def render_data_refresh():
    """Render UI for data refresh using Fyers API"""
    st.sidebar.header("Refresh Data")
    app_id = st.sidebar.text_input("Fyers App ID", value="5GIWTBB67D-100")
    app_secret = st.sidebar.text_input("Fyers App Secret", value="OHKSCRKBKQ", type="password")
    
    if 'auth_url' not in st.session_state:
        st.session_state.auth_url = None
        st.session_state.access_token = None
        st.session_state.download_status = None

    if st.sidebar.button("Generate Auth URL"):
        downloader = FyersNifty500Downloader(app_id, app_secret)
        st.session_state.auth_url = downloader.generate_auth_url()
        st.session_state.download_status = None

    if st.session_state.auth_url:
        st.sidebar.markdown(f"[Click here to authorize]({st.session_state.auth_url})")
        auth_code = st.sidebar.text_input("Enter Authorization Code")
        
        if st.sidebar.button("Download Data", disabled=not auth_code):
            with st.spinner("Generating access token..."):
                downloader = FyersNifty500Downloader(app_id, app_secret)
                if downloader.generate_access_token(auth_code):
                    st.session_state.access_token = downloader.access_token
                    st.session_state.download_status = "started"
                else:
                    st.sidebar.error("Failed to generate access token")
                    st.session_state.download_status = None

            if st.session_state.download_status == "started":
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("Downloading data for past 2 days...")
                try:
                    data = downloader.download_all_data(days=2)
                    progress_bar.progress(50)
                    if not data.empty:
                        status_text.info("Appending data to Supabase...")
                        if upload_to_supabase(data):
                            progress_bar.progress(100)
                            status_text.success("Data appended successfully!")
                            st.session_state.download_status = "completed"
                            st.cache_data.clear()  # Clear cache to reload fresh data
                        else:
                            status_text.error("Failed to append data to Supabase")
                            st.session_state.download_status = None
                    else:
                        status_text.error("No data downloaded")
                        st.session_state.download_status = None
                except Exception as e:
                    status_text.error(f"Error during data refresh: {e}")
                    st.session_state.download_status = None
                progress_bar.empty()

# Main App
def main():
    st.title("📈 Stock Screener - Range Filter Signals")
    st.markdown("Select a date and signal type to identify stocks with buy or sell signals.")

    # Sidebar
    st.sidebar.header("Screener Configuration")
    date_method = st.sidebar.radio("Choose date selection method:", ["Select from dropdown", "Enter manually"])
    today = datetime.now().date()
    date_options = [today - timedelta(days=i) for i in range(8)]

    if date_method == "Select from dropdown":
        selected_date = st.sidebar.selectbox(
            "Select Date:", date_options, format_func=lambda x: x.strftime("%Y-%m-%d (%A)")
        )
    else:
        date_input = st.sidebar.text_input("Enter Date (YYYY-MM-DD):", value=today.strftime("%Y-%m-%d"))
        try:
            selected_date = datetime.strptime(date_input, "%Y-%m-%d").date()
        except ValueError:
            st.sidebar.error("Invalid date format. Please use YYYY-MM-DD")
            selected_date = None

    signal_type = st.sidebar.selectbox("Signal Type", ["Buy", "Sell", "Both"])
    sampling_period = 100
    range_multiplier = 3.0

    if selected_date:
        st.sidebar.success(f"Selected Date: {selected_date.strftime('%Y-%m-%d')}")

    render_data_refresh()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Screener Parameters")
        st.write("**Current Settings:**")
        st.info(f"📅 Date: {selected_date.strftime('%Y-%m-%d') if selected_date else 'Not selected'}")
        st.info(f"📊 Sampling Period: {sampling_period}")
        st.info(f"🎯 Range Multiplier: {range_multiplier}")
        st.info(f"📡 Signal Type: {signal_type}")

        st.subheader("📊 Data Statistics")
        multi_stock_df = load_stock_data()
        data_stats = get_data_stats(multi_stock_df)
        if data_stats:
            st.metric("Total Records", f"{data_stats['total_records']:,}")
            st.metric("Unique Stocks", f"{data_stats['unique_stocks']:,}")
            st.metric("Latest Date", data_stats['latest_date'].strftime('%Y-%m-%d') if data_stats['latest_date'] else "N/A")
            st.metric("Latest Created", data_stats['latest_created'].strftime('%Y-%m-%d %H:%M') if data_stats['latest_created'] else "N/A")
        else:
            st.warning("No data loaded or available")

        st.write("**Range Filter Conditions:**")
        st.write("• Buy: Price above Range Filter, upward trend, previous state downward")
        st.write("• Sell: Price below Range Filter, downward trend, previous state upward")

    with col2:
        st.subheader("Screener Results")
        if st.button("Run Screener", disabled=not selected_date):
            with st.spinner("Running Range Filter screener..."):
                try:
                    screener_results = run_screener(selected_date, sampling_period, range_multiplier, signal_type)
                    if screener_results:
                        st.success(f"✅ Found {len(screener_results)} {'signals' if signal_type == 'Both' else signal_type.lower() + ' signals'}!")
                        df = pd.DataFrame(screener_results)
                        
                        st.markdown("### 🎯 Stocks with Signals")
                        for idx, stock in enumerate(screener_results):
                            col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
                            with col_a:
                                if stock["Strength"] == "Strong":
                                    st.success(f"**{stock['Stock']}** {'🚀' if stock['Signal'] == 'BUY' else '🔽'}")
                                elif stock["Strength"] == "Medium":
                                    st.warning(f"**{stock['Stock']}** {'⚡' if stock['Signal'] == 'BUY' else '📉'}")
                                else:
                                    st.info(f"**{stock['Stock']}** {'📊' if stock['Signal'] == 'BUY' else '↘️'}")
                            with col_b:
                                st.write(f"₹{stock['Price']:.2f}")
                            with col_c:
                                st.write(f"{stock['Signal']}")
                            with col_d:
                                st.write(f"{stock['Strength']}")

                        st.markdown("### 📋 Detailed Results")
                        display_df = df[['Stock', 'Price', 'Signal', 'Strength', 'Range_Filter', 'Upper_Band', 'Lower_Band']].copy()
                        display_df['Price'] = display_df['Price'].round(2)
                        display_df['Range_Filter'] = display_df['Range_Filter'].round(2)
                        display_df['Upper_Band'] = display_df['Upper_Band'].round(2)
                        display_df['Lower_Band'] = display_df['Lower_Band'].round(2)
                        st.dataframe(display_df, use_container_width=True)

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

                        st.markdown("### 📥 Download Results")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"range_filter_signals_{selected_date.strftime('%Y-%m-%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No {signal_type.lower() if signal_type != 'Both' else 'signals'} found for the selected date.")
                except Exception as e:
                    st.error(f"Error running screener: {e}")

    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Refresh data using Fyers API credentials if needed
    2. Select a date and signal type
    3. Click 'Run Screener' to analyze stocks
    4. View buy/sell signals with strength indicators
    5. Download results as CSV

    **Range Filter Algorithm:**
    - Uses EMA-based smoothing for price filtering
    - Detects trend changes and breakout conditions
    - Buy: Price crosses above filter after downtrend
    - Sell: Price crosses below filter after uptrend
    """)

if __name__ == "__main__":
    main()