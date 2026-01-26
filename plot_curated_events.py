
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

def plot_curated_events():
    """
    Plots each event in curated_events.csv with its event window.
    """
    events_df = pd.read_csv('curated/curated_events.csv', parse_dates=['t0_utc', 'event_window_start', 'event_window_end', 'val_window_start', 'val_window_end'])
    ohlcv_df = pd.read_parquet('ohlcv_5min.parquet')
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], utc=True)

    plots_dir = Path('curated/plots')
    plots_dir.mkdir(exist_ok=True)

    for _, event in events_df.iterrows():
        ticker = event['ticker']
        event_time = event['t0_utc']
        start_time = event['event_window_start']
        end_time = event['val_window_end']

        ticker_ohlcv = ohlcv_df[(ohlcv_df['ticker'] == ticker) & 
                                (ohlcv_df['timestamp'] >= start_time) & 
                                (ohlcv_df['timestamp'] <= end_time)]

        if ticker_ohlcv.empty:
            print(f"No OHLCV data for {ticker} in event window {start_time} to {end_time}")
            continue

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 7))
        
        ax.plot(ticker_ohlcv['timestamp'], ticker_ohlcv['close'], label='Close Price', color='cyan')
        ax.axvline(x=event_time, color='magenta', linestyle='--', linewidth=2, label=f'Event: {event_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        event_window_start = event['event_window_start']
        event_window_end = event['event_window_end']
        val_window_start = event['val_window_start']

        ax.axvspan(event_window_start, event_window_end, alpha=0.3, color='gray', label='Event Window')
        ax.axvline(x=event_window_start, color='green', linestyle='--', linewidth=1, label='Event Window Start')
        ax.axvline(x=event_window_end, color='red', linestyle='--', linewidth=1, label='Event Window End')
        ax.axvline(x=val_window_start, color='blue', linestyle='--', linewidth=1, label='Validation Window Start')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        ax.set_title(f'Event for {ticker} at {event_time.strftime("%Y-%m-%d %H:%M:%S")}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ticker_plots_dir = plots_dir / ticker
        ticker_plots_dir.mkdir(exist_ok=True)
        
        plot_filename = f'event_{event_time.strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = ticker_plots_dir / plot_filename
        
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    plot_curated_events()
