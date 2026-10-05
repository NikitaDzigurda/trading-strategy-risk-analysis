import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta

DEALS_FILE = 'data/deals_all_opt.csv'
DATA_DIR = 'data/candles'
TIMEFRAME = '15m'


def fetch_okx_candles(inst_id, after_ts=None):
    url = "https://www.okx.com/api/v5/market/history-candles"
    params = {
        "instId": inst_id,
        "bar": TIMEFRAME,
        "limit": 100
    }
    if after_ts:
        params["after"] = after_ts

    try:
        response = requests.get(url, params=params, timeout=20)
        data = response.json()
        if data['code'] == '0':
            return data['data']
        else:
            print(f"\n[!] Ошибка API OKX: {data['msg']}")
            return None
    except Exception as e:
        print(f"\n[!] Сетевая ошибка: {e}")
        return None


def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Читаем файл со сделками...")
    df_deals = pd.read_csv(DEALS_FILE)

    df_deals['entry_date'] = pd.to_datetime(df_deals['entry_date'])
    df_deals['exit_date'] = pd.to_datetime(df_deals['exit_date'])

    min_date = df_deals['entry_date'].min()
    max_date = df_deals['exit_date'].max()

    start_dt = min_date - timedelta(days=1)
    end_dt = max_date + timedelta(days=1)

    stop_timestamp_ms = int(start_dt.timestamp() * 1000)
    initial_after_ms = int(end_dt.timestamp() * 1000)

    instruments = df_deals['instrument'].unique()

    print(f"Найдено торговых пар: {len(instruments)}")
    print(f"Качаем строго в периоде: {start_dt.strftime('%Y-%m-%d')} ---> {end_dt.strftime('%Y-%m-%d')}")
    print("-" * 40)

    for inst_id in instruments:
        filename = f"{DATA_DIR}/{inst_id}_{TIMEFRAME}.csv"

        existing_df = None
        current_after = initial_after_ms

        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                oldest_saved_ts = int(existing_df['timestamp'].min().timestamp() * 1000)
                if oldest_saved_ts <= stop_timestamp_ms:
                    print(f"[*] {inst_id} уже ПОЛНОСТЬЮ скачан. Пропускаем...")
                    continue
                else:
                    resume_date = existing_df['timestamp'].min().strftime('%Y-%m-%d')
                    print(f"[*] {inst_id}: Найден недокачанный файл. Продолжаем с {resume_date}...")
                    current_after = oldest_saved_ts
            except Exception as e:
                print(f"Файл {filename} поврежден, качаем заново. Ошибка: {e}")
                existing_df = None
        else:
            print(f"Скачивание: {inst_id} (Новый файл)")

        new_candles = []

        while True:
            candles = fetch_okx_candles(inst_id, current_after)

            if candles is None:
                time.sleep(5)
                continue

            if len(candles) == 0:
                print(f"\n[+] История на бирже закончилась.")
                break

            new_candles.extend(candles)
            oldest_candle_ts = int(candles[-1][0])

            current_date = datetime.fromtimestamp(oldest_candle_ts / 1000).strftime('%Y-%m-%d')
            print(f"\rСкачано до: {current_date} | Новых свечей: {len(new_candles)}", end='')

            if oldest_candle_ts <= stop_timestamp_ms:
                print(f"\n[+] Достигнуто начало портфеля для {inst_id}.")
                break

            current_after = oldest_candle_ts
            time.sleep(0.15)

        if new_candles:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm']
            df_new = pd.DataFrame(new_candles, columns=columns)
            df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'vol']]

            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'vol']:
                df_new[col] = df_new[col].astype(float)

            if existing_df is not None:
                final_df = pd.concat([existing_df, df_new])
            else:
                final_df = df_new

            final_df = final_df.drop_duplicates(subset=['timestamp'])
            final_df = final_df.sort_values('timestamp').reset_index(drop=True)
            final_df.to_csv(filename, index=False)
            print(f"Сохранено: {filename} (Всего строк в файле: {len(final_df)})\n")

        time.sleep(1)



if __name__ == "__main__":
    main()