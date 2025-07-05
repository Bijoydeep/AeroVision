import requests
import pandas as pd
def load_openaq_data(location, pollutant):
    try:
        url = f"https://api.openaq.org/v2/measurements?city={location}&parameter={pollutant.lower()}&limit=100"
        r = requests.get(url).json()
        records = r["results"]
        df = pd.DataFrame.from_records(records)
        df["date_utc"] = pd.to_datetime(df["date"]["utc"])
        df = df[["date_utc", "value"]].set_index("date_utc").resample("D").mean()
        return df
    except Exception as e:
        print(e)
        return None
