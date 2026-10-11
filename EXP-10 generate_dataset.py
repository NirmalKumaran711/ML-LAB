# Generates forest_fire_dataset.csv (500 records) using attributes of the
# Algerian Forest Fires dataset (weather + Fire Weather Index components).
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 500
temp = np.clip(rng.normal(32, 3.8, n), 22, 42)
rh = np.clip(88 - 1.6*(temp-22) + rng.normal(0, 9, n), 21, 90)
ws = np.clip(rng.normal(15.5, 3.0, n), 6, 29)
rain = np.where(rng.random(n) < 0.62, 0.0, np.clip(rng.exponential(2.2, n), 0.1, 16))
ffmc = np.clip(60 + 0.9*(temp-22) - 0.18*(rh-21) - 3.2*np.sqrt(rain) + 18 + rng.normal(0, 3.5, n), 28, 96.5)
dmc = np.clip(2 + 0.9*(temp-22)*1.1 + 0.25*(100-rh) - 1.5*rain + rng.normal(0, 7, n) + 8, 0.7, 65)
dc = np.clip(10 + 3.2*dmc + rng.normal(0, 25, n), 6, 220)
isi = np.clip(0.0004*np.exp(0.095*ffmc)*(1 + 0.05*ws) + rng.normal(0, 0.5, n), 0, 19)
bui = np.clip(0.8*dmc + 0.2*dc/3 + rng.normal(0, 2, n), 1.1, 68)
fwi = np.clip(0.55*isi*(0.4 + bui/40) + rng.normal(0, 0.7, n), 0, 31)
z = (0.16*(temp-31) - 0.07*(rh-62) + 0.04*(ws-15.5) - 0.55*rain
     + 0.10*(ffmc-80) + 0.30*(fwi-6) + rng.normal(0, 1.1, n))
fire = (z > np.quantile(z, 0.42)).astype(int)
df = pd.DataFrame({"temperature": temp.round(1), "humidity": rh.round(0).astype(int),
    "wind_speed": ws.round(0).astype(int), "rain": rain.round(1), "ffmc": ffmc.round(1),
    "dmc": dmc.round(1), "dc": dc.round(1), "isi": isi.round(1), "bui": bui.round(1),
    "fwi": fwi.round(1), "fire": fire})
df.to_csv("forest_fire_dataset.csv", index=False)
print(df.shape, df.fire.mean())
