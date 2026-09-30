import numpy as np
import pandas as pd
import os

omega = np.round(np.arange(0.0, 20.1, 0.1), 1)

# Analytical pseudo-ITD base profile (continuum)
# Characterized by peak at omega ~ 3.5-3.8 and smooth asymptotic decay
base_profile = 0.0235 * omega * np.exp(-omega / 3.8)

# Lattice spacing corrections ~ c * a^2 (a in fm)
# c_a ensures a = 0.15 fm > a = 0.12 fm > a = 0.09 fm > continuum
c_a = 0.011 * omega * np.exp(-omega / 4.2)

delta_I_cont = base_profile
delta_I_009 = base_profile + c_a * (0.09**2) / (0.15**2) * 0.25
delta_I_012 = base_profile + c_a * (0.12**2) / (0.15**2) * 0.25
delta_I_015 = base_profile + c_a * 0.25

df = pd.DataFrame(
    {
        "W": omega,
        "15": np.round(delta_I_015, 5),
        "12": np.round(delta_I_012, 5),
        "9": np.round(delta_I_019 := delta_I_009, 5),
        #"continuum": np.round(delta_I_cont, 5),
    }
)

pvt = df.melt(value_vars=['9','12','15'], id_vars=['W'], value_name='I(W)' )
pvt.columns = ['W', 'FM', 'I(W)']

VERSION = 2026
DATA_DIR = f'../Data/{VERSION}/'
fp = os.path.join(DATA_DIR, 'rpitd', 'ITD.csv')
pvt.to_csv(fp, index=False)
print(f'File saved to {fp}')

del[pvt, df]

