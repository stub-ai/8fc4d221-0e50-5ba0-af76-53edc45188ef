import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame df with columns 'x' and 'y'
df = pd.DataFrame(np.random.randint(0,100,size=(100, 2)), columns=list('xy'))

# Perform the Granger causality test
gc_res = grangercausalitytests(df[['x', 'y']], maxlag=2, verbose=False)

# Extract the p-values and F-values into a DataFrame
results = pd.DataFrame({
    'Lag': [i for i in range(1, 3)],
    'F-value': [gc_res[i][0]['ssr_ftest'][0] for i in range(1, 3)],
    'p-value': [gc_res[i][0]['ssr_ftest'][1] for i in range(1, 3)]
})

# Plot the results
plt.figure(figsize=(12, 6))
sns.heatmap(results.set_index('Lag'), annot=True, cmap='viridis')
plt.title('Granger Causality Test Results')

# Save the plot to a PDF
pdf_pages = PdfPages('granger_causality_results.pdf')
pdf_pages.savefig(plt.gcf(), bbox_inches='tight')
pdf_pages.close()