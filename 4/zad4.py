import warnings

import matplotlib.pyplot as plt
import nolds
import numpy as np
from matplotlib.widgets import Slider
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

x = np.linspace(0, 100, 100)


def gaussian(x):
    mean = 50
    std_dev = 10
    return 100 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2)) + np.random.normal(0,
                                                                                                                   0.25,
                                                                                                                   100)


ts = gaussian(x)

# Define initial parameters
init_window1 = 0
init_window2 = 20

normalizer = preprocessing.Normalizer(norm='l2')
ts_norm = normalizer.transform(ts.reshape(1, -1))
ts_norm = ts_norm.flatten()
scaler = preprocessing.StandardScaler()
ts_std = scaler.fit_transform(ts.reshape(-1, 1))
ts_std = ts_std.flatten()
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
ts_plot, = ax.plot(ts, color='tab:green')
norm, = ax.plot(ts_norm, color='tab:red', label='Normalization')
std, = ax.plot(ts_std, color='tab:grey', label='Standardization')
window1, = ax.plot([init_window1, init_window1], [min(ts_std) - 2, max(ts)], 'b--')
window2, = ax.plot([init_window2, init_window2], [min(ts_std) - 2, max(ts)], 'b--')
plt.legend()
props_o = dict(boxstyle='round', facecolor='green', alpha=0.25)
props_norm = dict(boxstyle='round', facecolor='red', alpha=0.25)
props_std = dict(boxstyle='round', facecolor='grey', alpha=0.25)

textstr_o = '\n'.join([
    r'Oryginal' % (np.mean(ts)),
    r'$\mu=%.2f$' % (np.mean(ts)),
    r'$\sigma=%.2f$' % (np.std(ts)),
    r'$\left\Vert\vec{V}\right\Vert=%.2f$' % (np.linalg.norm(ts)),
    r'$max=%.2f$' % (np.max(ts)),
    r'$min=%.2f$' % (np.min(ts)),
    r'$median=%.2f$' % (np.median(ts)),
    r'$Kurt=%.2f$' % (kurtosis(ts)),
    r'$Entropy=%.2f$' % (nolds.sampen(ts)),
    r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts, 1)),
    r'$Hurst=%.2f$' % (nolds.hurst_rs(ts)),
])

textstr_norm = '\n'.join([
    r'Normalization' % (np.mean(ts_norm)),
    r'$\mu=%.2f$' % (np.mean(ts_norm)),
    r'$\sigma=%.2f$' % (np.std(ts_norm)),
    r'$\left\Vert\vec{V}\right\Vert=%.2f$' % (np.linalg.norm(ts_norm)),
    r'$max=%.2f$' % (np.max(ts_norm)),
    r'$min=%.2f$' % (np.min(ts_norm)),
    r'$median=%.2f$' % (np.median(ts_norm)),
    r'$Kurt=%.2f$' % (kurtosis(ts_norm)),
    r'$Entropy=%.2f$' % (nolds.sampen(ts_norm)),
    r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts_norm, 1)),
    r'$Hurst=%.2f$' % (nolds.hurst_rs(ts_norm)),
])

textstr_std = '\n'.join([
    r'Standardization' % (np.mean(ts_std)),
    r'$\mu=%.2f$' % (np.mean(ts_std)),
    r'$\sigma=%.2f$' % (np.std(ts_std)),
    r'$\left\Vert\vec{V}\right\Vert=%.2f$' % (np.linalg.norm(ts_std)),
    r'$max=%.2f$' % (np.max(ts_std)),
    r'$min=%.2f$' % (np.min(ts_std)),
    r'$median=%.2f$' % (np.median(ts_std)),
    r'$Kurt=%.2f$' % (kurtosis(ts_std)),
    r'$Entropy=%.2f$' % (nolds.sampen(ts_std)),
    r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts_std, 1)),
    r'$Hurst=%.2f$' % (nolds.hurst_rs(ts_std)),
])
# place a text box in upper left in axes coords

box_o = plt.text(-0.69, 0.65, textstr_o, transform=ax.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props_o)

box_norm = plt.text(-0.47, 0.65, textstr_norm, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props_norm)

box_std = plt.text(-0.25, 0.65, textstr_std, transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', bbox=props_std)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.4, bottom=0.25, right=0.95)

# Make a horizontal slider to control the frequency.
axwindow = fig.add_axes([0.35, 0.1, 0.55, 0.03])
freq_slider = Slider(
    ax=axwindow,
    label='Window',
    valmin=min(x),
    valmax=max(x) - init_window2,
    valinit=init_window1,
)


# The function to be called anytime a slider's value changes
def update(val):
    start_id = int(freq_slider.val)
    start = freq_slider.val
    end_id = int(freq_slider.val + init_window2)
    end = freq_slider.val + init_window2
    ts_norm = normalizer.transform(ts[start_id:end_id].reshape(1, -1))
    ts_norm = ts_norm.flatten()
    ts_std = scaler.fit_transform(ts[start_id:end_id].reshape(-1, 1))
    ts_std = ts_std.flatten()

    window1.set_xdata([start, start])
    window2.set_xdata([end, end])
    norm.set_xdata(np.linspace(start, end, end_id - start_id))
    norm.set_ydata(ts_norm.flatten())
    std.set_xdata(np.linspace(start, end, end_id - start_id))
    std.set_ydata(ts_std.flatten())

    global box_o, box_norm, box_std
    box_o.remove()
    box_norm.remove()
    box_std.remove()

    textstr_o = '\n'.join([
        r'Oryginal' % (np.mean(ts[start_id:end_id])),
        r'$\mu=%.2f$' % (np.mean(ts[start_id:end_id])),
        r'$\sigma=%.2f$' % (np.std(ts[start_id:end_id])),
        r'$\left\Vert\vec{V}\right\Vert=%.2f$' % (np.linalg.norm(ts[start_id:end_id])),
        r'$max=%.2f$' % (np.max(ts[start_id:end_id])),
        r'$min=%.2f$' % (np.min(ts[start_id:end_id])),
        r'$median=%.2f$' % (np.median(ts[start_id:end_id])),
        r'$Kurt=%.2f$' % (kurtosis(ts[start_id:end_id])),
        r'$Entropy=%.2f$' % (nolds.sampen(ts[start_id:end_id])),
        r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts[start_id:end_id], 1)),
        r'$Hurst=%.2f$' % (nolds.hurst_rs(ts[start_id:end_id])),
    ])

    textstr_norm = '\n'.join([
        r'Normalization' % (np.mean(ts_norm)),
        r'$\mu=%.2f$' % (np.mean(ts_norm)),
        r'$\sigma=%.2f$' % (np.std(ts_norm)),
        r'$\left\Vert\vec{V}\right\Vert=%.2f$' % (np.linalg.norm(ts_norm)),
        r'$max=%.2f$' % (np.max(ts_norm)),
        r'$min=%.2f$' % (np.min(ts_norm)),
        r'$median=%.2f$' % (np.median(ts_norm)),
        r'$Kurt=%.2f$' % (kurtosis(ts_norm)),
        r'$Entropy=%.2f$' % (nolds.sampen(ts_norm)),
        r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts_norm, 1)),
        r'$Hurst=%.2f$' % (nolds.hurst_rs(ts_norm)),
    ])

    textstr_std = '\n'.join([
        r'Standardization' % (np.mean(ts_std)),
        r'$\mu=%.2f$' % (np.mean(ts_std)),
        r'$\sigma=%.2f$' % (np.std(ts_std)),
        r'$\left\Vert\vec{V}\right\Vert=%.2f$' % (np.linalg.norm(ts_std)),
        r'$max=%.2f$' % (np.max(ts_std)),
        r'$min=%.2f$' % (np.min(ts_std)),
        r'$median=%.2f$' % (np.median(ts_std)),
        r'$Kurt=%.2f$' % (kurtosis(ts_std)),
        r'$Entropy=%.2f$' % (nolds.sampen(ts_std)),
        r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts_std, 1)),
        r'$Hurst=%.2f$' % (nolds.hurst_rs(ts_std)),
    ])
    # place a text box in upper left in axes coords
    box_o = plt.text(-0.69, 0.65, textstr_o, transform=ax.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props_o)

    box_norm = plt.text(-0.47, 0.65, textstr_norm, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props_norm)

    box_std = plt.text(-0.25, 0.65, textstr_std, transform=ax.transAxes, fontsize=14,
                       verticalalignment='top', bbox=props_std)

    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)

plt.show()
