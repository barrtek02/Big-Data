import matplotlib.pyplot as plt
import nolds
import numpy as np
from matplotlib.widgets import Slider
from scipy.stats import kurtosis
from sklearn.exceptions import UndefinedMetricWarning

x = np.linspace(0, 100, 100)
test = x


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

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(14, 5))
ts_plot, = ax.plot(ts, color='tab:green')
window1, = ax.plot([init_window1, init_window1], [min(ts), max(ts)], 'b--')
window2, = ax.plot([init_window2, init_window2], [min(ts), max(ts)], 'b--')

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='grey', alpha=0.5)

textstr = '\n'.join([
    r'$\mu=%.2f$' % (np.mean(ts)),
    r'$\sigma=%.2f$' % (np.std(ts)),
    r'$max=%.2f$' % (np.max(ts)),
    r'$min=%.2f$' % (np.min(ts)),
    r'$median=%.2f$' % (np.median(ts)),
    r'$Kurt=%.2f$' % (kurtosis(ts)),
    r'$Entropy=%.2f$' % (nolds.sampen(ts)),
    r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts, 1)),
    r'$Hurst=%.2f$' % (nolds.hurst_rs(ts)),
])
# place a text box in upper left in axes coords

box = plt.text(-0.25, 0.65, textstr, transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=props)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25, right=0.95)

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
    window1.set_xdata([start, start])
    window2.set_xdata([end, end])

    global box
    box.remove()
    textstr = '\n'.join([
        r'$\mu=%.2f$' % (np.mean(ts[start_id:end_id])),
        r'$\sigma=%.2f$' % (np.std(ts[start_id:end_id])),
        r'$max=%.2f$' % (np.max(ts[start_id:end_id])),
        r'$min=%.2f$' % (np.min(ts[start_id:end_id])),
        r'$median=%.2f$' % (np.median(ts[start_id:end_id])),
        r'$Kurt=%.2f$' % (kurtosis(ts[start_id:end_id])),
        r'$Entropy=%.2f$' % (nolds.sampen(ts[start_id:end_id])),
        r'$Corr Dim=%.2f$' % (nolds.corr_dim(ts[start_id:end_id], 1)),
        r'$Hurst=%.2f$' % (nolds.hurst_rs(ts[start_id:end_id])),
    ])
    # place a text box in upper left in axes coords
    box = plt.text(-0.25, 0.65, textstr, transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)

    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
plt.show()
