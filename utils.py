import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pypulse.utils import weighted_moments

def get_dmx_ranges(model, observations):
    """
    Return an array of the MJD range for each DMX parameter corresponding to a given set of observations_in_window
    """

    mjds = observations.get_mjds().value
    dmx_parameters = model.components['DispersionDMX']  # names of the DMX_xxxx parameters

    DMXR1_names = [a for a in dir(dmx_parameters) if a.startswith('DMXR1')]
    DMXR2_names = [a for a in dir(dmx_parameters) if a.startswith('DMXR2')]

    DMXR1_values = [getattr(dmx_parameters, par).value for par in DMXR1_names]
    DMXR2_values = [getattr(dmx_parameters, par).value for par in DMXR2_names]

    DMXR = np.column_stack((DMXR1_values, DMXR2_values))  # Zip the two arrays

    # Out of all the DMX windows, we will only keep those with at least one TOA inside it
    mask = (DMXR1_values < max(mjds)) & (DMXR2_values > min(mjds))

    return DMXR[mask]


def get_dmx_observations(observations, low_mjd, high_mjd):
    """
    Return an array for selecting TOAs from toas in a DMX range.

    toas is a PINT TOA object of TOAs in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = observations.get_mjds().value
    mask = (low_mjd < mjds) & (mjds < high_mjd)

    return observations[mask]

def epoch_scrunch(toas, data=None, errors=None, epochs=None, decimals=0, getdict=False, weighted=False, harmonic=False):
    if epochs is None:
        epochsize = 10 ** (-decimals)
        bins = np.arange(np.around(min(toas), decimals=decimals) - epochsize,
                         np.around(max(toas), decimals=decimals) + 2 * epochsize,
                         epochsize)  # 2 allows for the extra bin to get chopped by np.histogram
        freq, bins = np.histogram(toas, bins)
        validinds = np.where(freq != 0)[0]

        epochs = np.sort(bins[validinds])
        diffs = np.array(list(map(lambda x: np.around(x, decimals=decimals), np.diff(epochs))))
        epochs = np.append(epochs[np.where(diffs > epochsize)[0]], [epochs[-1]])
    else:
        epochs = np.array(epochs)
    reducedTOAs = np.array(list(map(lambda toa: epochs[np.argmin(np.abs(epochs - toa))], toas)))

    if data is None:
        return epochs

    Nepochs = len(epochs)

    if weighted and errors is not None:
        averaging_func = lambda x, y: weighted_moments(x, 1.0 / y ** 2, unbiased=True, harmonic=harmonic)
    else:
        averaging_func = lambda x, y: (np.mean(x), np.std(y))  # is this correct?

    if getdict:
        retval = dict()
        retvalerrs = dict()
    else:
        retval = np.zeros(Nepochs)
        retvalerrs = np.zeros(Nepochs)
    for i in range(Nepochs):
        epoch = epochs[i]
        inds = np.where(reducedTOAs == epoch)[0]
        if getdict:
            retval[epoch] = data[inds]
            if errors is not None:
                retvalerrs[epoch] = errors[inds]
        else:
            if errors is None:
                retval[i] = np.mean(data[inds])  # this is incomplete
                retvalerrs[i] = np.std(data[inds])  # temporary
            else:
                retval[i], retvalerrs[i] = averaging_func(data[inds], errors[inds])
    #            print data[inds],errors[inds]
    if getdict and errors is None:  # is this correct?
        return epochs, retval
    return epochs, retval, retvalerrs

def broadband_observations(toas):
    """Function that, given a set of TOAs, separates the narrowband observations_in_window"""

    # ----------------------------------------------------------------
    # First, we obtain the backend that was used in each epoch
    # ----------------------------------------------------------------

    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])

    # ----------------------------------------------------------------
    # Now we separate the GASP observations_in_window and the GUPPI observations_in_window
    # ----------------------------------------------------------------

    # Firstly, we mark with 'True' the GUPPI observations_in_window
    broadband_ok = np.isin(backends, ['GUPPI'])

    # Now we separate the broadband TOAs
    broadband_TOAs = toas[broadband_ok]
    n_broadband = np.count_nonzero(broadband_ok)
    print('Broadband TOAs = ' + str(n_broadband))

    return broadband_TOAs

def make_plot(PSR_name, df):
    windows_centers = df["DMXR1"] + (df["DMXR2"] - df["DMXR1"]) / 2.0

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=3.0)
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(12, 24), sharex=True,
                           gridspec_kw={'hspace': 0})
    fig.suptitle(PSR_name)

    # Plot and label each subplot
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(6):
        ax[i].scatter(windows_centers, df[f'C{i}'], color=colors[i])
        ax[i].set_ylabel(f"C{i}")
        ax[i].grid(True)  # Add grid
        ax[i].label_outer()  # Hide inner x labels and ticks

    ax[5].set_xlabel("Window Center [MJD]")

    plt.tight_layout()
    plt.savefig('./' + PSR_name + '_results.png')
    plt.show()

    return


def map_domain(frequencies):

    lambdas = np.power(frequencies, -1.0)  # Inverse of the frequency
    x_aux_values = (lambdas - np.amin(lambdas))/(np.amax(lambdas)-np.amin(lambdas))  # Between 0 and 1
    x_values = np.subtract(np.multiply(x_aux_values, 2.0), 1.0)                      # Between -1 and 1

    return x_values


class PulsarData:
    def __init__(self, name, toas, tm, dmx_ranges):
        self.name = name
        self.toas = toas
        self.tm = tm
        self.dmx_ranges = dmx_ranges

    # The __str__() function controls what should be returned when the class object is represented as a string.
    def __str__(self):
        return f"{self.name}"