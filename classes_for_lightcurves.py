import numpy as np
from astropy.io import fits
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from matplotlib import ticker as mticker
from matplotlib.ticker import ScalarFormatter
rcParams['xtick.top'] = 'True'
rcParams['ytick.right'] = 'True'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.size'] = 7
rcParams['ytick.major.size'] = 7
rcParams['xtick.minor.size'] = 4
rcParams['ytick.minor.size'] = 4
rcParams['axes.labelsize'] = 'large'
# rcParams['xtick.labelsize'] = 'large'
rcParams['font.size'] = 12


class LightCurve:
    """
    A class to handle light curves from the NuSTAR X-ray telescope.

    Methods
    -------
    readFits(srcFile, bkgFile)
        Read in on- and off-source data to create a LightCurve object.
    create(time, rate, rerr, back, berr)
        Create a LightCurve object given the relevant data columns.
    clean(min_fracexp=1.0, msg=True)
        Clean the raw light curve by using only those times where the minimum
        fractional exposure is >= min_fracexp (deafult=1.0, i.e. 100% up-time).
    bin(dt, min_counts=0)
        Bin the raw light curve data into time bins of size dt. If min_counts is
        specified, remove bins with counts<min_counts (default=0 will remove 
        bins where dt/10<min_counts as NuSTAR has a 10 second intrinsic time
        resolution).
    fit_gp(log_y=True, hyperparameters=None, msg=True)
        Apply Gaussian process regression using a rational quadratic kernel to
        the light curve in order to interpolate over the orbital gaps. Since X-
        ray light curves are log-normally distributed in flux, the log-count
        rates are used by default. Specifying the kernel hyperparameters can be
        done by passing a dictionary with the const, scale, and alpha values.
    sample_gp(nsamples=1000, save='no', odir=None, prefix=None):
        Sample light curves are drawn from the Gaussian process results. The
        default of 1000 samples ensures sufficient statistics to compute 68%
        range in predicted fluxes that are used in subsequent computations.
    plot(plot_bkg=False, plot_gp_samples=True):
        Plots the source light curve, as well as background and Gaussian 
        process sample light curves if desired.
    """

    def readFITS(self, srcFile: str, bkgFile: str):
        """
        A function to handle reading FITS light curve files. The total (source
        + background, i.e. on-source) and background (i.e. off-source) FITS 
        files are used to create a source light curve that can be handled by 
        other class methods.

        Parameters
        ----------
        srcFile : str
            File path to the total (on-source) light curve FITS file.
        bkgFile : str
            File path to the background (off-source) light curve FITS file.
        """

        # Read in the total (on-source) light curve
        with fits.open(srcFile) as hdul:
            src_instrument = hdul[0].header['INSTRUME']
            src_time = hdul[1].data['TIME']
            src_rate = hdul[1].data['RATE']
            src_error = hdul[1].data['ERROR']
            src_fracexp = hdul[1].data['FRACEXP']

        # Read in the background (off-source) data
        with fits.open(bkgFile) as hdul:
            bkg_instrument = hdul[0].header['INSTRUME']
            bkg_time = hdul[1].data['TIME']
            bkg_rate = hdul[1].data['RATE']
            bkg_error = hdul[1].data['ERROR']
            bkg_fracexp = hdul[1].data['FRACEXP']

        if (src_instrument != bkg_instrument):
            # If the user provides data from different detectors, raise error
            print("***mismatch in input file detectors!")
        elif (src_instrument == bkg_instrument):
            # if the detectors match, proceed with object creation
            self.instrument = src_instrument
            delta_t = np.min(np.diff(src_time))
            self.time = src_time + delta_t/2
            self.terr = delta_t/2
            self.rate = src_rate-bkg_rate
            self.rerr = np.sqrt(src_error**2+bkg_error**2)
            self.back = bkg_rate
            self.berr = bkg_error
            self.fracexp = src_fracexp
            self.raw = True
            self.binned = False


    def create(self, time: list, rate: list, rerr: list, back: list, berr: list):
        """
        A function to create a LightCurve object if the user already has access
        to the time, rate, rate error (rerr), background rate, and background
        rate error (berr) data. This can take raw or binned data as input.

        Parameters
        ----------
        time : list
            Light curve observed times.
        rate : list
            Source (i.e. total-background) count rates.
        rerr : list
            Source count rate errors.
        back : list
            Background count rates.
        berr : list
            Background count rate errors.
        """
        
        self.time, self.rate, self.rerr, self.back, self.berr = time, rate, rerr, back, berr
        delta_t = np.min(np.diff(self.time))
        self.terr = delta_t/2


    def clean(self, min_fracexp: float = 1.0, msg: bool = True):
        """
        A function to clean a LightCurve object by filtering out times where
        the fractional exposure (i.e. detector up-/live-time) falls below the
        user-defined minimum (min_fracexp). Has an option to output a message
        to the terminal indicating how many points were removed.

        Parameters
        ----------
        min_fracexcp : float
            Minimum fractional exposure that will be kept after filtering. The
            default=1.0 ensures only time bins with 100% detector up-/live-
            time are kept.
        msg : bool
            Whether or not to output how many data points were removed. The
            default=True provides the message.
        """

        if (self.raw == False and self.binned == True):
            # If the data are binned, raise an error, as it is best-practice to
            # filter the raw data prior to binning
            print("***you should be cleaning the raw data!")
        elif (self.raw == True and self.binned == False):
            # If the data are unbinned, proceed
            preclean_length = len(self.time)
            self.time = self.time[self.fracexp >= min_fracexp]
            self.rate = self.rate[self.fracexp >= min_fracexp]
            self.rerr = self.rerr[self.fracexp >= min_fracexp]
            self.back = self.back[self.fracexp >= min_fracexp]
            self.berr = self.berr[self.fracexp >= min_fracexp]
            self.fracexp = self.fracexp[self.fracexp >= min_fracexp]
            postclean_length = len(self.time)
            if (msg == True):
                print(f"{preclean_length-postclean_length} data points were \
                      cleaned from the {self.instrument} data")


    def bin(self, dt: float, min_counts: int = 0):
        """
        A function to bin a LightCurve object in time.

        Parameters
        ----------
        dt : float
            Time bin size with which to bin the data. The binning starts at t=0
            and continues on to max(t).
        min_counts : int
            Only time bins that contain this minimum number of points are
            returned. The default min_counts=0 actually uses the intrinsic 
            NuSTAR time resolution of 10 seconds to compute dt/10 as the 
            minimum number of counts.
        """
        
        n_bins = int(np.ceil((np.max(self.time)+self.terr)/dt))
        bin_edges = np.linspace(0, n_bins*dt, n_bins+1)
        bin_time, bin_rate, bin_rerr, bin_back, bin_berr, bin_counts = [], [], [], [], [], []

        if (min_counts == 0):
            min_counts = int(dt/10)

        # Perform the binning procedure
        for i in range(n_bins):
            # Only compute the binned values if the minimum number of conuts is achieved
            if (len(self.time[(self.time >= bin_edges[i]) & (self.time < bin_edges[i+1])]) >= min_counts):
                bin_counts = np.append(bin_counts, len(self.time[(self.time >= bin_edges[i]) & (self.time < bin_edges[i+1])]))
                bin_time = np.append(bin_time, (bin_edges[i]+bin_edges[i+1])/2)
                bin_rate = np.append(bin_rate, np.average(self.rate[(self.time >= bin_edges[i]) & (self.time < bin_edges[i+1])]))
                bin_rerr = np.append(bin_rerr, np.sqrt(np.sum(self.rerr[(self.time >= bin_edges[i]) & (self.time < bin_edges[i+1])]**2))/bin_counts[-1])
                bin_back = np.append(bin_back, np.average(self.back[(self.time >= bin_edges[i]) & (self.time < bin_edges[i+1])]))
                bin_berr = np.append(bin_berr, np.sqrt(np.sum(self.berr[(self.time >= bin_edges[i]) & (self.time < bin_edges[i+1])]**2))/bin_counts[-1])

        self.time, self.terr, self.rate, self.rerr, self.back, self.berr, self.counts = bin_time, dt/2, bin_rate, bin_rerr, bin_back, bin_berr, bin_counts
        self.raw = False
        self.binned = True


    def fit_gp(self, log_y: bool = True, hyperparameters: dict = None):
        """
        A function to perform a Gaussian process regression on the LightCurve
        data. This inherently assumes a rational quadratic kernel function,
        which is based on the current (2025) X-ray literature that shows it
        provides the best or equivalent to the best reproduction of X-ray light
        curves. 

        Parameters
        ----------
        log_y : bool
            Whether or not to log the count rates prior to performing the 
            regression. Default=True as X-ray light curve fluxes are log-
            normally distributed.
        hyperparameters : dict
            The dictionary of kernel hyperparameters (const, scale, alpha) to 
            be used. The default=None means that the GaussianProcessRegressor 
            will freely fit the data with no restrictions. If the hyper-
            parameters are supplied by the user, then the fitting will restrict
            the GaussianProcessRegressor to use the specified values only.
        """

        # Set up some variables needed for the regression
        y_variance = (self.rerr[self.rate>0]/self.rate[self.rate>0])**2
        self.X = np.linspace(np.min(self.time), np.max(self.time), num=int((np.max(self.time)-np.min(self.time))/(self.terr*2))+1).reshape(-1,1)
        self.gp_time = self.X.flatten()
        X_train = self.time[self.rate>0].reshape(len(self.time[self.rate>0]),1)

        if (log_y == False):
            self.gp_log_y = False
            y_train = self.rate[self.rate>0]
        elif (log_y == True):
            self.gp_log_y = True
            y_train = np.log10(self.rate[self.rate>0])

        # Set up the kernel and regressor
        if (hyperparameters == None):
            kernel = 1 * RationalQuadratic(length_scale=self.terr, length_scale_bounds=(self.terr/10, (np.max(self.time)-np.min(self.time))*2), alpha=1e-1, alpha_bounds=(1e-4, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True, alpha=y_variance)
        elif (hyperparameters != None):
            kernel = hyperparameters['const'] * RationalQuadratic(length_scale=hyperparameters['scale'], length_scale_bounds=(hyperparameters['scale'], hyperparameters['scale']), alpha=hyperparameters['alpha'], alpha_bounds=(hyperparameters['alpha'], hyperparameters['alpha']))
            self.gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_restarts_optimizer=100, normalize_y=True, alpha=y_variance)
        
        # Perform the regressions
        self.gp.fit(X_train, y_train)

        # Save the hyperparameters
        fit_params = self.gp.kernel_.get_params()
        self.gp_hyperparameters = {
                'const': fit_params['k1__constant_value'],
                'scale': fit_params['k2__length_scale'],
                'alpha': fit_params['k2__alpha']
                }
        
        # Compute the predicted mean and standard deviation and store them
        gp_mean, gp_std = self.gp.predict(self.X, return_std=True)

        if (self.gp_log_y == False):
            self.gp_mean, self.gp_lower, self.gp_upper = gp_mean, gp_mean-gp_std, gp_mean+gp_std
        elif (self.gp_log_y == True):
            self.gp_mean, self.gp_lower, self.gp_upper = 10**gp_mean, 10**(gp_mean-gp_std), 10**(gp_mean+gp_std)


    def sample_gp(self, nsamples: int = 1000, save: bool = False, odir: str = None, prefix: str = None):
        """
        A function to sample from the Gaussian process regression kernel based
        on the fitted hyperparameters.

        Parameters
        ----------
        nsamples : int
            How many samples to produce. Default=1000.
        save : True
            Whether or not to save the samples. Default=False.
        odir : str
            Output directory for the files if save=True.
        prefix : str
            Output file prefix if save=True.
        """
        
        # If the log count rates were used, make sure to account for that
        if (self.gp_log_y == False):
            self.gp_samples = self.gp.sample_y(self.X, nsamples)
        elif (self.gp_log_y == True):
            self.gp_samples = 10**self.gp.sample_y(self.X, nsamples)

        # Store how many samples were produced
        self.gp_nsamples = nsamples
        
        # Save the files if desired
        if (save == True) and (odir != None) and (prefix != None):
            for i in range(nsamples):
                sample = np.vstack([
                    self.gp_time,
                    self.gp_samples[:,i],
                    self.gp_samples[:,i] * np.median(self.rerr[self.rate>0]/self.rate[self.rate>0]),
                    np.ones(len(self.gp_samples[:,i])) * 0.01,
                    np.ones(len(self.gp_samples[:,i])) * 0.005
                    ]).T
                np.savetxt(odir+"/"+prefix+"_lc{0}.dat".format(i+1), sample, fmt=['%d', '%.4f', '%.4f', '%.4f', '%.4f'])

    
    def plot(self, plot_bkg: bool = False, plot_gp_samples: bool = True):
        """
        A function to plot a LightCurve object as well as its Gaussian process 
        sample(s), if available.

        Parameters
        ----------
        plot_bkg : bool
            Whether or not to plot the background light curve (default=False).
        plot_gp_samples : bool
            If Gaussian process regression has been performed on the light
            curve, then plot the samples that have been produced (default=
            True).
        """
        
        fig = plt.figure(figsize=(12, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.errorbar(self.time, self.rate, xerr=self.terr, yerr=self.rerr, fmt='o', markersize=5, color='black', zorder=1.4, label='Data')

        if (plot_bkg == True):
            ax.bar(self.time, self.back, self.terr*2, facecolor='darkgrey', zorder=1.0)
        
        if hasattr(self, 'gp_nsamples') and (plot_gp_samples == True):
            ax.fill_between(self.gp_time, self.gp_lower, self.gp_upper, facecolor='grey', alpha=0.5, zorder=1.2, label='GP 68% Range')
            ax.plot(self.gp_time, self.gp_mean, color='grey', zorder=1.2, label='GP Mean')
            ax.plot(self.gp_time, self.gp_samples, '-o', markersize=5, lw=0.5, zorder=1.2, label='GP Samples')
            ax.legend(loc='best', ncol=1, labelspacing=0.25, handletextpad=0.25, fancybox=False, frameon=False, edgecolor='k', framealpha=1, facecolor=None)

        ax.set_xlabel('Time [seconds]')
        ax.set_ylabel('X-ray flux [counts second$^{-1}$]')
        plt.show()


class CrossSpectrum:
    """
    A class for the cross spectrum between two LightCurve objects.

    Methods
    -------
    __init__(softLightCurve, hardLightCurve)
        Given a pair soft and hard band light curves, compute the cross-
        spectral power between them. 
    bin_freq(start_idx, geometric_factor, num_points)
        Bin the cross-spectrum in frequency space using either a constant 
        geometric binning factor (geometric_factor) or a specified number of
        points per bin (num_points) binning scheme. The start_idx (default=0)
        parameter allows the user to adjust the minimum frequency to start
        binning up from.
    compute_lag(min_freq, max_freq)
        Compute the phase-lag across a range of frequencies (min_freq to 
        max_freq), and then use the bin mid-point frequency to compute a time-
        lag (more conventional for easier interpretation than the phase-lag).
    lagfreq_spectrum(start_idx, geometric_factor, num_points)
        Combines the bin_freq() and compute_lag() methods to return a binned
        lag-frequency spectrum. The convention is to compute the cross-spectrum
        and thus lag-frequency spectrum such that negative lags correspond to
        the hard band leading the soft band and positive lags thus correspond
        to the soft band leading the hard band.
    plot_lagfreq(min_freq=0, max_freq=0, plot_allfreqs=False, plot_gp_samples=False)
        Plots the lag-frequency spectrum. Optionally, it is possible to plot
        a filled frequency range of interest (min_freq and max_freq given the 
        bounds), all of the Fourier frequencies, and, if they exist, the
        Gaussian process sample lag-frequency spectra.
    """

    def __init__(self, softLightCurve: LightCurve, hardLightCurve: LightCurve):
        """
        Initialize the CrossSpectrum object by computing the cross-spectral
        power between the given pair of soft and hard band LightCurve objects.

        Parameters
        ----------
        softLightCurve : LightCurve
            Soft band LightCurve object.
        hardLightCurve : LightCurve
            Hard band LightCurve object.
        """

        dt = hardLightCurve.terr*2
        n = len(hardLightCurve.gp_time)
        self.freq = np.fft.fftfreq(n, d=dt)[1:int(n/2)]
        hard_dft = np.fft.fft(hardLightCurve.gp_samples, axis=0)[1:int(n/2),:]
        soft_dft = np.fft.fft(softLightCurve.gp_samples, axis=0)[1:int(n/2),:]
        self.cross_power = (hard_dft.conjugate()*soft_dft) * (2*dt)/(n*np.average(hardLightCurve.gp_samples, axis=0)*np.average(softLightCurve.gp_samples, axis=0))
        self.gp_nsamples = hardLightCurve.gp_nsamples


    def bin_freq(self, start_idx: int = 0, geometric_factor: float = 0, num_points: int = 0):
        """
        A function to bin the cross spectrum in frequency space using either a
        constant geometric factor or a specified number of points per bin.

        Parameters
        ----------
        start_idx : int
            The index of the start frequency for the binning scheme. This is
            useful if the user wants to, for example, exclude frequencies that
            correspond to <3 observed cycles by using start_idx=2.
        geometric_factor : float
            Geometric binning factor.
        num_points : int
            Number of points per bin.
        """
                
        # Combine frequencies from the minimum up to the specified starting
        # frequency into a single bin
        self.freq_edges = [self.freq[0], self.freq[start_idx]]

        if (geometric_factor == 0) and (num_points == 0):
            print("***you need to select a binning scheme for the lag-frequency spectrum!")
        
        elif (geometric_factor > 1) and (num_points == 0):
            # Perform the geometric binning scheme
            while np.max(self.freq_edges)*geometric_factor <= np.max(self.freq):
                self.freq_edges = np.append(self.freq_edges, np.max(self.freq_edges)*geometric_factor)
            
        elif (geometric_factor == 0) and (num_points > 1):
            # Perform the number of points per bin binning scheme
            i = start_idx
            while (i+num_points) <= (len(self.freq)-1):
                self.freq_edges = np.append(self.freq_edges, self.freq[i+num_points])
                i += num_points
        
        elif (geometric_factor != 0) and (num_points != 0):
            print("***you need to select a single binning scheme!")


    def compute_lag(self, min_freq: float = 0, max_freq: float = 0):
        """
        A function to compute the time-lag in a given frequency range. This is
        done by first computing the phase-lag, and then dividing it by the bin
        mid-frequency following the convention of Uttley et al (2011).

        Parameters
        ----------
        min_freq : float
            Minimum frequency / left frequency bin edge.            
        max_freq : float
            Maximum frequency / right frequency bin edge.
        """
        
        freq_binmid = 10**((np.log10(min_freq) + np.log10(max_freq))/2)
        freq_neg = freq_binmid - min_freq
        freq_pos = max_freq - freq_binmid
        lag = np.angle(np.average(self.cross_power[(self.freq >= min_freq) & (self.freq < max_freq),:], axis=0))/(2*np.pi*freq_binmid)
        return freq_binmid, freq_neg, freq_pos, lag


    def lagfreq_spectrum(self, start_idx: int = 0, geometric_factor: float = 0, num_points: int = 0):
        """
        A function to compute the lag-frequency spectrum, which essentially 
        just combines the bin_freq() and compute_lag() methods above into a
        single callable. 

        Parameters
        ----------
        start_idx : int
            The index of the start frequency for the binning scheme. This is
            useful if the user wants to, for example, exclude frequencies that
            correspond to <3 observed cycles by using start_idx=2.
        geometric_factor : float
            Geometric binning factor.
        num_points : int
            Number of points per bin.
        """
        
        # Do the desired frequency binning
        self.bin_freq(start_idx, geometric_factor, num_points)

        # Compute the phase-lag -> time-lag in the computed frequency bins
        self.freq_binmid, self.freq_neg, self.freq_pos = np.empty(len(self.freq_edges)-1), np.empty(len(self.freq_edges)-1), np.empty(len(self.freq_edges)-1)
        self.lag = np.empty((len(self.freq_edges)-1, self.gp_nsamples))
        for i in range(len(self.freq_edges)-1):
            self.freq_binmid[i], self.freq_neg[i], self.freq_pos[i], self.lag[i,] = self.compute_lag(self.freq_edges[i], self.freq_edges[i+1])
        
        # Compute the median lag as well as 16th and 84th percentiles to get 
        # the 68% range across all of the Gaussian process samples
        self.lag_median, self.lag_p16, self.lag_p84 = np.percentile(self.lag, 50, axis=1), np.percentile(self.lag, 16, axis=1), np.percentile(self.lag, 84, axis=1)
        self.lag_neg, self.lag_pos = self.lag_median-self.lag_p16, self.lag_p84-self.lag_median


    def plot_lagfreq(self, min_freq: float = 0, max_freq: bool = 0, plot_allfreqs: bool = False, plot_gp_samples: bool = False, yrange: float = 0):
        """
        A function to plot the lag-frequency spectrum.

        Parameters
        ----------
        min_freq : float
            If a frequency range is to be plotted, this is the left edge.
        max_freq : float
            If a frequency range is to be plotted, this is the right edge.
        plot_allfreqs : bool
            Plot all of the Fourier frequencies. 
        plot_gp_samples : bool
            If Gaussian process samples exist, plot them all.
        """

        fig = plt.figure(figsize=(8, 6), tight_layout=True)
        ax = fig.add_subplot(111)

        if (plot_allfreqs == True):
            for f in self.freq:
                ax.axvline(f, color='grey', linewidth=0.5, zorder=1.0)

        if (plot_gp_samples == True):
            for i in range(self.gp_nsamples):
                ax.step(self.freq_binmid-self.freq_neg, self.lag[:,i], where='post', color='darkgrey', zorder=1.2)
                ax.errorbar(self.freq_binmid, self.lag[:,i], xerr=[self.freq_neg, self.freq_pos], fmt='o', markersize=5, color='darkgrey', zorder=1.2)

        ax.plot(self.freq, 1/(2*self.freq), color='red', zorder=1.6)
        ax.plot(self.freq, -1/(2*self.freq), color='red', zorder=1.6)
        ax.step(self.freq_binmid-self.freq_neg, self.lag_median, where='post', color='black', zorder=1.4)
        ax.errorbar(self.freq_binmid, self.lag_median, xerr=[self.freq_neg, self.freq_pos], yerr=[self.lag_neg, self.lag_pos], fmt='o', markersize=5, color='black', zorder=1.4)
        ax.axhline(0, color='black', linestyle='dotted', zorder=1.0)

        if (min_freq > 0) and (max_freq > min_freq):
            ax.fill_betweenx(y=[-1e5, 1e5], x1=min_freq, x2=max_freq, facecolor='dodgerblue', alpha=0.25, zorder=1.0)
        
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xscale('log')
        ax.set_ylabel('Lag [seconds]')

        if (yrange == 0):
            yrange = 1/(2*(10**((np.log10(np.min(self.freq))+np.log10(np.max(self.freq)))/2)))
            if (np.max(abs(self.lag_median)) > yrange):
                yrange = np.max(abs(self.lag_median))
        
        ax.set_ylim(-yrange, yrange)
        # ax.legend(loc='best', ncol=1, labelspacing=0.25, handletextpad=0.25, fancybox=False, frameon=False, edgecolor='k', framealpha=1, facecolor=None)
        plt.show()


class LagEnergySpectrum:
    """
    A class to handle the creation of a lag-energy spectrum given a reference
    band and a list of channels-of-interest.

    Methods
    -------
    loadLightCurves(refFile, ciFile, inDir, min_fracexp, dt, n_gp_samples)
        Read in the reference band and channel of interest light curves, which
        are defined in terms of channels within the refFile and ciFile input
        files, respectively, with the light curve files themselves stored in
        inDir. Cleaning (min_fracexp), binning (dt), and the number of Gaussian
        process samples (n_gp_samples) are also required.
    compute_lag(min_freq, max_freq)
        Compute the cross spectrum between the channel of interest and channel
        of interest subtracted reference band light curves, then compute the
        time-lag over the frequency range bound by min_freq and max_freq for
        each channel of interest energy band.
    plot()
        Plot the lag-energy spectrum.
    """

    def loadLightCurves(self, refFile: str, ciFile: str, dt: float, inDir: str = ".", min_fracexp: float = 1.0, min_counts: int = 0, log_y: bool = True, hyperparameters: dict = None, n_gp_samples: int = 1000):
        """
        A function .
        """

        self.gp_nsamples = n_gp_samples

        # Read in the reference band light curve
        refChans = np.genfromtxt(refFile, dtype=int)
        reflc_A = LightCurve()
        reflc_A.readFITS(f'{inDir}/{refChans[0]}_{refChans[1]}_A_sr.lc', f'{inDir}/{refChans[0]}_{refChans[1]}_A_bk.lc')
        reflc_A.clean(min_fracexp, msg=False)
        reflc_B = LightCurve()
        reflc_B.readFITS(f'{inDir}/{refChans[0]}_{refChans[1]}_B_sr.lc', f'{inDir}/{refChans[0]}_{refChans[1]}_B_bk.lc')
        reflc_B.clean(min_fracexp, msg=False)
        self.reflc = merge_LightCurves(reflc_A, reflc_B)
        self.reflc.bin(dt, min_counts)
        if (hyperparameters == None):
            self.reflc.fit_gp(log_y=log_y)
        elif (hyperparameters != None):
            self.reflc.fit_gp(log_y=log_y, hyperparameters=hyperparameters)
        self.reflc.sample_gp(self.gp_nsamples)

        # Read in the light curves for the channels of interest
        ciChans = np.genfromtxt(ciFile, dtype=int)
        self.energy_binmid, self.energy_neg, self.energy_pos = np.empty(len(ciChans[:,0])), np.empty(len(ciChans[:,0])), np.empty(len(ciChans[:,0]))
        self.lag_median, self.lag_p16, self.lag_p84 = np.empty(len(ciChans[:,0])), np.empty(len(ciChans[:,0])), np.empty(len(ciChans[:,0]))
        self.lag_neg, self.lag_pos = np.empty(len(ciChans[:,0])), np.empty(len(ciChans[:,0]))
        self.cilcs = {}
        for i in range(len(ciChans[:,0])):
            # Set up the energy grid now while we're reading in the light curves
            low_energy, high_energy = convert_channel_to_energy(ciChans[i,0]), convert_channel_to_energy(ciChans[i,1])
            self.energy_binmid[i] = 10**((np.log10(low_energy) + np.log10(high_energy))/2)
            self.energy_neg[i] = self.energy_binmid[i] - low_energy
            self.energy_pos[i] = high_energy - self.energy_binmid[i]

            cilc_A = LightCurve()
            cilc_A.readFITS(f'{inDir}/{ciChans[i,0]}_{ciChans[i,1]}_A_sr.lc', f'{inDir}/{ciChans[i,0]}_{ciChans[i,1]}_A_bk.lc')
            cilc_A.clean(min_fracexp, msg=False)
            cilc_B = LightCurve()
            cilc_B.readFITS(f'{inDir}/{ciChans[i,0]}_{ciChans[i,1]}_B_sr.lc', f'{inDir}/{ciChans[i,0]}_{ciChans[i,1]}_B_bk.lc')
            cilc_B.clean(min_fracexp, msg=False)
            self.cilcs[f"ci{i}"] = merge_LightCurves(cilc_A, cilc_B)
            self.cilcs[f"ci{i}"].bin(dt, min_counts)
            self.cilcs[f"ci{i}"].fit_gp(log_y=log_y, hyperparameters=self.reflc.gp_hyperparameters)
            self.cilcs[f"ci{i}"].sample_gp(self.gp_nsamples)


    def compute_lag(self, min_freq: float = 0, max_freq: float = 0):
        """
        A function .
        """
    
        if (min_freq > 0) and (max_freq > min_freq):
            # hello
            for i in range(len(self.energy_binmid)):
                subreflc = LightCurve()
                subreflc.create(self.reflc.time, self.reflc.rate-self.cilcs[f"ci{i}"].rate, self.reflc.rerr, self.reflc.back, self.reflc.berr)
                subreflc.fit_gp(hyperparameters=self.reflc.gp_hyperparameters)
                subreflc.sample_gp(self.gp_nsamples)

                cs = CrossSpectrum(subreflc, self.cilcs[f"ci{i}"])
                _, _, _, lag = cs.compute_lag(min_freq, max_freq)

                self.lag_median[i], self.lag_p16[i], self.lag_p84[i] = np.percentile(lag, 50), np.percentile(lag, 16), np.percentile(lag, 84)
                self.lag_neg[i], self.lag_pos[i] = self.lag_median[i]-self.lag_p16[i], self.lag_p84[i]-self.lag_median[i]
        
        else:
            print("***you have entered an invalid frequency range!")


    def plot(self):
        """
        A function .
        """
        
        fig = plt.figure(figsize=(8, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.step(self.energy_binmid-self.energy_neg, self.lag_median, where='post', color='black')
        ax.errorbar(self.energy_binmid, self.lag_median, xerr=[self.energy_neg, self.energy_pos], yerr=[self.lag_neg, self.lag_pos], fmt='o', markersize=5, color='black', zorder=1.4, label='Data')
        ax.set_xlabel('Observed energy [kiloelectrovolts]')
        ax.set_xscale('log')
        ax.set_xticks([4, 10, 40])
        ax.set_xticklabels([4, 10, 40])
        ax.set_ylabel('Lag [seconds]')
        plt.show()


def merge_LightCurves(LightCurve1: LightCurve, LightCurve2: LightCurve):
    """
    A function to merge two LightCurve objects into a single one. This is
    useful for NuSTAR observations as the data sets from FPMA and FPMA can be
    merged into a single data set. The merging is a simple sum of the two data
    sets (i.e. merged = FPMA + FPMB), with Gaussian error propagation. For
    statistical validity, it is better to merge two raw data sets, not the
    binned (in time) versions of those data sets.

    Parameters
    ----------
    LightCurve1 : LightCurve
        The first LightCurve object.
    LightCurve2 : LightCurve
        The second LightCurve object.
    
    Returns
    -------
    merged_LightCurve : LightCurve
        A merged LightCurve object.
    """
            
    if (LightCurve1.raw == True) and (LightCurve1.binned == False) and (LightCurve2.raw == True) and (LightCurve2.binned == False):
        merged_time = np.unique(np.append(LightCurve1.time, LightCurve2.time))
        merged_rate, merged_rerr, merged_back, merged_berr = np.zeros(len(merged_time)), np.zeros(len(merged_time)), np.zeros(len(merged_time)), np.zeros(len(merged_time))

        for i, time in enumerate(merged_time):
            if any(LightCurve1.time == time) == True and any(LightCurve2.time == time) == True:
                merged_rate[i] = LightCurve1.rate[LightCurve1.time == time] + LightCurve2.rate[LightCurve2.time == time]
                merged_rerr[i] = np.sqrt(LightCurve1.rerr[LightCurve1.time == time]**2 + LightCurve2.rerr[LightCurve2.time == time]**2)/2
                merged_back[i] = LightCurve1.back[LightCurve1.time == time] + LightCurve2.back[LightCurve2.time == time]
                merged_berr[i] = np.sqrt(LightCurve1.berr[LightCurve1.time == time]**2 + LightCurve2.berr[LightCurve2.time == time]**2)/2
            
            elif any(LightCurve1.time == time) == True and any(LightCurve2.time == time) == False:
                merged_rate[i] = LightCurve1.rate[LightCurve1.time == time]
                merged_rerr[i] = LightCurve1.rerr[LightCurve1.time == time]
                merged_back[i] = LightCurve1.back[LightCurve1.time == time]
                merged_berr[i] = LightCurve1.berr[LightCurve1.time == time]

            elif any(LightCurve1.time == time) == False and any(LightCurve2.time == time) == True:
                merged_rate[i] = LightCurve2.rate[LightCurve2.time == time]
                merged_rerr[i] = LightCurve2.rerr[LightCurve2.time == time]
                merged_back[i] = LightCurve2.back[LightCurve2.time == time]
                merged_berr[i] = LightCurve2.berr[LightCurve2.time == time]    

        merged_LightCurve = LightCurve()
        merged_LightCurve.create(merged_time, merged_rate, merged_rerr, merged_back, merged_berr)
        return merged_LightCurve
    else:
        print("***you should only be merging the raw data!")


def convert_channel_to_energy(channel: int):
    """
    A function to convert from NuSTAR FPM PI channel into photon energy.

    Parameters
    ----------
    channel : int
        NuSTAR FPM PI channel.
    
    Returns
    -------
    energy : float
        photon energy, in units of kiloelectronvolts (keV).
    """
    
    return channel*0.04+1.6