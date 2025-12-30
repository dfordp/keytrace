import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display


class Tonal_Fragment:
    """
    Analyzes the musical key of an audio file using the Krumhansl-Schmuckler algorithm.
    
    The algorithm compares the chromatic distribution of the audio to typical profiles
    of major and minor keys to determine the most likely key.
    
    Attributes:
        waveform: Audio data loaded by librosa
        sr: Sampling rate of the audio
        tstart: Start time in seconds (optional)
        tend: End time in seconds (optional)
        key: The detected musical key
        bestcorr: Correlation coefficient of the detected key
        altkey: Alternative key if correlation is close to the best match
        altbestcorr: Correlation coefficient of the alternative key
    """
    
    PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Krumhansl-Schmuckler key profiles
    MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    def __init__(self, waveform, sr, tstart=None, tend=None):
        """
        Initialize Tonal_Fragment and analyze the audio.
        
        Args:
            waveform: Audio waveform array from librosa
            sr: Sampling rate
            tstart: Start time in seconds (default: beginning of file)
            tend: End time in seconds (default: end of file)
        """
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend
        
        # Convert time boundaries to sample indices
        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        
        # Extract segment and compute chromagram
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(
            y=self.y_segment, 
            sr=self.sr, 
            bins_per_octave=24
        )
        
        # Calculate total chroma values for each pitch class
        self.chroma_vals = [np.sum(self.chromograph[i]) for i in range(12)]
        self.keyfreqs = {self.PITCHES[i]: self.chroma_vals[i] for i in range(12)}
        
        # Compute correlations with key profiles
        self._compute_key_correlations()
        
        # Determine primary and alternative keys
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = self.key_dict[self.key]
        
        # Find alternative key if correlation is within 90% of best
        self.altkey = None
        self.altbestcorr = None
        self._find_alternative_key()
    
    def _compute_key_correlations(self):
        """Compute correlation coefficients for all 24 major/minor keys."""
        maj_key_corrs = []
        min_key_corrs = []
        
        for i in range(12):
            # Rotate pitch frequencies to match each key
            key_test = [self.keyfreqs.get(self.PITCHES[(i + m) % 12]) for m in range(12)]
            
            # Calculate correlation with major and minor profiles
            maj_corr = round(np.corrcoef(self.MAJ_PROFILE, key_test)[1, 0], 3)
            min_corr = round(np.corrcoef(self.MIN_PROFILE, key_test)[1, 0], 3)
            
            maj_key_corrs.append(maj_corr)
            min_key_corrs.append(min_corr)
        
        # Create dictionary mapping key names to correlation coefficients
        keys = [self.PITCHES[i] + ' major' for i in range(12)] + \
               [self.PITCHES[i] + ' minor' for i in range(12)]
        
        self.key_dict = {
            **{keys[i]: maj_key_corrs[i] for i in range(12)},
            **{keys[i + 12]: min_key_corrs[i] for i in range(12)}
        }
    
    def _find_alternative_key(self):
        """Find alternative key if its correlation is within 90% of the best."""
        for key, corr in self.key_dict.items():
            if corr > self.bestcorr * 0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr
                break
    
    def get_key_with_context(self, prefer_minor=False, threshold=0.9):
        """
        Get the detected key with option to prefer minor keys for relative key pairs.
        
        This helps disambiguate between relative major/minor keys (e.g., C major vs A minor).
        
        Args:
            prefer_minor: If True, prefer minor keys when they're close to major correlations
            threshold: Correlation threshold for preferring minor (default: 0.9)
        
        Returns:
            Tuple of (key_name, correlation_coefficient)
        """
        key = self.key
        bestcorr = self.bestcorr
        
        # Check if a relative minor key is very close in correlation
        if prefer_minor and 'major' in key:
            root = key.split()[0]
            # Calculate relative minor (3 semitones down)
            idx = self.PITCHES.index(root)
            relative_minor = self.PITCHES[(idx - 3) % 12] + ' minor'
            
            if relative_minor in self.key_dict:
                minor_corr = self.key_dict[relative_minor]
                # If minor correlation exceeds threshold, prefer minor
                if minor_corr > bestcorr * threshold:
                    return relative_minor, minor_corr
        
        return key, bestcorr
    
    def print_chroma(self):
        """Print the relative prominence of each pitch class."""
        chroma_max = max(self.chroma_vals)
        for pitch, chrom in self.keyfreqs.items():
            print(f"{pitch}\t{chrom / chroma_max:5.3f}")
    
    def corr_table(self):
        """Print correlation coefficients for all major/minor keys."""
        for key, corr in sorted(self.key_dict.items()):
            print(f"{key}\t{corr:6.3f}")
    
    def print_key(self):
        """Print the detected key and alternative key (if available)."""
        key, corr = self.get_key_with_context(prefer_minor=True)
        print(f"Detected key: {key}, correlation: {corr}")
        if self.altkey is not None:
            print(f"also possible: {self.altkey}, correlation: {self.altbestcorr}")
    
    def chromagram(self, title=None):
        """
        Display a chromagram of the audio file.
        
        The chromagram shows the intensity of frequencies associated with each
        of the 12 pitch classes over time.
        
        Args:
            title: Title for the plot (default: 'Chromagram')
        """
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=self.sr, bins_per_octave=24)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(C, sr=self.sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        plt.title(title if title else 'Chromagram')
        plt.colorbar()
        plt.tight_layout()
        plt.show()