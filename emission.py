

"""
Given a correlation function ⟨𝐴(𝜏 )𝐵(0)⟩ we can define the corresponding power spectrum as

𝑆(𝜔) = ∫︁<𝐴(𝜏)𝐵(0)> exp(−𝑖𝜔𝜏)𝑑𝜏.

In QuTiP, we can calculate 𝑆(𝜔) using either qutip.correlation.spectrum_ss, which first calculates
the correlation function using the qutip.essolve.essolve solver and then performs the Fourier transform
semi-analytically, or we can use the function qutip.correlation.spectrum_correlation_fft to
numerically calculate the Fourier transform of a given correlation data using FFT.
The following example demonstrates how these two functions can be used to obtain the emission power spectrum.
"""
