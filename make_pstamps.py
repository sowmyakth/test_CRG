"""Code to create postage stamp images of parametric bulge+disk galaxies convolved 
by a chromatic PSF as seen in HST V and I bands.

Galaxy: Chromatic sum of bulge and disk as Sersics (with n=1 and 4 respectively),
that are co-centeric and have the same ellipticities.Galaxies are drawn with bulge
SED (E) and disk SED (Irr) at different 15 redshifts eqispaced from z=0 to z=1.2. At each 
redshift a galaxy is drawn with 3 different ellipticities: e=(0.3,0), e=(0,0.3) and e=(0.3/2^0.5,0.3/2^0.5).

Noise:In each band postage stamps are drawn with and without adding noise at different redshifts and ellipticities.
25 postage stamps with different noise realizations are created for each redshift and ellipticity.

PSF: For each galaxy a psf with size scaling exponent alpha=1 is drawn with the effective SED of the galaxy.

Different galaxy parametrs are writtren to file.

Note: The SEDs are normalized so as to have a specified value at a predifined wavelength. Thus different redshifted galaxies will have different total flux within the measuring bandpass.
"""
import galsim
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
import cg_fns_reloaded as cg 

def get_HST_filter(filter_name='f606w'):
    """ Returns HST filter.
    @param filter_name    Name of HST optical photometric bands (f606w,f814w).
    @return               galsim.Bandpass object. 
    """
    filter_filename = 'data/HST_{}.dat'.format(filter_name)
    filter_bandpass = galsim.Bandpass(filter_filename, wave_type='nm').thin(rel_err=1e-4)
    return filter_bandpass

def get_gal_im(Args, filter_name):
    """Returns galsim image of psf and psf convolved galaxy."""
    Args.b_SED,Args.d_SED,Args.c_SED = cg.get_SEDs(Args)
    Args.bp = get_HST_filter(filter_name)
    psf = cg.get_PSF(Args)
    gal = cg.get_gal_cg(Args)
    con = galsim.Convolve(gal,psf) 
    gal_im = con.drawImage(bandpass=Args.bp,
                       nx=Args.npix, ny=Args.npix,
                       scale=Args.scale)
    star = galsim.Gaussian(half_light_radius=1e-9)*Args.c_SED
    con = galsim.Convolve(psf, star)
    psf_im = con.drawImage(Args.bp, scale=Args.scale,
                           ny=Args.npix, nx=Args.npix)
    return gal_im, psf_im

def save_gal(gal, psf, name_str):
    hdu = fits.PrimaryHDU(gal.array)
    name = name_str.replace('change', 'gal')
    hdu.writeto(name, clobber=True)
    hdu = fits.PrimaryHDU(psf.array)
    name = name_str.replace('change', 'psf')
    hdu.writeto(name, clobber=True)


def main():
    noise_sr = 100
    """Creates galxy and psf images with different parametrs and saves to file."""
    filter_names = ['V', 'I']
    band_name = ['f606w', 'f814w']
    noise_mean = [-4.13E-06, -2.35E-05] #[V,I]
    noise_var = [4.1E-06, 3.95E-06]
    redshifts=np.linspace(0., 1.2, 15)
    # Galaxy ellipticities
    e_s =[[0.3,0.0],[0.0, 0.3] ,[0.3/2.**0.5,0.3/2.**0.5]]
    #File with correlation function of noise
    noise_file = 'data/acs_filter_unrot_sci_cf.fits' 
    # file name of table image parameters
    for f, filt in enumerate(filter_names):
        # Column names of table entries.
        # Names in capital are written into galsim.RealGalaxy catalog file.
        names = ('NUMBER', 'redshift', 'FLUX', 'NOISE_VARIANCE', 'NOISE_MEAN')
        dtype = ('int', 'float', 'float', 'float', 'float')
        # Number of noise realization
        noise_num = 25
        # total number of stamps. For a redshift and ellipticty a galaxy is 
        # drawn noise_num times with noise and once with no noise
        num = (noise_num+1)*len(e_s)*len(redshifts)
        cols = [range(num), np.zeros(num), np.zeros(num),
                np.ones(num)*noise_var[f], np.ones(num)*noise_mean[f]]
        index_table = Table(cols, names=names, dtype = dtype)
        col = Column(np.zeros([num,2]), name='e', shape=(2,), dtype='f8')
        index_table.add_column(col)
        col = Column(np.zeros(num), name='has_noise', dtype='int')
        index_table.add_column(col)
        count = 0
        for j in range(len(e_s)):
            print "gal e:", e_s[j] 
            for num, z in enumerate(redshifts):
                print "Creating gal number {0} at redshift {1} in {2} band".format(num, z, filt)
                #galaxy without noise
                input_p  =  cg.Eu_Args(shear_est='REGAUSS', scale=0.03,
                                       bulge_e=e_s[j], disk_e=e_s[j]) 
                input_p.redshift = z
                name = 'images/HST_{0}_change_{1}.fits'.format(filter_names[f], count)
                gal_im, psf_im = get_gal_im(input_p, filter_name=band_name[f])
                save_gal(gal_im, psf_im, name)
                index_table['FLUX'][count] = gal_im.array.sum()               
                index_table['redshift'][count] = z
                index_table['e'][count] = e_s[j]
                index_table['has_noise'][count] = 0
                count+=1
                #galaxy with noise
                noise_file1 = noise_file.replace('filter', filt)
                rng = galsim.BaseDeviate(123456)
                for i in range(noise_num):
                    input_p  =  cg.Eu_Args(shear_est='REGAUSS', scale=0.03,
                                       bulge_e=e_s[j], disk_e=e_s[j]) 
                    input_p.redshift = z
                    gal_im, psf_im = get_gal_im(input_p, filter_name=band_name[f])
                    noise = galsim.getCOSMOSNoise(file_name = noise_file1,
                                                  rng=rng)
                    if noise_sr:
                        gal_im.addNoiseSNR(noise,snr=noise_sr)
                        #output file name with snr set
                        op_file = 'index_table_filter_snr%i.fits'%noise_sr
                    else:
                        gal_im.addNoise(noise)
                        op_file = 'index_table_filter.fits'
                    name = 'images/HST_{0}_change_{1}.fits'.format(filter_names[f], count)
                    save_gal(gal_im, psf_im, name)
                    index_table['FLUX'][count] = gal_im.array.sum()               
                    index_table['redshift'][count] = z
                    index_table['e'][count] = e_s[j]
                    index_table['has_noise'][count] = 1
                    #Check position of entries in table
                    if index_table['NUMBER'][count] != count:
                        raise ValueError('index numbers don\'t match')
                    count+=1
        index_table.write(op_file.replace('filter', filt), format='fits',
                                          overwrite=True)

if __name__ == "__main__":
    main()
        
