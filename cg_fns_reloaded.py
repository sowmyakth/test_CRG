""" Functions for color gradient(CG) analysis.

Define functions to measure bias from CG in shape measurements. Create galaxy 
with CG for Euclid and LSST using GalSim. Compare it's calculated shape to 
galaxy without CG as defined in Semboloni et al. (2013). Galaxy shape can be
measured by either HSM module in GalSim or direct momemtsvcalculation without
PSF correction.

Implementation is for Euclid (as used in Semboloni 2013) and LSST parameters.

Notes:
calc_cg_new is either faster or same as calc_cg.
REGAUSS  is slower than KSB




Edits from original in ~/sum2015
, flux_type= 'flambda' added in get_SEDs

"""



import galsim
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.console import ProgressBar
from lmfit import minimize, Parameters, report_fit, report_errors,fit_report


def make_Euclid_filter(res=1.0):
    """ Make a Euclid-like filter (eye-balling Semboloni++13).

    @param res  Resolution in nanometers.
    @return     galsim.Bandpass object.
    """

    x = [550.0, 750.0, 850.0, 900.0]
    y = [0.3, 0.3, 0.275, 0.2]
    tab = galsim.LookupTable(x, y, interpolant='linear')
    w = np.arange(550.0, 900.01, res)
    return galsim.Bandpass(galsim.LookupTable(w, tab(w), interpolant='linear'))

def get_LSST_filter(filter_name='r'):
    """ Return LSST filter stored in /data file
    @param filter_name    Name of LSST optical photometric bands (u,g,r,i,z,y).
    @return               galsim.Bandpass object. 
    """

    datapath = os.getcwd()+'/data/'
    filter_filename = datapath+'LSST_{}.dat'.format(filter_name)
    filter_bandpass = galsim.Bandpass(filter_filename, wave_type='nm').thin(rel_err=1e-4)
    return filter_bandpass

def get_SEDs(Args):
    """ Return bulge, disk and composite SEDs at given redshift.

    The bulge and disk SEDs are normalized to 1.0 at 550 nm in rest-frame and
    then redshifted. Composite SED is the flux weighted sum of bulge and disk 
    SEDs.

    Note: Total flux of SEDs are not normalized.


    @param Args    Class with the following attributes:
        Args.disk_SED_name     One of ['Sbc', 'Scd', 'Im', 'E'] to indicate disk 
                            SED.(default: 'Im')
        Args.bulge_SED_name     One of ['Sbc', 'Scd', 'Im', 'E'] to indicate bulge 
                            SED.(default:'E')
        Args.redshift       Redshift of galaxy (both bulge and disk).
        Args.bulge_frac     Fraction of flux in bulge at 550 nm rest-frame.
    @returns  bulge SED, disk SED, composite SED.
    """
    datapath = os.getcwd()+'/data/'
    b_SED = galsim.SED(datapath+"CWW_{}_ext.sed".format(Args.bulge_SED_name), 
                       wave_type='Ang', flux_type= 'flambda')
    d_SED = galsim.SED(datapath+"CWW_{}_ext.sed".format(Args.disk_SED_name), 
                       wave_type='Ang', flux_type= 'flambda')
    b_SED = b_SED.withFluxDensity(1.0, 550.0).atRedshift(Args.redshift)
    d_SED = d_SED.withFluxDensity(1.0, 550.0).atRedshift(Args.redshift)
    c_SED = b_SED * Args.bulge_frac + d_SED * (1. - Args.bulge_frac)
    return b_SED, d_SED, c_SED

def get_PSF(Args):
    """ Return a chromatic PSF. Size of PSF is wavelength dependent.

    @param Args    Class with the following attributes:
        Args.psf_sigma_o   Gaussian sigma of PSF at known wavelength Args.psf_w_o.
        Args.psf_w_o       Wavelength at which PSF size is known (nm).
        Args.alpha         PSF wavelength scaling exponent.  1.0 for diffraction 
                           limit, -0.2 for Kolmogorov turbulence.
    @return chromatic PSF.
    """
    mono_PSF = galsim.Gaussian(sigma=Args.psf_sigma_o)
    chr_PSF = galsim.ChromaticObject(mono_PSF).dilate(lambda w: (w/Args.psf_w_o)**Args.alpha)
    return chr_PSF

def get_gal_cg(Args):
    """ Return surface brightness profile (SBP) of cocentric bulge + disk galaxy. 

    Bulge and disk have co-centric Sersic profiles.

    @param Args    Class with the following attributes:
        Args.bulge_n       Sersic index of bulge.
        Args.bulge_HLR     Half-light-radius of the bulge.
        Args.bulge_e       Shape of bulge [e1, e2].
        Args.bulge_frac    Fraction of flux in bulge at 550 nm rest-frame.
        Args.T_flux        Total flux in the galaxy.
        Args.disk_n        Sersic index of disk.
        Args.disk_HLR      Half-light-radius of the disk.
        Args.disk_e        Shape of disk [e1, e2].
        Args.b_SED         SED of bulge.
        Args.d_SED         SED of disk.
    @return galaxy with CG
    """
    bulge = galsim.Sersic(n=Args.bulge_n, half_light_radius=Args.bulge_HLR,
                          flux=Args.T_flux * Args.bulge_frac)
    bulge = bulge.shear(e1=Args.bulge_e[0], e2=Args.bulge_e[1])
    disk = galsim.Sersic(n=Args.disk_n, half_light_radius=Args.disk_HLR,
                         flux=Args.T_flux * (1 - Args.bulge_frac))
    disk = disk.shear(e1=Args.disk_e[0], e2=Args.disk_e[1])
    gal = bulge * Args.b_SED + disk * Args.d_SED
    return gal

def get_gal_nocg(Args, gal_cg, chr_PSF):
    """ Construct a galaxy SBP with no CG that yields the same PSF convolved 
    image as the given galaxy with CG convolved with the PSF. 

    To reduduce pixelization effects, resolution is incresed 4 times when 
    drawing images of effective PSF and PSF convolved galaxy with CG. These
    images don't represent physical objects that the telescope will see.

    @param Args    Class with the following attributes:
        Args.npix   Number of pixels across square postage stamp image.
        Args.scale  Pixel scale for postage stamp image.
        Args.bp     GalSim Bandpass describing filter.
        Args.c_SED  Flux weighted composite SED. 
    @param gal_cg   GalSim GSObject describing SBP of galaxy with CG.
    @param chr_PSF  GalSim ChromaticObject describing the chromatic PSF.
    @return     SBP of galaxy with no CG, with composite SED.
    """
    # PSF is convolved with a delta function to draw effective psf image
    star = galsim.Gaussian(half_light_radius=1e-9)*Args.c_SED
    con = galsim.Convolve(chr_PSF, star)
    psf_eff_img = con.drawImage(Args.bp, scale=Args.scale/4.0,
                                ny=Args.npix*4.0, nx=Args.npix*4.0,
                                method='no_pixel')
    psf_eff = galsim.InterpolatedImage(psf_eff_img, calculate_stepk=False,
                                       calculate_maxk=False)
    con = galsim.Convolve(gal_cg,chr_PSF) 
    gal_cg_eff_img = con.drawImage(Args.bp, scale=Args.scale/4.0, 
    	                           nx=Args.npix*4.0, ny=Args.npix*4.0,
                                   method='no_pixel')
    gal_cg_eff = galsim.InterpolatedImage(gal_cg_eff_img, 
    	                                  calculate_stepk=False, calculate_maxk=False)
    gal_nocg = galsim.Convolve(gal_cg_eff, galsim.Deconvolve(psf_eff))
    return gal_nocg*Args.c_SED

def get_moments(array):
    """ Compute second central moments of an array.
    @param array  Array of profile to calculate second moments       
    @return Qxx, Qyy, Qxy second central moments of the array.
    """
    nx, ny = array.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    denom = np.sum(array)
    xbar = np.sum(array*x)/denom
    ybar = np.sum(array*y)/denom
    Qxx = np.sum(array*(x-xbar)**2)/denom
    Qyy = np.sum(array*(y-ybar)**2)/denom
    Qxy = np.sum(array*(x-xbar)*(y-ybar))/denom
    return Qxx, Qyy, Qxy

def fcn2min(params,data,Args,psf):
    """Function given as input to lmfit, to compute residual of fit and true
    galaxy (galaxy with no CG)
    @param params  fit parameters
    @param data    true data
    @param Args
    @param mod_psf psf
    @returns difference betwwen fit and true"""
    g1 = params['g1'].value       #shear of galaxy
    g2 = params['g2'].value       #shear of galaxy
    rb = params['rb'].value       #half light radius of buldge
    rd = params['rd'].value       #half light radius disk
    bf = params['bf'].value     #ratio Flux of buldge to total flux

    mod_bulge = galsim.Sersic(n=Args.bulge_n, half_light_radius=rb,
                              flux=Args.T_flux * bf)
    mod_disk = galsim.Sersic(n=Args.disk_n, half_light_radius=rd,
                                     flux=Args.T_flux * (1 - bf))
    mod_gal = (mod_bulge + mod_disk)*Args.c_SED
    mod_gal = mod_gal.shear(g1=g1, g2=g2)
    obj_con = galsim.Convolve(mod_gal, psf)
    mod_im = (obj_con.drawImage(bandpass=Args.bp, scale=Args.scale,
                                nx=Args.npix, ny=Args.npix)).array
    model1 = mod_im.flatten()
    resid = model1 - data
    return resid

def param_in(Args):
    """To make sure every fit gets the same initial params.Else multiple runs take parameters of previous fit
    @param   params   parameters class for fit
    @returns parameters with preset values""" 
    d=(1 + 0.1*np.random.random())
    params = Parameters()
    params.add('g1', value=0.1, vary=True, min=-1., max=1.)
    params.add('g2', value=0.1, vary=True, min=-1.,max=1.)
    params.add('rb', value=Args.bulge_HLR*d, vary=True, min=0.,max=3.)
    params.add('rd', value=Args.disk_HLR*d, vary=True, min=0., max=3.)
    params.add('bf', value=Args.bulge_frac*d, vary=True, min=0, max=1.)
    return params

def estimate_shape(Args, gal_img, PSF_img, method):
    """ Estimate the shape (ellipticity) of a galaxy. 

    Shape is calculated by either one of the HSM methods or by direct 
    calculation of moments wihtout any PSF correction. Of the HSM
    methods, KSB has the option of manually setting size of weight function.


    @param Args    Class with the following attributes:
        Args.sig_w  Sigma of Gaussian weight function.
        Args.npix   Number of pixels across postage stamp image.
        Args.scale  Pixel scale of postage stamp image.
    @param gal_img  A GalSim Image of the PSF-convolved galaxy.
    @param PSF_img  A GalSim Image of the PSF.
    @param method   Method to use to estimate shape.  One of:
        'S13'  Use Semboloni++13 observed second moments method 
        'REGAUSS', 'LINEAR', 'BJ', 'KSB'  Use GalSim.hsm module
    @returns galsim.Shear object holding galaxy ellipticity.
    """
    if method == 'S13':
        weight = galsim.Gaussian(sigma=Args.sig_w)
        weight_img = weight.drawImage(nx=Args.npix, ny=Args.npix, scale=Args.scale)
        Qxx, Qyy, Qxy = get_moments(weight_img.array * gal_img.array)
        R = Qxx + Qyy
        e1 = (Qxx-Qyy)/R
        e2 = 2*Qxy/R
        shape = galsim.Shear(e1=e1, e2=e2)
    elif method in ['REGAUSS', 'LINEAR', 'BJ']:
        new_params = galsim.hsm.HSMParams(nsig_rg=200, nsig_rg2=200,
        	                              max_moment_nsig2=40000)
        result = galsim.hsm.EstimateShear(gal_img, PSF_img, shear_est=method,
        	                              hsmparams = new_params)
        shape = galsim.Shear(e1=result.corrected_e1, e2=result.corrected_e2)
    elif method == 'KSB':
        if Args.sig_w :
            #Manually set size of weight fn in HSM
            new_params = galsim.hsm.HSMParams(ksb_sig_weight=Args.sig_w/Args.scale,
                                              nsig_rg=200, nsig_rg2=200,
                                              max_moment_nsig2=40000)
            result = galsim.hsm.EstimateShear(gal_img, PSF_img, shear_est=method,
            	                              hsmparams = new_params)
        else:
            #Weight size is not given; HSM calculates the appropriate weight
            new_params = galsim.hsm.HSMParams(nsig_rg=200, nsig_rg2=200,
                                              max_moment_nsig2=40000)
            result = galsim.hsm.EstimateShear(gal_img, PSF_img, shear_est=method,
                                              hsmparams = new_params)
        shape = galsim.Shear(g1=result.corrected_g1, g2=result.corrected_g2)
    elif method == 'fit':
        params = param_in(Args)
        data = (gal_img.array).flatten()
        fit_kws = {'maxfev':1000,'ftol':1.49012e-38,'xtol':1.49012e-38,}
        chr_psf = get_PSF(Args)
        result = minimize(fcn2min, params ,args=(data, Args, chr_psf), **fit_kws)
        #print 'chisqr',result.chisqr
        #print ('rb, rd, b_frac', result.params['rb'].value, 
               #result.params['rd'].value, 
               #result.params['bf'].value )
        #print 'g1,g2', result.params['g1'].value, result.params['g2'].value
        shape = galsim.Shear(g1=result.params['g1'].value,
                             g2=result.params['g2'].value)
        #print fit_report(result)
    return shape

def cg_ring_test(Args, gal_cg, gal_nocg, chr_PSF):
    """ Ring test to reduce measuremnt bias in shape estimation.
    @param Args         Class with the following attributes:
        Args.npix       Number of pixels across postage stamp image
        Args.scale      Pixel scale of postage stamp image
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian 
                        weight funcion.
    @return  Multiplicate bias estimate.
    """
    star = galsim.Gaussian(half_light_radius=1e-9)*Args.c_SED
    con = galsim.Convolve(chr_PSF,star)
    PSF_img = con.drawImage(Args.bp, nx=Args.npix, ny=Args.npix, scale=Args.scale)
    n = len(Args.rt_g) 
    ghat_cg,ghat_nocg = np.zeros([n,2]),np.zeros([n,2])
    T=n*Args.n_ring*2
    with ProgressBar(T) as bar:
        for i,g in enumerate(Args.rt_g):
            ehat_cg,ehat_nocg = [], []
            for beta in np.linspace(0.0, 360.0, 2*Args.n_ring, endpoint=False):
                gal_cg1 = gal_cg.rotate(beta*galsim.degrees).shear(g1=g[0], g2=g[1])
                gal_nocg1 = gal_nocg.rotate(beta*galsim.degrees).shear(g1=g[0], g2=g[1])
                obj_cg = galsim.Convolve(gal_cg1, chr_PSF)
                obj_nocg = galsim.Convolve(gal_nocg1, chr_PSF)
                img_cg = obj_cg.drawImage(bandpass=Args.bp,
                                          nx=Args.npix, ny=Args.npix,
                                          scale=Args.scale)
                img_nocg = obj_nocg.drawImage(Args.bp,
                                              nx=Args.npix, ny=Args.npix,
                                              scale=Args.scale)
                result_cg   = estimate_shape(Args, img_cg, PSF_img, Args.shear_est)
                result_nocg = estimate_shape(Args, img_nocg, PSF_img, Args.shear_est)
                ehat_cg.append((result_cg.e1, result_cg.e2))
                ehat_nocg.append((result_nocg.e1, result_nocg.e2))
            ehat_cg      = np.mean(np.array(ehat_cg), axis=0)
            ehat_cg      = galsim.Shear(e1=ehat_cg[0], e2=ehat_cg[1])
            ehat_nocg    = np.mean(np.array(ehat_nocg), axis=0)
            ehat_nocg    = galsim.Shear(e1=ehat_nocg[0], e2=ehat_nocg[1])
            ghat_cg[i]   = [ehat_cg.g1, ehat_cg.g2]
            ghat_nocg[i] = [ehat_nocg.g1, ehat_nocg.g2]
            bar.update()
    return ghat_cg.T,ghat_nocg.T



def getFWHM(image):
    """Calculate FWHM of image.

    Compute the circular area of profile that is greater than half the maximum
    value. The diameter of this circle is the FWHM. Note: Method applicable 
    only to circular profiles.
    @param image    Array of profile whose FWHM is to be computed
    @return         FWHM in pixels"""
    mx = image.max()
    ahm =  (image > mx/2.0).sum() 
    return np.sqrt(4.0/np.pi * ahm)

def getHLR(image):
    """Function to calculate Half light radius of image.

    Compute the flux within a circle of increasing radius, till the enclosed 
    flux is greater than half the total flux. Lower bound on HLR is calculated
    from the FWHM. Note: Method applicable only to circular profiles.

    @param image    Array of profile whose half light radius(HLR) is to be computed.
    @return         HLR in pixels"""
    max_x,max_y=np.unravel_index(image.argmax(), image.shape) # index of max value; center
    flux = image.sum()
    # fwhm ~ 2 HLR. HLR will be larger than fwhm/4
    low_r=getFWHM(image)/4.                                   
    for r in range(np.int(low_r),len(image)/2):
        if get_rad_sum(image,r,max_x,max_y)>flux/2.:
            return r-1

def get_rad_sum(image,ro,xo,yo):
    """Compute the total flux of image within a given radius.

    Function is implmented in getHLR to compute half light radius.
    @param image    Array of profile.
    @param ro       radius within which to calculate the total flux in pixel
                    (in pixels).
    @xo,yo          center of the circle within which to calculate the total
                    flux (in pixels) .
    @return         flux within given radius. """
    area=0.
    xrng=range(xo-ro,xo+ro)
    yrng=range(yo-ro,yo+ro)
    for x in xrng:
        for y in yrng:
            if (x-xo)**2+(y-yo)**2 <ro**2 :
                area+=image[x,y]
    return area 

class Eu_Args():
    """Class containing input parameters for Euclid"""
    def __init__(self,npix=360, scale=0.05,
                 psf_sig_o=0.102, psf_w_o=800,
                 sig_w=None,shear_est='REGAUSS',
                 redshift=0.3, alpha=1,
                 disk_n=1.0,bulge_n=1.5,
                 disk_e=[0.0,0.0],bulge_e=[0.0,0.0],
                 bulge_HLR=0.17,disk_HLR=1.2,
                 bulge_frac=0.25,n_ring=3,
                 rt_g=[[0.01,0.01]],res=0.5):
        self.telescope='Euclid'
        self.npix=npix
        self.scale=scale
        self.psf_sigma_o=psf_sig_o
        self.psf_w_o=psf_w_o
        self.bulge_HLR=bulge_HLR
        self.disk_HLR=disk_HLR
        self.redshift=redshift
        self.bulge_n=bulge_n
        self.disk_n=disk_n
        self.bulge_e=bulge_e
        self.disk_e=disk_e
        self.bulge_frac=bulge_frac
        self.disk_SED_name='Im'
        self.bulge_SED_name='E'
        self.b_SED=None
        self.d_SED=None
        self.c_SED=None
        self.sig_w=sig_w
        self.bp=None
        self.shear_est=shear_est
        self.n_ring=n_ring
        self.alpha=alpha
        self.res=res
        self.T_flux=1.
        self.rt_g=rt_g

class LSST_Args():
    """Class containing input parameters for LSST
    @npix    Number of pixels across postage stamp image.
    @scale  Pixel scale of postage stamp image.
    @psf_sigma_o   Gaussian sigma of PSF at known wavelength psf_w_o.
    @psf_w_o       Wavelength at which PSF size is known (nm).
    @alpha         PSF wavelength scaling exponent.  1.0 for diffraction 
                   limit, -0.2 for Kolmogorov turbulence.
    @sig_w         Sigma of Gaussian weight function. 
    @bulge_n       Sersic index of bulge.
    @bulge_HLR     Half light radius of the bulge.
    @bulge_e       Shape of bulge [e1, e2].
    @bulge_frac    Fraction of flux in bulge at 550 nm rest-frame.
    @disk_n        Sersic index of disk.
    @disk_HLR      Half-light-radius of the disk.
    @disk_e        Shape of disk [e1, e2].
    """
    def __init__(self,npix=360, scale=0.2,
                 psf_sigma_o=0.297, psf_w_o=550,
                 alpha=-0.2, redshift=0.3,
                 sig_w=0.8, shear_est='REGAUSS',                 
                 disk_n=1.0, bulge_n=1.5,
                 disk_e=[0.0, 0.0], bulge_e=[0.0, 0.0],
                 bulge_HLR=0.17, disk_HLR=1.2,
                 bulge_frac=0.25, n_ring=3,
                 rt_g=[[0.01,0.01]], filter_name='r' ):
        self.telescope ='LSST'
        self.npix = npix
        self.scale = scale
        self.psf_sigma_o = psf_sigma_o
        self.psf_w_o = psf_w_o
        self.bulge_HLR = bulge_HLR
        self.disk_HLR = disk_HLR
        self.redshift = redshift
        self.bulge_n = bulge_n
        self.disk_n = disk_n
        self.bulge_e = bulge_e
        self.disk_e = disk_e
        self.bulge_frac = bulge_frac
        self.disk_SED_name='Sbc'
        self.bulge_SED_name='E'
        self.b_SED=None
        self.d_SED=None
        self.c_SED=None
        self.sig_w=sig_w
        self.bp=None
        self.shear_est=shear_est
        self.n_ring=n_ring
        self.alpha=alpha
        self.filter_name = filter_name

        self.T_flux=1.
        self.rt_g=rt_g

def calc_cg(Args, calc_weight=False):
    """Compute shape of galaxy with CG and galaxy with no CG 
    @param Args         Class with the following attributes:
        Args.telescope  Telescope the CG bias of which is to be meaasured
                        (Euclid or LSST)
        Args.bp         GalSim Bandpass describing filter.
        Args.b_SED      SED of bulge.
        Args.d_SED      SED of disk.
        Args.c_SED      Flux weighted composite SED.
        Args.scale      Pixel scale of postage stamp image.
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.  See `estimate_shape` docstring.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian weight funcion.
    @return  Shape of galaxy with CG, shape of galaxy with no CG ."""
    Args.b_SED,Args.d_SED,Args.c_SED= get_SEDs(Args)
    if Args.telescope is 'Euclid':
        Args.bp = make_Euclid_filter(Args.res)
    elif Args.telescope is 'LSST': 
        Args.bp = get_LSST_filter(Args.filter_name)
    chr_psf = get_PSF(Args)
    gal_cg = get_gal_cg(Args)
    gal_nocg = get_gal_nocg(Args, gal_cg, chr_psf)
    #compute HLR of galaxy with CG and set it as the size of the weight function
    if calc_weight is True:
        con_cg = (galsim.Convolve(gal_cg,chr_psf))
        im1 = con_cg.drawImage(Args.bp, nx=Args.npix, ny=Args.npix, scale=Args.scale )
        Args.sig_w = (getHLR(im1.array)*Args.scale)
        print 'HLR for redshift={0} is {1}'.format(Args.redshift, Args.sig_w)
    return cg_ring_test(Args, gal_cg, gal_nocg, chr_psf)


#!!!!!!!!! Functions for SED with photometric weights!!!!!!!!!!!

def new_cSED(Args, psf, sig_wp ):
    """ Compute new bulge fraction and composite SED with photometric weight
    function.

    New bulge fraction is computed as ratio of weighted 
    bulge and galaxy image. New Composite SED uses this bulge fraction to 
    combine bulge and disk SEds

    @param Args    Class with the following attributes:
        Args.bulge_n       Sersic index of bulge.
        Args.bulge_HLR     Half-light-radius of the bulge.
        Args.bulge_frac    Fraction of flux in bulge at 550 nm rest-frame.
        Args.T_flux        Total flux in the galaxy.
        Args.disk_n        Sersic index of disk.
        Args.disk_HLR      Half-light-radius of the disk.
        Args.b_SED         SED of bulge.
        Args.d_SED         SED of disk.
    psf                    Chromatic PSF
    sig_wp                 Width (sigma) of Gaussian photometric weight function  
    @return New bulge fraction and composite SED, computed with photo weight 
    function.
    """
    bulge = galsim.Sersic(n=Args.bulge_n, half_light_radius=Args.bulge_HLR,
                         flux=Args.T_flux*Args.bulge_frac)*Args.b_SED
    disk  = galsim.Sersic(n=Args.disk_n, half_light_radius=Args.disk_HLR,
                        flux=Args.T_flux*(1-Args.bulge_frac))*Args.d_SED
    bul_con =galsim.Convolve(bulge,psf) 
    gal = bulge+disk
    con = galsim.Convolve(gal,psf)
    weight = galsim.Gaussian(sigma=sig_wp)
    bulge_im = bul_con.drawImage(Args.bp,nx=Args.npix,ny=Args.npix,scale=Args.scale)
    gal_im = con.drawImage(Args.bp,nx=Args.npix,ny=Args.npix,scale=Args.scale)  
    weight_im = weight.drawImage(nx=Args.npix,ny=Args.npix,scale=Args.scale)
    nmtr = (bulge_im.array*weight_im.array).sum()
    dnmtr = (gal_im.array*weight_im.array).sum()
    new_bf = nmtr/dnmtr    
    c_SED = Args.b_SED * new_bf + Args.d_SED * (1. - new_bf)
    return new_bf,c_SED

def cg_shape_wp(Args, sig_wp, calc_weight=False):
    """Measure shape of galaxy with CG, corrected with photometric weight fn
    used correction PSF.
    @param Args         Class with the following attributes:
        Args.telescope  Telescope the CG bias of which is to be meaasured
                        (Euclid or LSST).
        Args.bp         GalSim Bandpass describing filter.
        Args.b_SED      SED of bulge.
        Args.d_SED      SED of disk.
        Args.c_SED      Flux weighted composite SED.
        Args.scale      Pixel scale of postage stamp image.
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.  See `estimate_shape` docstring.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian weight funcion.
    @return New bulge fraction computed with photo weight function.
    @return Shape of galaxy with CG, shape of galaxy with no CG."""
    Args.b_SED,Args.d_SED,Args.c_SED = get_SEDs(Args)
    if Args.telescope is 'Euclid':
        Args.bp  = make_Euclid_filter(Args.res)
    elif Args.telescope is 'LSST': 
        Args.bp = get_LSST_filter(Args.filter_name)
    chr_psf = get_PSF(Args)
    gal_cg = get_gal_cg(Args)
    gal_nocg = get_gal_nocg(Args, gal_cg, chr_psf)
    if calc_weight is True:
        con_cg = (galsim.Convolve(gal_cg,chr_psf))
        im1 = con_cg.drawImage(Args.bp, nx=Args.npix, ny=Args.npix, scale=Args.scale )
        Args.sig_w = getHLR(im1.array)*Args.scale
    new_bf, Args.c_SED = new_cSED(Args, chr_psf, sig_wp)
    return [new_bf, cg_ring_test(Args, gal_cg, gal_nocg, chr_psf)]

#!!!! Alterante to cg_ringtest

def ring_test_single_gal(Args, gal, chr_PSF, noise_sigma=None):
    """ Ring test to measure shape of galaxy.
    @param Args         Class with the following attributes:
        Args.npix       Number of pixels across postage stamp image
        Args.scale      Pixel scale of postage stamp image
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian 
                        weight funcion.
    @return  Multiplicate bias estimate.

    !!! Need to correct ehat
    """
    star = galsim.Gaussian(half_light_radius=1e-9)*Args.c_SED
    con = galsim.Convolve(chr_PSF,star)
    PSF_img = con.drawImage(Args.bp, nx=Args.npix, ny=Args.npix, scale=Args.scale)
    n = len(Args.rt_g) 
    ghat = np.zeros([n,2])
    T = n*Args.n_ring*2
    random_seed = 141508
    rng = galsim.BaseDeviate(random_seed)
    with ProgressBar(T) as bar:
        for i,g in enumerate(Args.rt_g):
            gsum = []
            betas = np.linspace(0.0, 360.0, 2*Args.n_ring, endpoint=False)/2.
            for beta in betas:
                gal1 = gal.rotate(beta*galsim.degrees).shear(g1=g[0], g2=g[1])
                obj = galsim.Convolve(gal1, chr_PSF)
                img = obj.drawImage(bandpass=Args.bp,
                                    nx=Args.npix, ny=Args.npix,
                                    scale=Args.scale)
                if noise_sigma:
                    gaussian_noise = galsim.GaussianNoise(rng, noise_sigma)
                    img.addNoise(gaussian_noise)
                result   = estimate_shape(Args, img, PSF_img, Args.shear_est)
                gsum.append([result.g1, result.g2])
            gmean = np.mean(np.array(gsum), axis=0)
            ghat[i]   = [gmean[0], gmean[1]]
    return ghat.T

def calc_cg_new(Args, calc_weight=False):
    """Compute shape of galaxy with CG and galaxy with no CG 
    @param Args         Class with the following attributes:
        Args.telescope  Telescope the CG bias of which is to be meaasured
                        (Euclid or LSST)
        Args.bp         GalSim Bandpass describing filter.
        Args.b_SED      SED of bulge.
        Args.d_SED      SED of disk.
        Args.c_SED      Flux weighted composite SED.
        Args.scale      Pixel scale of postage stamp image.
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.  See `estimate_shape` docstring.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian weight funcion.
    @return  Shape of galaxy with CG, shape of galaxy with no CG ."""
    Args.b_SED,Args.d_SED,Args.c_SED = get_SEDs(Args)
    if Args.telescope is 'Euclid':
        Args.bp = make_Euclid_filter(Args.res)
    elif Args.telescope is 'LSST': 
        Args.bp = get_LSST_filter(Args.filter_name)
    chr_psf = get_PSF(Args)
    gal_cg = get_gal_cg(Args)
    gal_nocg = get_gal_nocg(Args, gal_cg, chr_psf)
    #compute HLR of galaxy with CG and set it as the size of the weight function
    if calc_weight is True:
        con_cg = (galsim.Convolve(gal_cg,chr_psf))
        im1 = con_cg.drawImage(Args.bp, nx=Args.npix, ny=Args.npix, scale=Args.scale )
        Args.sig_w = (getHLR(im1.array)*Args.scale)
        print 'Sigma of weight fn:', Args.sig_w
    g_cg = ring_test_single_gal(Args, gal_cg, chr_psf)
    g_ncg = ring_test_single_gal(Args, gal_nocg, chr_psf)
    return g_cg, g_ncg

def calc_shape_galcg(Args, calc_weight=False):
    """Compute shape of galaxy with CG only.
    @param Args         Class with the following attributes:
        Args.telescope  Telescope the CG bias of which is to be meaasured
                        (Euclid or LSST)
        Args.bp         GalSim Bandpass describing filter.
        Args.b_SED      SED of bulge.
        Args.d_SED      SED of disk.
        Args.c_SED      Flux weighted composite SED.
        Args.scale      Pixel scale of postage stamp image.
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.  See `estimate_shape` docstring.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian weight funcion.
    @return  Shape of galaxy with CG, shape of galaxy with no CG ."""
    Args.b_SED,Args.d_SED,Args.c_SED= get_SEDs(Args)
    if Args.telescope is 'Euclid':
        Args.bp = make_Euclid_filter(Args.res)
    elif Args.telescope is 'LSST': 
        Args.bp = get_LSST_filter(Args.filter_name)
    chr_psf = get_PSF(Args)
    gal_cg = get_gal_cg(Args)
    g_cg = ring_test_single_gal(Args, gal_cg, chr_psf)
    return g_cg

def calc_shape_galnocg(Args, calc_weight=False):
    """Compute shape of galaxy with no CG only.
    @param Args         Class with the following attributes:
        Args.telescope  Telescope the CG bias of which is to be meaasured
                        (Euclid or LSST)
        Args.bp         GalSim Bandpass describing filter.
        Args.b_SED      SED of bulge.
        Args.d_SED      SED of disk.
        Args.c_SED      Flux weighted composite SED.
        Args.scale      Pixel scale of postage stamp image.
        Args.n_ring     Number of intrinsic ellipticity pairs around ring.
        Args.shear_est  Method to use to estimate shape.  See `estimate_shape` docstring.
        Args.sig_w      For S13 method, the width (sigma) of the Gaussian weight funcion.
    @return  Shape of galaxy with CG, shape of galaxy with no CG ."""
    Args.b_SED,Args.d_SED,Args.c_SED= get_SEDs(Args)
    if Args.telescope is 'Euclid':
        Args.bp = make_Euclid_filter(Args.res)
    elif Args.telescope is 'LSST': 
        Args.bp = get_LSST_filter(Args.filter_name)
    chr_psf = get_PSF(Args)
    gal_cg = get_gal_cg(Args)
    gal_nocg = get_gal_nocg(Args, gal_cg, chr_psf)
    #compute HLR of galaxy with CG and set it as the size of the weight function
    if calc_weight is True:
        con_cg = (galsim.Convolve(gal_cg,chr_psf))
        im1 = con_cg.drawImage(Args.bp, nx=Args.npix, ny=Args.npix, scale=Args.scale )
        Args.sig_w = (getHLR(im1.array)*Args.scale)
    g_ncg = ring_test_single_gal(Args, gal_nocg, chr_psf)
    return g_ncg











