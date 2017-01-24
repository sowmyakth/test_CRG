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
"""

import galsim
import os
import extinction
import meas_cg_fns as mcgf
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

def get_gaussian_PSF(Args):
    """ Return a chromatic PSF. Size of PSF is wavelength dependent.
    @param     Class with the following attributes:
        sigma_o   Gaussian sigma of PSF at known wavelength Args.psf_w_o.
        w_o       Wavelength at which PSF size is known (nm).
        alpha     PSF wavelength scaling exponent.  1.0 for diffraction 
                           limit, -0.2 for Kolmogorov turbulence.
    @return chromatic PSF.
    """
    mono_PSF = galsim.Gaussian(sigma=Args.sigma_o)
    chr_PSF = galsim.ChromaticObject(mono_PSF).dilate(lambda w: (w/Args.w_o)**Args.alpha)
    return chr_PSF

class psf_params():
    """Parametrs describing a chromatic LSST PSF"""
    def __init__(self, sigma_o=0.297,
                 w_o=550, alpha=-0.2):
        self.sigma_o = sigma_o
        self.w_o = w_o
        self.alpha = alpha

class meas_args():
    """Class containing input parameters for measurement
    @npix    Number of pixels across postage stamp image.
    @scale  Pixel scale of postage stamp image.
    @psf_sigma_o   Gaussian sigma of PSF at known wavelength psf_w_o.
    """
    def __init__(self, npix=360, scale=0.2,
                 shear_est='REGAUSS', n_ring=3,
                 rt_g=[[0.01,0.01]], filter_name='r' ):
        self.npix = npix
        self.scale = scale
        self.shear_est = shear_est
        self.n_ring = n_ring
        self.filter_name = filter_name
        self.rt_g = rt_g
        self.sig_w = 0.
        self.c_SED = None
        self.bp = None

def get_LSST_filter(filter_name='r', file_path=None):
    """ Return LSST filter stored in llst stack
    @param filter_name    Name of LSST optical photometric bands (u,g,r,i,z,y).
    @param file path      Path where file is saved
    @return               galsim.Bandpass object. 
    """
    if file_path is None:
            file_path = galsim.meta_data.install_dir + '/examples/data/'
            filter_filename = file_path +'LSST_{}.dat'.format(filter_name)
            filter_bandpass = galsim.Bandpass(filter_filename, 
                                      wave_type='Ang').thin(rel_err=1e-4)
            
    else:
        filter_filename = file_path + 'filter_{}.dat'.format(filter_name)
        filter_bandpass = galsim.Bandpass(filter_filename, 
                                      wave_type='nm').thin(rel_err=1e-4)
    
    return filter_bandpass

def redden(SED, a_v, r_v=3.1, model='ccm89' ):
    """ Return a new SED with the specified extinction applied.  Note that this will
    truncate the wavelength range to lie between 91 nm and 6000 nm where the extinction
    correction law is defined.
    """
    return SED/(lambda w: extinction.reddening(w * 10, a_v=a_v,
                                                   r_v=r_v, model=model))

def get_catsim_SED(sed_name, redshift=0.,
                   a_v=None, r_v=3.1, model='ccm89' ):
    """Returns SED of a galaxy in CatSim catalog corrected for redshift and extinction.
    @param sed_file    Name(wih path) of file with SED information.
    @flux_norm         Multiplicative scaling factor to apply to the SED.
    @bandpass          GalSim bandpass object which models the transmission fraction.
    @ redshift         Redshift of the galaxy.
    @a_v               Total V band extinction, in magnitudes.
    @r_v               Extinction R_V parameter.
    @model             Dust model name in the object's rest frame.
    """
    sed_path = '/nfs/slac/g/ki/ki19/deuce/AEGIS/LSST_cat/galaxySED/'
    if sed_name == str(-1):
        SED =  galsim.SED(lambda w: 1,  wave_type='nm',
                          flux_type='flambda')
    else:
        full_sed_name = sed_path + sed_name +'.gz' 
        SED = galsim.SED(full_sed_name, wave_type='nm',
                     flux_type='flambda')
    SED = redden(SED, a_v, r_v, model=model)
    SED = SED.atRedshift(redshift)
    return SED*30*np.pi*(6.67*100/2.)**2

def get_chrom_Sersic(args):
    """Returns Sersic with parametrs defined in args"""
    ser = galsim.Sersic(n=args.ns, half_light_radius=args.re,
                        flux=args.f0)
    if ~np.isnan(args.e):
        ser = ser.shear(e=args.e, beta=args.phi*galsim.degrees)
    ser = ser*args.sed_type
    return ser

def get_catsim_gal(cat):
    sed_d = get_catsim_SED(cat['sedname_disk'], redshift=cat['redshift'],
                       a_v=cat['av_d'],
                       r_v=cat['rv_d'], model='ccm89' )
    sed_b = get_catsim_SED(cat['sedname_bulge'], redshift=cat['redshift'],
                        a_v=cat['av_b'],
                        r_v=cat['rv_b'],model='ccm89' )
    #composite SED
    c_sed =  sed_b*cat['fluxnorm_bulge'] + sed_d*cat['fluxnorm_disk'] 
    re_b, e_b = mcgf.a_b2re_e(cat['a_b'],
                             cat['b_b'])
    re_d, e_d = mcgf.a_b2re_e(cat['a_d'],
                             cat['b_d'])
    n_b = cat['bulge_n']
    n_d = cat['disk_n']
    phi_b = cat['pa_bulge']
    phi_d = cat['pa_disk']
    params_bulge = mcgf.ser_params(x0=0, y0=0, f0=1, re=re_b,
                                 ns=n_b, e=e_b, phi=phi_b,
                                 sed_type=sed_b)
    params_disk = mcgf.ser_params(x0=0, y0=0, f0=1, re=re_d,
                                 ns=n_d, e=e_d, phi=phi_d,
                                 sed_type=sed_d)
    bulge = get_chrom_Sersic(params_bulge)*cat['fluxnorm_bulge']
    disk = get_chrom_Sersic(params_disk) *cat['fluxnorm_disk']
    gal =  bulge+ disk
    return gal, c_sed

def get_gal_nocg(Args, gal_cg,
                 chr_PSF):
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
    	                                  calculate_stepk=False,
                                          calculate_maxk=False)
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
        try:
            result = galsim.hsm.EstimateShear(gal_img, PSF_img, shear_est=method,
        	                                 hsmparams = new_params)
            shape = galsim.Shear(e1=result.corrected_e1, e2=result.corrected_e2)
        except:
            return "Fail"

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

#!!!! Alterante to cg_ringtest

def ring_test_single_gal(Args, gal,
                        chr_PSF, noise_sigma=None):
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
            if result is "Fail":
                return "Fail"
            gsum.append([result.g1, result.g2])
        gmean = np.mean(np.array(gsum), axis=0)
        ghat[i]   = [gmean[0], gmean[1]]
    return ghat.T

def calc_cg_catsim(cat, meas_args,
                   psf_args, calc_weight=False):
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
    @param cal_weight   if True, manually computes size of galaxy and sets it as weight size
    @return  Shape of galaxy with CG, shape of galaxy with no CG ."""
    chr_psf = get_gaussian_PSF(psf_args)
    gal_cg, c_sed = get_catsim_gal(cat)
    meas_args.c_SED = c_sed
    gal_nocg = get_gal_nocg(meas_args, gal_cg,
                           chr_psf)
    #compute HLR of galaxy with CG and set it as the size of the weight function
    if calc_weight is True:
        con_cg = (galsim.Convolve(gal_cg,chr_psf))
        im1 = con_cg.drawImage(Args.bp, nx=meas_args.npix,
                               ny=meas_args.npix, scale=Args.scale)
        meas_args.sig_w = (getHLR(im1.array)*meas_args.scale)
        print 'Sigma of weight fn:', sig_w
    g_cg = ring_test_single_gal(meas_args, gal_cg,
                                chr_psf)
    g_ncg = ring_test_single_gal(meas_args, gal_nocg,
                                 chr_psf)
    return g_cg, g_ncg

def calc_cg_crg(crg, meas_args,
                psf_args, calc_weight=False):
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
    @param cal_weight   if True, manually computes size of galaxy and sets it as weight size
    @return  Shape of galaxy with CG, shape of galaxy with no CG ."""
    chr_psf = get_gaussian_PSF(psf_args)
    gal_cg = crg
    meas_args.c_SED = crg.SED
    gal_nocg = get_gal_nocg(meas_args, gal_cg,
                           chr_psf)
    #compute HLR of galaxy with CG and set it as the size of the weight function
    if calc_weight is True:
        con_cg = (galsim.Convolve(gal_cg,chr_psf))
        im1 = con_cg.drawImage(Args.bp, nx=meas_args.npix,
                               ny=meas_args.npix, scale=Args.scale)
        meas_args.sig_w = (getHLR(im1.array)*meas_args.scale)
        print 'Sigma of weight fn:', sig_w
    g_cg = ring_test_single_gal(meas_args, gal_cg,
                                chr_psf)
    g_ncg = ring_test_single_gal(meas_args, gal_nocg,
                                 chr_psf)
    return g_cg, g_ncg






