"""Make a mock catalog of galaxies with columns same as real galaxy catalog.
This will serve as test input to ChromaticRealGalaxy
Galaxy parameters will be similar to those used in parametric color gradient study 

Note: pstampos drawn with script make_pstamps.py
input galaxy pstamps name: 'images/HST_filt_gal_num.fits'
catalog redshift: 0:1.2  (15 values)
25 different noise realizations
3 ellipticities, |e|=0.3 (x,y and 45)
same seed for all redshifts and ellipticities
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table,Column, vstack, hstack, join

def get_images(index_table, filt):
    """Make fits files of galaxy and psf postage stamps.
    Each fits file has 100 images with different HDU.
    """
    print "Saving images"
    num = index_table['NUMBER']
    file_nums = np.array(num)/100 + 1
    print "Making %f fits files with galaxy stamps "%max(file_nums)
    col = Column(file_nums, name='file_num')
    index_table.add_column(col)
    col = Column(np.zeros(len(num)), name='HDU')
    index_table.add_column(col)
    col = Column(['HST_V_000_000.fits']*len(num), name='GAL_FILENAME')
    index_table.add_column(col)
    col = Column(['HST_V_000_000.fits']*len(num), name='PSF_FILENAME')
    index_table.add_column(col)
    for j in range(1,max(file_nums)+1):
        val, = np.where(index_table['file_num']== j)
        im_hdul = fits.HDUList()
        psf_hdul = fits.HDUList()
        im_name = 'catalog/HST_filt_gal_n{0}.fits'.format(j).replace('filt', filt)
        psf_name = 'catalog/HST_filt_psf_n{0}.fits'.format(j).replace('filt', filt)
        hdu_count = 0
        for i in val:
            #input galaxy and psf images
            im_file = 'images/HST_filt_gal_{0}.fits'.format(i).replace('filt', filt)
            psf_file =  'images/HST_filt_psf_{0}.fits'.format(i).replace('filt', filt)
            print im_file, i
            h = fits.open(im_file)
            im = h[0].data
            h.close()
            h = fits.open(psf_file)
            psf = h[0].data
            h.close()
            im_hdul.append(fits.ImageHDU(im))
            psf_hdul.append(fits.ImageHDU(psf))
            index_table['HDU'][i] = hdu_count
            index_table['GAL_FILENAME'][i] = 'HST_filt_gal_n{0}.fits'.format(j).replace('filt', filt)
            index_table['PSF_FILENAME'][i] = 'HST_filt_psf_n{0}.fits'.format(j).replace('filt', filt)
            hdu_count+=1
        #output fits file name

        im_hdul.writeto(im_name, clobber=True)
        psf_hdul.writeto(psf_name, clobber=True)
    return index_table

def get_main_catalog(index_table, filt, band_name):
    """Makes main catalog containing information on all galaxies.
    Columns are identical to COSMOS Real Galaxy catalog"""
    print "Creating main catalog"
    cat_name = 'catalog/HST_filt_catalog.fits' 
    noise_file = 'acs_filt_unrot_sci_cf.fits' 
    num = index_table['NUMBER']
    names = ('IDENT', 'RA', 'DEC', 'MAG', 'BAND', 'WEIGHT', 'GAL_FILENAME')
    names+= ('PSF_FILENAME', 'GAL_HDU', 'PSF_HDU', 'PIXEL_SCALE')
    names+= ('NOISE_MEAN', 'NOISE_VARIANCE', 'NOISE_FILENAME', 'stamp_flux')
    dtype = ('i4', 'f8', 'f8', 'f8', 'S40', 'f8', 'S256')
    dtype+= ('S288', 'i4', 'i4', 'f8')
    dtype+= ('f8', 'f8', 'S208', 'f8')

    num = index_table['NUMBER']
    im_names =[]
    psf_names = []
    im_names = index_table['GAL_FILENAME']
    psf_names = index_table['PSF_FILENAME']
    hdus = index_table['HDU']
    n_means = index_table['NOISE_MEAN']
    n_vars = index_table['NOISE_VARIANCE']
    n_names = [noise_file.replace('filt', filt)]*len(num)
    fluxs = index_table['FLUX']
    # ra, dec, magnitude set to 0, weight set to 1
    cols = [num, np.zeros(len(num)), np.zeros(len(num)), np.zeros(len(num)), [band_name]*len(num),
           np.ones(len(num)), im_names, psf_names, hdus, hdus, np.ones(len(num))*0.03,
           n_means, n_vars, n_names, fluxs ]
    table = Table(cols, names=names, dtype=dtype)
    table.write(cat_name.replace('filt', filt), format='fits',
                                  overwrite=True)

def main():
    """Saves Postage stamps and final catalogs in a format that can be read by
    galsim modules"""
    filter_names = ['V', 'I']
    band_name = ['F606W', 'F814W']
    #read input catalog with galaxy parameters
    in_file = 'index_table_filter.fits'
    for f, filt in enumerate(filter_names):
        index_table = Table.read(in_file.replace('filter',filt), format='fits')
        idx = get_images(index_table, filt)
        get_main_catalog(idx, filt, band_name[f])

if __name__== "__main__":
    main()
        
