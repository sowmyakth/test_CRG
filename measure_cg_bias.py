from astropy.table import Table, Column
import galsim
import os
import numpy as np
import cg_revolutions as cgr

def get_LSST_filter(filter_name, file_path):
    """ Returns HST filter.
    @param filter_name    Name of HST optical photometric bands (f606w,f814w).
    @return               galsim.Bandpass object. 
    """
    filter_filename = file_path + 'LSST_{0}.dat'.format(filter_name)
    filter_bandpass = galsim.Bandpass(filter_filename, wave_type='nm').thin(rel_err=1e-4)
    return filter_bandpass

def get_SEDs(redshift):
    """Returns the bulge (E) and disk (Irr) SED that was used in creating the
    parameric galaxies.
    """
    datapath = '/Users/choyma/work/test_cge/make_param_gal/data/'
    b_SED = galsim.SED(datapath+"CWW_{}_ext.sed".format('E'), 
                       wave_type='Ang', flux_type= 'flambda')
    d_SED = galsim.SED(datapath+"CWW_{}_ext.sed".format('Im'), 
                       wave_type='Ang', flux_type= 'flambda')
    b_SED = b_SED.withFluxDensity(1.0, 550.0).atRedshift(redshift)
    d_SED = d_SED.withFluxDensity(1.0, 550.0).atRedshift(redshift)
    return b_SED, d_SED

def main():
    # set to True if true seds are to given as input seds to crg.
    # else default setting uses polynomial seds 
    assert_true_seds =True
    filters=['f606w', 'f814w']
    file_filter_name = ['V', 'I']
    rgc = {}
    for f, filt in enumerate(filters):
        cat_name = 'HST_filter_catalog.fits'.replace('filter', file_filter_name[f])
        rgc[filt] = galsim.RealGalaxyCatalog(cat_name, dir=os.getcwd()+'/catalog/')
    v606 = rgc['f606w'].getBandpass()
    i814 = rgc['f814w'].getBandpass()
    path = '/Users/choyma/work/test_cge/make_param_gal/data/'
    in_file = 'index_table_filter.fits'.replace('filter', file_filter_name[0])
    index_table = Table.read(in_file)
    
    #CHANGE!!!!!!
    #q, = np.where(index_table['redshift']==index_table['redshift'][f_num])
    #q, = np.where(index_table['TYPE']==0)
    #print "measuring bias for %i galaxies "%len(q)
    num = index_table['NUMBER']
    print num
    col = Column(np.zeros([len(num), 2,2]), name='g_cg', 
                 shape=(2,2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([len(num), 2,2]), name='g_no_cg', 
                 shape=(2,2), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([len(num), 2]), name='m_cg', 
                 shape=(2,), dtype='f8')
    index_table.add_column(col)
    col = Column(np.zeros([len(num), 2]), name='c_cg', 
                 shape=(2,), dtype='f8')
    index_table.add_column(col)
    for i, indx in enumerate(num):
        print "Running Galaxy Index: ", indx
        if assert_true_seds is True:
            print "True SEDs are input to CRG"
            b_SED, d_SED = get_SEDs(index_table['redshift'][indx])
            crg = galsim.ChromaticRealGalaxy([rgc['f606w'], rgc['f814w']], index=indx,
                                            SEDs=[b_SED, d_SED])
            out_file = 'results_r_cg_bias_all_z_true_sed.fits'
            print "output to be saved at ", out_file
        else:
            crg = galsim.ChromaticRealGalaxy([rgc['f606w'], rgc['f814w']], index=indx)
            out_file = 'results_r_cg_bias_all_z.fits'
            print "output to be saved at ", out_file
        meas_args = cgr.meas_args(rt_g=[[0.005,0.005],[0.01,0.01]])
        psf_args = cgr.psf_params()
        meas_args.bp = get_LSST_filter(filter_name='r', file_path=path)
        gcg, gnocg = cgr.calc_cg_crg(crg, meas_args, psf_args)
        print gcg
        gtrue = np.array(meas_args.rt_g)
        fit_fin   = np.polyfit(gtrue.T[0],gcg.T-gnocg.T,1)
        index_table['m_cg'][indx] = fit_fin[0]
        index_table['c_cg'][indx] = fit_fin[1]
        index_table['g_cg'][indx] = gcg
        index_table['g_no_cg'][indx] = gnocg
    index_table.write(out_file, format='fits',
                      overwrite=True)

if __name__== "__main__":
    main()