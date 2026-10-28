# -*- coding: utf-8 -*-
"""
Created on Oct 28 2026

@author: Matthias Cuntz

This subroutine is the FREQCOR_Read.py routine,
adapted for data coming from HEddySoft

Inputs :
      - HEddySoft (half-)hourly output files
Outputs :
      - (co)spectra matrices:
        icon(nfreq, nspec),
        rcon(nfreq, nspec) (either cospectra or spectra; depends on `sps`);
      - meteo and flux: x vectors(nspec);
      - General Flags: x vectors(nspec);

Comments :
    - filtering is applied :
        - for H, using Vitale 2020 tests and user limit on absolute value
          (from the config)
        - for N2O flux, using Vitale 2020 tests and user limit on absolute
          value (from the config)
        - for VOC flux, using only abs(flux) > percentile 90 of absolute
          value and removal of plume events
    - ! the mz choice is hard-coded, at the beginning of the routine !
    - some variables are missing in the HEddySoft output but are needed
      by FREQCOR for filtering steps. They are therefore created, filled with
      values that will end in no-filtering, and returned, to mimic FREQCOR_Read
TO DO:
    - target_mz should not be hard-coded but aim was to not modify the FREQCOR
      structure
"""
import datetime
import glob
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FREQCOR_Read_GEddySoft import calculate_freqcount, filtering


__all__ = ['FREQCOR_Read_HEddySoft']


def FREQCOR_Read_HEddySoft(config, jsite):
    # %%  Initialisation

    # USER CHOICE !
    # mz to be selected (will choose the closest one)
    # 33.033 : proposed flux treshold of 0.1
    # 45.033 : proposed flux treshold of 0.01
    # 69.0699 : proposed flux treshold of 0.5
    # 137.132 : proposed flux treshold of 0.2
    target_mz = 44.013
    apply_filtering = True
    plot = True

    # Collect all candidate files
    total_to_process = glob.glob(config['IO']['input_path'] + '/' +
                                 config['IO']['binned_cosp'] + '/*.pkl')
    total_to_process.sort()

    # Filter files by date in filename if time window enabled
    if config['TIME_WINDOW']['enable_time_window'] == '1':
        # Read time window parameters
        start_datetime_str = config['TIME_WINDOW']['start_datetime']
        end_datetime_str = config['TIME_WINDOW']['end_datetime']
        # Parse datetime strings to datetime objects
        start_datetime = datetime.datetime.strptime(start_datetime_str,
                                                    '%Y-%m-%d %H:%M')
        end_datetime = datetime.datetime.strptime(end_datetime_str,
                                                  '%Y-%m-%d %H:%M')

        # Filter files by date in filename
        # inclusive date-range based on filename day
        # (since filename has no time)
        start_day = start_datetime.date()
        end_day = end_datetime.date()
        date_pattern = re.compile(r'spectral_(\d{14})\.pkl$', re.IGNORECASE)
        filtered = []
        skipped_badname = 0
        for filen in total_to_process:
            base = os.path.basename(filen)
            m = date_pattern.search(base)
            if not m:
                skipped_badname += 1
                continue
            yyyymmddhhmmss = m.group(1)
            try:
                file_day = datetime.datetime.strptime(yyyymmddhhmmss,
                                                      '%Y%m%d%H%M%S').date()
            except ValueError:
                skipped_badname += 1
                continue
            if start_day <= file_day <= end_day:
                filtered.append(filen)
        total_to_process = sorted(filtered)
        print(f'Selected {len(total_to_process)} file(s) within window days'
              f' [{start_day} .. {end_day}]')
        if skipped_badname:
            print(f'Skipped {skipped_badname} file(s) due to unexpected'
                  f' name/date format')

    # %% loop over HEddySoft daily output files

    ispecies = int(config['PROCEDURE_OPTIONS']['gss'])
    if ispecies == 1:
        species = 'co2'
    elif ispecies == 2:
        species = 'h2o'
    else:
        raise ValueError('Species not implemented in HEddySoft')

    # loop over the daily files in the given directory matching the
    # name pattern
    sdf = pd.DataFrame()
    idf = pd.DataFrame()
    icon = pd.DataFrame()
    rcon = pd.DataFrame()
    for filen in total_to_process:

        ldf = pickle.load(open(filen, 'rb'))
        # ['TIMESTAMP_END', 'wT', 'wind_speed', 'wind_direction', 'u_star',
        #  'z/L', 'num_spikes_w', 'itc_w', 'steady_state_wc_co2_Mahrt1998',
        #  'steady_state_wc_h2o_Mahrt1998', 'F_co2', 'F_h2o']
        isdf = ldf['scalar']
        sdf = pd.concat([sdf, isdf])

        # ipt_w, ipt_co2, ipt_h2o
        instrument_df = ldf['instrument']
        iidf = pd.DataFrame(
            {'OOR_w': instrument_df['ipt_w'].iloc[11],
             'HF5': instrument_df[f'ipt_{species}'].iloc[4],
             'HD5': instrument_df[f'ipt_{species}'].iloc[6],
             'KID': instrument_df[f'ipt_{species}'].iloc[3],
             'DIP': instrument_df[f'ipt_{species}'].iloc[10],
             'DDI': instrument_df[f'ipt_{species}'].iloc[9]},
            index=isdf.index)
        iidf.index.name = isdf.index.name
        idf = pd.concat([idf, iidf])

        # freq, cospec_wT_scaled, spec_co2_scaled, cospec_wc_co2_scaled,
        # spec_h2o_scaled, cospec_wc_h2o_scaled
        spectral_df = ldf['spectral']
        freqm = spectral_df['freq']

        iicon = pd.DataFrame(spectral_df['cospec_wT_scaled'].T,
                             columns=freqm.values, index=isdf.index)
        iicon.index.name = iicon.index.name
        icon = pd.concat([icon, iicon])

        if config['PROCEDURE_OPTIONS']['sps'] == '1':
            # Real cospectral data from file (VOC)
            rron = spectral_df[f'cospec_wc_{species}_scaled']
        elif config['PROCEDURE_OPTIONS']['sps'] == '2':
            # Real spectral data from file (VOC)
            rron = spectral_df[f'spec_{species}_scaled']
        ircon = pd.DataFrame(rron.T, columns=freqm.values, index=isdf.index)
        ircon.index.name = ircon.index.name
        rcon = pd.concat([rcon, ircon])

    sdf = pd.concat([sdf, idf], axis=1)
    sdf['H'] = sdf['wT'] * 1206.

    # %% Preparation of variables

    # Natural frequency of each bin
    freqn = np.divide(freqm * (float(config['EC_SETUP']['zmeas']) -
                               float(config['EC_SETUP']['disph'])),
                      sdf['wind_speed'].values.reshape(1, -1))
    
    # create freqcount, the population in each bin.
    # Present in the EP outputs but not in the GEddySoft outputs
    freqcount = calculate_freqcount(freqm.to_numpy(), 1800.0)

    # Division of (co)spectral density by frequency
    # (because GEddysoft output gives  f_nat*cospec/cov  or f_nat*spec/var)
    icon = icon.div(freqm)
    rcon = rcon.div(freqm)

    # %% Filtering and cleaning

    # based on raw data flagging
    if apply_filtering:
        FlagVM_w, FlagVM_T, FlagVM_g, FlagF_H, FlagF_g = filtering(
            sdf['HF5'], sdf['HD5'], sdf['KID'], sdf['DIP'], sdf['DDI'],
            sdf[f'steady_state_wc_{species}_Mahrt1998'], sdf['ITC_w'],
            sdf['OOR_w'])

    # Remove freq bins fully filled with Nans
    idx = icon.notna().any(axis=1) & rcon.notna().any(axis=1)
    freqm = freqm[idx]
    freqn = freqn[idx]
    icon = icon[idx]
    rcon = rcon[idx]
    
    # filtering based on full NaNs or null data
    idx = (icon.notna().any(axis=0) & 
           rcon.notna().any(axis=0) & 
           (~sdf['wind_speed'].isnull().values))

    freqm = freqm.loc[:, idx]
    freqn = freqn.loc[:, idx]
    icon = icon.loc[:, idx]
    rcon = rcon.loc[:, idx]
    idx = idx.reset_index(drop=True)
    sdf = sdf.loc[idx, :]
    FlagVM_w = FlagVM_w[idx]
    FlagVM_T = FlagVM_T[idx]
    FlagVM_g = FlagVM_g[idx]
    FlagF_H = FlagF_H[idx]
    FlagF_g = FlagF_g[idx]

    nspec = sum(idx)
    
    # Reset the indexes so to have a continuous integer series of indexes
    freqm = freqm.reset_index(drop=True)
    freqn = freqn.reset_index(drop=True)
    icon = icon.reset_index(drop=True)    
    rcon = rcon.reset_index(drop=True)    
    sdf = sdf.reset_index(drop=True)

    # %% 
    
    # Series of natural frequencies, taken as the first column of the freqm
    # DataFrame where all the columns should contain the same values (see here above)
    freq = freqm.copy()
    freq = freq.iloc[:, 0]
    # Same for normalised frequencies
    # freqn = freqn.iloc[:,0]
   
    # Cleaned (co)spectra vector length (number of actually present frequency bins)
    nfreq = freq.shape[0]

    #%% Read Massman coefficients from file
    if config['IO']['massman_path']:
        massman_coef = pd.read_csv(config['IO']['massman_path'], header=0, comment='#')
        massman_coef = massman_coef.set_index('var')
        massman_coef = massman_coef.loc[:,jsite]
        
        A0 = [float(x) for x in massman_coef['A0'].split(',')]
        kf0 = [float(x) for x in massman_coef['kf0'].split(',')]
        mu = [float(x) for x in massman_coef['mu'].split(',')]

        massman_coef = [A0, kf0, mu]
    else:
        massman_coef = None

    # create a copy of icon for further use
    icosp = icon.copy()
    
    if plot:    
        # Use the first column as the time axis
        time = sdf.index
    
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8),
                                 sharex=True, sharey=False)
    
        # First subplot: H flux
        axes[0].scatter(time, sdf['wT'], s=10, color='black', label='H flux')
        axes[0].scatter(time[FlagVM_w > 0],
                        sdf.loc[FlagVM_w > int(config['USER_LIMITS']['fvm']), 'H'],
                        s=10, color='green',
                        label=f'FlagVM_w > {config["USER_LIMITS"]["fvm"]}',
                        alpha=0.7)
        axes[0].scatter(time[FlagVM_T > 0],
                        sdf.loc[FlagVM_T > int(config['USER_LIMITS']['fvm']), 'H'],
                        s=10, color='purple',
                        label=f'FlagVM_T > {config["USER_LIMITS"]["fvm"]}',
                        alpha=0.7)
        axes[0].scatter(time[FlagF_H > 0],
                        sdf.loc[FlagF_H > int(config['USER_LIMITS']['ff']), 'H'],
                        s=10, color='red',
                        label=f'FlagF_H > {config["USER_LIMITS"]["ff"]}',
                        alpha=0.7)

        percentile_90 = np.percentile(abs(sdf.loc[:, 'H']), 90)
        axes[0].axhline(y=percentile_90, color='red', linestyle='--',
                        label='90th percentile')
        axes[0].axhline(y=-percentile_90, color='red', linestyle='--',
                        label='-90th percentile')
        axes[0].axhline(y=float(config['USER_LIMITS']['hlim']), color='green',
                        linestyle='--', label='user limit')
        axes[0].set_title('H flux')
        axes[0].legend()
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('H')
    
        # Second subplot: gas flux
        axes[1].scatter(time, sdf.loc[:, 'Fc'], s=10, color='black',
                        label='Fgas flux')

        axes[1].scatter(time[FlagF_H > 0],
                        sdf.loc[FlagF_H > int(config['USER_LIMITS']['ff']), 'Fc'],
                        s=10, color='red',
                        label=f'FlagF_H > {config["USER_LIMITS"]["ff"]}',
                        alpha=0.7)
        axes[1].scatter(time[FlagF_g > 0],
                        sdf.loc[FlagF_g > int(config['USER_LIMITS']['ff']), 'Fc'],
                        s=10, color='blue',
                        label=f'FlagF_g > {config["USER_LIMITS"]["ff"]}',
                        alpha=0.7)
        axes[1].scatter(time[FlagVM_w > 0],
                        sdf.loc[FlagVM_w > int(config['USER_LIMITS']['fvm']), 'Fc'],
                        s=10, color='green',
                        label=f'FlagVM_w > {config["USER_LIMITS"]["fvm"]}',
                        alpha=0.7)
        axes[1].scatter(time[FlagVM_T > 0],
                        sdf.loc[FlagVM_T > int(config['USER_LIMITS']['fvm']), 'Fc'],
                        s=10, color='purple',
                        label=f'FlagVM_T > {config["USER_LIMITS"]["fvm"]}',
                        alpha=0.7)
        axes[1].scatter(time[FlagVM_g > 0],
                        sdf.loc[FlagVM_g > int(config['USER_LIMITS']['fvm']), 'Fc'],
                        s=10, color='orange',
                        label=f'FlagVM_g > {config["USER_LIMITS"]["fvm"]}',
                        alpha=0.7)

        percentile_90 = np.percentile(abs(sdf.loc[:, 'Fc']), 90)
        axes[1].axhline(y=percentile_90, color='red', linestyle='--',
                        label='90th percentile')
        axes[1].axhline(y=-percentile_90, color='red', linestyle='--',
                        label='-90th percentile')
        axes[1].axhline(y=float(config['USER_LIMITS']['fclim']),
                        color='green', linestyle='--', label='user limit')
        axes[1].axhline(y=-float(config['USER_LIMITS']['fclim']),
                        color='green', linestyle='--', label='user limit')
        axes[1].set_title('Fgas flux')
        axes[1].legend()
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Fgas')
    
        plt.tight_layout()

        if int(config['PROCEDURE_OPTIONS']['plot_save']) == 1:
            plt.savefig(config['IO']['output_path'] + '/flux filtering.png')
            plt.close()
        else:
            plt.show()

    # remove the timestamp column
    sdf = sdf.iloc[:, 1:]
    return [nspec, icon, rcon, icosp, nfreq, freq, freqn, freqcount,
            sdf, sdf['wind_speed'], sdf['wind_direction'], sdf['u_star'],
            sdf['z/L'],
            FlagF_H, FlagF_g, FlagVM_w, FlagVM_T, FlagVM_g,
            massman_coef]
