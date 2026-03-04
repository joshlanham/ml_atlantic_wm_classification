import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trainKfold import randomized_kfold_training, ensemble_inference
from trainBlock import R2
#from trainBlock import get_block_test_train, main_training_loop_with_checkpointing, main_training_loop, getXY, train_and_evaluate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#import seaborn as sns
import glob
from netCDF4 import Dataset
import numpy as np
import sys
import os
from itertools import islice
import gsw

def mkdir(dr):
    # Check if directory exists and create if it doesn't
    if not os.path.exists(dr):
        try:
            os.makedirs(dr)
            print(f"Directory {dr} created.")
        except PermissionError:
            print(f"No permission to create {dr}. Exiting.")
            exit()
        except Exception as e:
            print(f"Directory likely exists...An error occurred: {e}")


#recursively loops through a directory and find all files of a certain pattern
def find_nc_files(var, root_path='.' ):
    return [os.path.abspath(file) for file in glob.glob(f'{root_path}/**/{var}*.nc', recursive=True)]


#Extract the Atlantic Ocean indices
def find_lon_range(longitude, lon_min=-90, lon_max=40):
    return np.where((longitude >= lon_min) & (longitude <= lon_max))[0]


#find matched pairs of theta and salt files in root_dir and also create corresponding predictions and uncertainties
#files. The only difference
def fileIO(root_dir, pred_dir = None):
    if pred_dir is None:
        pred_dir = root_dir + '/watermass_ml'
    mkdir(pred_dir)
    files = find_nc_files('OCEAN_TEMPERATURE_SALINITY', root_dir); files.sort()
    print(files)
    #tfiles_base = [os.path.basename(path) for path in tfiles]

    
    #Create salt dictionary which maps timestamp to salt file path
    #salt_dict = {'_'.join(os.path.basename(f).split('_')[1:3]): f for f in sfiles}
    #print(salt_dict)
    files_processed = []
    #sfiles_processed = []
    predfiles = []
    uncfiles = []
    for file in files:
        # Extract time from filename
        time_stamp = os.path.basename(file).split('_')[5]
   
        # Find the corresponding salt file
        #salt_file = salt_dict.get(time_stamp)
        #if salt_file:
            # Define target directory and filename
        predfiles.append(pred_dir + f"/wmass_{time_stamp}")
        uncfiles.append(pred_dir + f"/unc_{time_stamp}")
        files_processed.append(file)
            #sfiles_processed.append(salt_file)
        print(file)
        #else:
        #    print(f'*******salt file does not seem to exist')
    #print(tfiles_processed, sfiles_processed, predfiles, uncfiles)
    return files_processed, predfiles, uncfiles  
# Function to find the index range for limited longitude
def find_lon_range(longitude, lon_min=-90, lon_max=40):
    return np.where((longitude >= lon_min) & (longitude <= lon_max))[0]


def predict_and_write(file, ofile, ofile_unc, num_batches = 5):
# Read NetCDF file for 'theta' and other variables
    with Dataset(file, 'r') as nc:
        nc.set_auto_mask(False)
        nc.set_auto_scale(False)

        theta = nc.variables['THETA'][:]
        salt = nc.variables['SALT'][:]
        
        latitude = nc.variables['latitude'][:]
        longitude = nc.variables['longitude'][:]
        Z = nc.variables['Z'][:]
       
        '''
        theta = nc.variables['THETA'][:]
        latitude = nc.variables['lat'][:]
        longitude = nc.variables['lon'][:]
        Z = nc.variables['dep'][:]
        '''
    # Limit longitude range to the Atlantic Ocean
    lon_indices = find_lon_range(longitude)
    #theta = theta[:, :, :, lon_indices]
    theta = theta[:,:, :, lon_indices]
    salt = salt[:,:, :, lon_indices]
    longitude = longitude[lon_indices]
    # Read 'salt' similarly
    '''with Dataset(sfile, 'r') as nc:
        #salt = nc.variables['SALT'][:, :, :, lon_indices]
        salt = nc.variables['SALT'][0,:, :, lon_indices]'''
    
    #theta[theta==0]=np.nan
    #salt[salt==0]=np.nan
    
    #Flatten data arrays to feed into ML model 
    theta_flat = np.squeeze(theta).ravel()
    salt_flat = np.squeeze(salt).ravel()
    
    
    Z_grid, Y, X = np.meshgrid(Z, latitude, longitude, indexing='ij')
    
    #Z_grid[np.isnan(theta[0,...])==True]=np.nan
    #X[np.isnan(theta[0,...])==True]=np.nan
    #Y[np.isnan(theta[0,...])==True]=np.nan
    
    Z_rep = Z_grid.ravel()
    lat_rep = Y.ravel()
    lon_rep = X.ravel()
    
    bottom_depth_3d = np.load('ecco_bottom_depth_3d.npy')
    bottom_depth_3d = bottom_depth_3d[:, :, lon_indices]
    hab_flat = (bottom_depth_3d + Z_grid).ravel()
    hab_flat = np.where(np.isnan(hab_flat), 0, hab_flat)
    
    
    lon_rep_sin = np.sin(np.deg2rad(lon_rep))
    lon_rep_cos = np.cos(np.deg2rad(lon_rep))
    
    pressure_rep = gsw.p_from_z(Z_rep, lat_rep)
    
    #CONVERTING TO CT
    
    # Mask out invalid values (e.g., ECCO fill values like 1e20)
    mask = theta_flat > 1e5

    # Replace masked values with neutral placeholders
    SP_clean = np.where(mask, 35, salt_flat)     # Practical salinity placeholder
    PT_clean = np.where(mask, 0, theta_flat)     # Potential temp placeholder

    # GSW calculations
    SA = gsw.SA_from_SP(SP_clean, pressure_rep, lon_rep, lat_rep)  # Absolute Salinity
    CT = gsw.CT_from_pt(SA, PT_clean)                               # Conservative Temp

    # Restore original invalid values where mask was True
    SA[mask] = salt_flat[mask]
    CT[mask] = theta_flat[mask]

    # Final assignment
    salt_flat = np.copy(SA)
    theta_flat = np.copy(CT)
    
    print(
    'temp:', np.min(theta_flat), np.max(theta_flat),
    'salt:', np.min(salt_flat), np.max(salt_flat),
    'lat:', np.min(lat_rep), np.max(lat_rep),
    'lon_sin:', np.min(lon_rep_sin), np.max(lon_rep_sin),
    'lon_cos:', np.min(lon_rep_cos), np.max(lon_rep_cos),
    'pressure:', np.min(pressure_rep), np.max(pressure_rep),
    'hab:',np.min(hab_flat),np.max(hab_flat)
    )
    
    #Size of this data is (N,feature_dim)
    input_data = np.column_stack((theta_flat, salt_flat, lat_rep, lon_rep_sin,lon_rep_cos, pressure_rep,hab_flat))
    #input_data = np.column_stack((theta_flat,lat_rep, lon_rep, Z_rep))
    Ninp = input_data.shape[0]
    batch_size = Ninp//num_batches
    splits = np.split(input_data, num_batches, axis = 0)
    print('input shape is',input_data.shape)
    #print(splits[0].shape)
    #sys.exit()
    #splits = np.linspace(0, len(input_data), batch_size)
    #slice()
    meanl, uncl = [], []
    
    for data in splits:
        print('data shape is',data.shape)
        mean, unc = ensemble_inference(models, data)
        print('mean shape:', mean.shape)
        meanl.append(mean)
        uncl.append(unc)
    mean = np.concatenate(meanl, axis = 0)
    
    
    unc = np.concatenate(uncl, axis = 0)
    
    # Write water mass predictions into ofile
    with Dataset(ofile, 'w') as nc_out:
        # Copy dimensions from original NetCDF file but limit longitude
        nc_out.createDimension('time', None)
        nc_out.createDimension('k', len(Z))
        nc_out.createDimension('j', len(latitude))
        nc_out.createDimension('i', len(longitude))
        #First write temperature and salinity into output file
        var = nc_out.createVariable('theta', 'float32', ('time', 'k', 'j', 'i'))
        reshaped_data = theta_flat.reshape(np.squeeze(theta).shape)
        var[:] = reshaped_data[None, :,:,:]
        
                
        var_lat = nc_out.createVariable('latitude','float32',('j'))
        var_lat[:] = latitude
        var_lon = nc_out.createVariable('longitude','float32',('i'))
        var_lon[:] = longitude
        var_depth = nc_out.createVariable('depth','float32',('k'))
        var_depth[:] = Z
        
        var = nc_out.createVariable('salt', 'float32', ('time', 'k', 'j', 'i'))
        reshaped_data = salt_flat.reshape(np.squeeze(theta).shape)
        #Create a land mask 
        mask = reshaped_data*0
        mask[reshaped_data>0] = 1
        var[:] = reshaped_data[None, :,:,:]
        # Create variables and write predictions
        for io, outvar in enumerate(outvars):
            var = nc_out.createVariable(outvar, 'float32', ('time', 'k', 'j', 'i'))
            reshaped_data = mean[:, io].reshape(np.squeeze(theta).shape)
            var[:] = reshaped_data[None, :,:,:]*mask[None,:,:,:]
        

    # Write water mass uncertainties into ofile_unc
    with Dataset(ofile_unc, 'w') as nc_out:
        # Copy dimensions from original NetCDF file but limit longitude
        nc_out.createDimension('time', None)
        nc_out.createDimension('k', len(Z))
        nc_out.createDimension('j', len(latitude))
        nc_out.createDimension('i', len(longitude))
   
        
        # Create variables and write predictions
        for io, outvar in enumerate(outvars):
            var = nc_out.createVariable(outvar, 'float32', ('time', 'k', 'j', 'i'))
            reshaped_data = unc[:, io].reshape(np.squeeze(theta).shape)
            var[:] = reshaped_data[None, :,:,:]


##########################MAIN################################################
##########All Modifications to be made here###################################
##########################MAIN################################################

if __name__ == "__main__":
    root_dir =  sys.argv[1]
    pred_dir = sys.argv[2]
    if pred_dir == 'same':
        pred_dir = None
    train = sys.argv[3].lower()=='true'
    try:
        depth = int(sys.argv[4])
    except:
        depth = 16
    if train:
        print('TRAINING')
         
        #your csv here
        csv = ''
        
        df = pd.read_csv(csv).dropna().copy()
        invars1 = ['conservative_temperature', 'abs_salinity', 'latitude', 'longitude_sin','longitude_cos', 'pressure','hab']
        #invars1 = ['Potential_temperature', 'Absolute_salinity', 'Latitude', 'Longitude', 'Pressure']
        #invars1 = ['Potential_temperature','Latitude', 'Longitude', 'Pressure']
        invars2 = ['Oxygen', 'Silicate',  'Phosphate', 'Nitrate']
        invars = invars1#  + invars2
        outvars = ['CW', 'AAIW', 'SAIW', 'uNADW', 'lNADW', 'CDW', 'AABW']
        models, scores =  randomized_kfold_training(df, invars, outvars, n_splits=5, model_depth=depth, checkpoint_path="./models")
    else:
        models = glob.glob(f'./models/RF_depth{depth}_fold*'); 
        outvars = ['CW', 'AAIW', 'SAIW', 'uNADW', 'lNADW', 'CDW', 'AABW']

    files, predfiles, uncfiles = fileIO(root_dir, pred_dir)
    
    
    for f, predf, uncf in zip(files, predfiles, uncfiles):
        print('WRITING',f)
        predict_and_write(f, predf, uncf)
    '''    
    for f, predf, uncf in islice(zip(files, predfiles, uncfiles), 2):
        print('WRITING', f)
        predict_and_write(f, predf, uncf)
    '''
        
    print('FINISHED!')
