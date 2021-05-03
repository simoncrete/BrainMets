import json
import os
import nibabel as nib
import csv

from operator import itemgetter

# PATH TO PREPROCESSED DATA
raw_data_path = '/home/lab/nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BrainMets'
pixdim_ind = [1,2,3] # Indexes at which the voxel size [x,y,z] is stored


# PATH TO JSON FILE
with open('/home/lab/nnUNet_data/RESULTS_FOLDER/nnUNet/3d_fullres/Task500_BrainMets/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json') as file:
  data = json.load(file)

with open('json_parsed.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Case Number', 'Dice Score', 'Voxel Size-X', 'Voxel Size-Y', 'Voxel Size-Z'])

    for img in data['results']['all']: 
        # Get dice score on image
        dice = img['1']['Dice']

        # Get nifti data on image
        img_filename = (os.path.basename(img['reference']).split('.'))[0]
        img_ni = nib.load(raw_data_path + '/imagesTr/' + img_filename + '_0000.nii.gz')
        label_ni = nib.load(raw_data_path + '/labelsTr/' + img_filename + '.nii.gz')

        voxel_size = itemgetter(*pixdim_ind)(img_ni.header["pixdim"])

        # Get tumor dimensions
        # tumor_size = 

        # Get case number corresponding to image
        case_number = img_filename.split('_')[1]

        # Write to csv file
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow([case_number, dice, voxel_size[0], voxel_size[1], voxel_size[2]])

