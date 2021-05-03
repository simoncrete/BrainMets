import nibabel as nib
import nibabel.processing as nibproc
import os
import SimpleITK as sitk
import radiomics
import csv

data_path = "/home/lab/Task500_BrainMets/imagesTr_copy/" #FSRTCASE_200_0000.nii" # Directory holding NIFTI files images
data_path_mask = "/home/lab/Task500_BrainMets/labelsTr_copy/" #FSRTCASE_200.nii # Directory holding NIFTI files masks
row_flag = 1

with open('radiomicfeatures.csv', mode='w') as csv_file:


    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in sorted(files):
            img_name = os.path.join(root,name)        
            mask_name = os.path.join(data_path_mask,name.rsplit('_',1)[0]+'.nii')
            patient_name = name.split('_')[0] + name.split('_')[1]

            # Read the .nii image containing the volume with SimpleITK:
            raw = sitk.ReadImage(img_name)
            mask = sitk.ReadImage(mask_name)

            # Load raw image into variable "raw"
            zsc = sitk.GetArrayFromImage(raw)
            zsc = (zsc - zsc.mean())/zsc.std()
            zsc = sitk.GetImageFromArray(zsc)
            zsc.CopyInformation(raw)
            # Save zsc

            #Setting up the pyradiomics extractor object:
            shift_val = 0.8
            lower, upper = (-0.587027480803037, 5.234361187485263)
            bin_width = (upper-lower)/50
            mean_spacing = (0.9,)*3
            voxelArrayShift=shift_val    
            parameters = dict(
                    preCrop=True, # Speed up and reduce memory
                    binWidth=bin_width, # use computed binwidth above # TODO Determine why this causes linalg error
                    correctMask=False,
                    voxelArrayShift=shift_val,
                    resampledPixelSpacing=(mean_spacing[0],)*3
            )

            extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**parameters)

            # Now for each image, you can extract features via
            features_dict = extractor.execute(zsc, mask)
            usefulfeatures_dict = features_dict.copy()

            for k in features_dict.keys():
                if k.startswith('diagnostics') or isinstance(features_dict[k],list): 
                    usefulfeatures_dict.pop(k)

            if row_flag:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                column_names = ['patient']
                column_names.extend(list(usefulfeatures_dict.keys()))
                csv_writer.writerow(column_names)

            row_flag = 0

            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            column_values = [patient_name]
            column_values.extend(list(usefulfeatures_dict.values()))
            csv_writer.writerow(column_values)
            
