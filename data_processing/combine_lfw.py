import torch
'''This script is designed to merge two .pt files. This file was used specifically to merge the augmented LFW datasets into a single file for training.
It was created with the assistance of AI and is currently hard coded for the use case of merging the two LFW datasets. 
It can be easily modified to merge any two .pt files with the same structure.'''
def merge_pt_files(file1_path, file2_path, output_path):
    print("Loading first file")
    data1 = torch.load(file1_path)
    images1, labels1 = data1['images'], data1['labels']
    
    print("Loading second file")
    data2 = torch.load(file2_path)
    images2, labels2 = data2['images'], data2['labels']

    # ensure channel dimension is correct for PyTorch 
    if images1.shape[-1] == 3:
        images1 = images1.permute(0, 3, 1, 2)
    if images2.shape[-1] == 3:
        images2 = images2.permute(0, 3, 1, 2)
    
    # Adjust labels in the second file to avoid overlap
    offset = labels1.max().item() + 1
    labels2 = labels2 + offset
    
    print(f"Concatenating tensors (Total samples: {len(images1) + len(images2)})...")
    combined_images = torch.cat((images1, images2), dim=0)
    combined_labels = torch.cat((labels1, labels2), dim=0)
    
    # Clear memory of old variables to avoid OOM issues
    del images1, images2, data1, data2
    
    print(f"Saving to {output_path}")
    torch.save({'images': combined_images, 'labels': combined_labels}, output_path)
    print("Complete!")
#input file to merge listed as first file, second file and output file name as third argument
merge_pt_files('lfw_full_norm_augmented_9.pt', 'lfw_full_norm_aug3_10.pt', 'norm_full_lfw.pt')
