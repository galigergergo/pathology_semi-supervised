import os
import shutil


classes = ['TE', 'NEC', 'LYM', 'TAS']


def luad_dataset_to_imagenet_format(orig_path, target_path):
    """
    Creates a new version of the LUAD dataset in ImageNet format:
        - root
            - class 1
                - img 1
                - img 2
                ...
            - class 2
                - img 1
                - img 2
                ...
            ...


    Parameters
    ----------
    orig_path : string
        Path to root directory of LUAD-HistoSeg dataset.
    target_path : string
        Path to desired root directory of formatted dataset.
    
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for root_folder in [f.name for f in os.scandir(orig_path) if f.is_dir()]:
        if not os.path.exists(os.path.join(target_path, root_folder)):
            os.makedirs(os.path.join(target_path, root_folder))
        for classs in classes:
            if not os.path.exists(os.path.join(target_path, root_folder, classs)):
                os.makedirs(os.path.join(target_path, root_folder, classs))

        file_names = [fn for fn in os.listdir(os.path.join(orig_path, root_folder)) if fn.endswith('png')]
        if not len(file_names):
            continue    # only do this for the train folder
            file_names = [fn for fn in os.listdir(os.path.join(orig_path, root_folder, 'img')) if fn.endswith('png')]
        for file in file_names:
            file_name = file[:-4]
            idx = file_name[:-10]
            label_str = file_name.split(']')[0].split('[')[-1].split(' ')
            labels = [int(label_str[0]), int(label_str[1]),
                      int(label_str[2]), int(label_str[3])]

            for i in range(len(classes)):
                if labels[i]:
                    shutil.copyfile(os.path.join(orig_path, root_folder, file),
                                    os.path.join(target_path, root_folder, classes[i], idx + '.png'))


orig_path = '../../Datasets/lung_cancer_ANNOTATED/LUAD-HistoSeg'
target_path = '../../Datasets/lung_cancer_ANNOTATED/formatted'
luad_dataset_to_imagenet_format(orig_path, target_path)
