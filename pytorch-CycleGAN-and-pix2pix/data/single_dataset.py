import os
from PIL import Image

import glob

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # Yechan added ->
        if opt.datarootA:
            print(f'For domain "A", the given opt.dataroot, {opt.dataroot} will be ignored! '
                  f'Instead, the given opt.datarootA, {opt.datarootA} will be used.')
            self.dir_A = opt.datarootA
        else:
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))

        if opt.datarootB and not opt.datarootA:
            print('Please use variable named "datarootA"!!!')
            exit(-1)

        if opt.filterA:
            self.A_paths = glob.glob(f'{opt.datarootA}/{opt.filterA}')
            if len(self.A_paths) == 0:
                raise RuntimeError(f"Found 0 images in subfolders of: [opt.datarootA/opt.filterA] " +
                                   f'{opt.datarootA}/{opt.filterA}' + "\n")
            else:
                self.A_paths = sorted(self.A_paths[:min(opt.max_dataset_size, len(self.A_paths))])
        else:
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))


        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
