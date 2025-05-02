import os
import shutil

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import tensor2im
from PIL import Image


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        result = tensor2im(visuals['fake'])
        result_img = Image.fromarray(result)

        fname = os.path.basename(img_path[0])
        fname_split_list = fname.split('.')[0].split('_')
        scene_num, angle = fname_split_list[1], fname_split_list[2]

        os.makedirs(f'{opt.results_dir}/{scene_num}/{angle}', exist_ok=True)

        # label migrate
        csv_label_path = f'{opt.datarootA}/{scene_num}/{angle}/ANNOTATION-EO_{scene_num}_{angle}.csv'
        shutil.copyfile(csv_label_path,
                        f'{opt.results_dir}/{scene_num}/{angle}/ANNOTATION-EO_{scene_num}_{angle}.csv')

        # translated image save
        result_img.save(f'{opt.results_dir}/{scene_num}/{angle}/{fname}')

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

