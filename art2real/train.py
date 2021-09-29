import time

from models.fcn import AddaDataLoader
from models.feature_loss import Feature_loss, get_model
from models.pixel_adapt import train as train_pixel
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # # can add self.get_model from torch try that
    # net = get_model(model, num_cls=num_cls, pretrained=True, weights_init=weights_init,
    #                 output_last_ft=discrim_feat)

    # loader = AddaDataLoader(net.transform, dataset, datadir, downscale,
    #                         crop_size=crop_size, half_crop=half_crop,
    #                         batch_size=batch, shuffle=True, num_workers=2)

    net = get_model(model, num_cls=2, pretrained=True, weights_init=True, output_last_ft=False)
    # loader = AddaDataLoader(net.transform, dataset, datadir, 1,
    #                         None, None, 1, True, 2)
    loader = AddaDataLoader(net.transform, dataset, 1, False, 2)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        # for i, data in enumerate(dataset):  # inner loop within one epoch
        for i, data, im_s, im_t, label_s, label_t in enumerate(dataset), loader:  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # train_pixel(src, datadir, model, num_cls,
            #              outdir=outdir, num_epoch=src_num_epoch, batch=batch,
            #              lr=src_lr, betas=betas, weight_decay=weight_decay)
            # train_pixel(model, num_cls=10, batch=128, lr=1e-4, betas=(0.9, 0.999), weight_decay=0, epoch=epoch)
            Feature_loss(1e-5, 0.99, True, 19, False, 1, 0.1, False, None, True, False, im_s, im_t, label_s, label_t, net)
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # for im_s, im_t, label_s, label_t in loader:
        #     Feature_loss(im_s, im_t, label_s, label_t)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.