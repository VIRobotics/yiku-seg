import time

from yiku_ad.data import create_dataset
from yiku_ad.utils.init import create_model
from yiku_ad.utils.options import TrainOptionParser

def train(opt):
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"Training size is = {dataset_size}")
    model = create_model(opt)  # create model (AE, AAE)
    model.setup(opt)  # set model : if mode is 'train', define schedulers and if mode is 'test', load saved networks
    total_iters = 0
    loss_name = model.loss_name  # loss name for naming
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # start epoch time
        model.update_learning_rate(epoch)  # update learning rate change with schedulers
        epoch_iters = 0

        for i, data in enumerate(dataset):  # dataset loop
            iter_start_time = time.time()  # start iter time
            model.set_input(data)  # unpacking input data for processing
            model.train()  # start model train
            total_iters += 1
            epoch_iters += 1
        if epoch % opt.print_epoch_freq == 0:  # model loss, time print frequency
            losses = model.get_current_losses(*loss_name)
            epoch_time = time.time() - epoch_start_time
            message = f"epoch : {epoch} | total_iters : {total_iters} | epoch_time:{epoch_time:.3f}"
            for k, v in losses.items():
                message += f" | {k}:{v}"
            print(message)
        if epoch % opt.save_epoch_freq == 0:  # save model frequency
            print(
                "saving the latest model (epoch %d, total_iters %d)"
                % (epoch, total_iters)
            )
            model.save_networks()

if __name__ == '__main__':
    opt=TrainOptionParser().opt
    train(opt)
