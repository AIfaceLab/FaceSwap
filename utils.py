from pathlib import Path
import cv2
import numpy as np
import os
import torch
import math
from matplotlib import pyplot as plt
import multiprocessing
import time
import pickle


def get_image_list(path2dir):
    '''
    get the image paths from the path2dir
    '''

    dir_path = Path(path2dir)
    if not dir_path.exists():
        raise ValueError("the path to image dir doesn't exist")
    images = []
    for root, directory, filenames in os.walk(dir_path):
        for file in filenames:
            if(Path(file).suffix == '.jpg' or Path(file).suffix == '.png'):
                images.append(Path(os.path.join(root, file)))
    return images


def get_label_images(image_batch, size=(128, 128)):
    label = []
    for i in range(image_batch.shape[0]):
        b, g, r = image_batch[i][0], image_batch[i][1], image_batch[i][2]
        image = np.stack((b, g, r), axis=-1)
        image = cv2.resize(image, size)
        b, g, r = cv2.split(image)
        b, g, r = b[np.newaxis][:], g[np.newaxis][:], r[np.newaxis][:]
        out = np.vstack((b, g, r))
        label.append(out)
    return np.array(label, dtype=np.float32)


def get_label_mask(image_batch, size=(128, 128)):
    label = []
    for i in range(image_batch.shape[0]):
        mask = cv2.resize(image_batch[i][0], size)
        mask = mask[np.newaxis][:]
        label.append(mask)
    return np.array(label, dtype=np.float32)


def visualize_output(name, batch_image):
    imgs = batch_image.transpose(1, 3)*255
    num_images, h, w = imgs.size()[0:3]
    cols = 2
    rows = math.ceil(num_images/cols)
    window = np.zeros((cols*w, rows*h, 3))

    images = imgs.type(torch.uint8).cpu().numpy()
    for i in range(num_images):
        r = i//cols
        c = i % cols
        img = images[i]
        window[c*w:(c+1)*w, r*h:(r+1)*h, :] = img

    # cv2.imshow(name, cv2.cvtColor(
    #     images[0].astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imshow(name, window.astype(np.uint8))
    cv2.waitKey(1)


class LogModel(object):
    def __init__(self, input_resolution=256, saving_path='./Training_Results/models', model_name="A2B"):
        super().__init__()
        self.saving_path = saving_path
        self.input_resolution = input_resolution
        self.model_name = "model_{}_{}.pth".format(
            self.input_resolution, model_name)
        self.path_model = os.path.join(
            self.saving_path, self.model_name)

    def load_model(self, encoder, decoder_a, decoder_b):
        if not Path(self.path_model).exists():
            print('no model file ,training from the start')
            return 0
        print('loading model...')
        checkpoint = torch.load(self.path_model)
        encoder.load_state_dict(checkpoint['Encoder_state_dict'])
        decoder_a.load_state_dict(checkpoint['DecoderA_state_dict'])
        decoder_b.load_state_dict(checkpoint['DecoderB_state_dict'])
        print('finished loading!')
        return int(checkpoint['epoch'])

    def log_model(self, epoch, encoder, decoder_a, decoder_b):
        print('Saving latest model ,epoch {}'.format(epoch))
        saving_dict = {
            'epoch': epoch,
            'Encoder_state_dict': encoder,
            'DecoderA_state_dict': decoder_a,
            'DecoderB_state_dict': decoder_b
        }
        torch.save(saving_dict, self.path_model)
        print('finished saving')


class LossVisualize(object):
    def __init__(self, title="Loss", xlabel="num_iter", ylabel="loss", saving_path='./Training_Results/Loss_Curve', resolution=256, loss_name="A2B"):
        super().__init__()
        self.input_resolution = resolution
        self.loss_name = "loss_{}_{}.pkl".format(
            self.input_resolution, loss_name)
        self.path_loss = os.path.join(
            saving_path, self.loss_name)
        # ---------------------if exist preview loss then load it-----------------------------------
        if Path(self.path_loss).exists():
            with open(self.path_loss, 'rb') as handle:
                self.loss_dict = pickle.load(handle)
        else:
            self.loss_dict = {}
        # ------------------------------------------------------------------------------------------
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.s2c = multiprocessing.Queue()
        # self.lock = multiprocessing.Lock()
        self.p = multiprocessing.Process(
            target=self.visualize, args=())
        self.p.start()
        # self.handle_loss = open(self.path_loss, 'wb')

    def log(self, loss_dict):
        for loss_name, value in loss_dict.items():
            if loss_name not in self.loss_dict.keys():
                self.loss_dict[loss_name] = []
            self.loss_dict[loss_name].append(value)
        # with self.lock:
        # self.s2c.get()
        self.s2c.put(self.loss_dict)

    def save_loss2file(self):
        with open(self.path_loss, 'wb') as handle:
            pickle.dump(self.loss_dict, handle, protocol=0)
        print('finished dumpping loss file')

    def kill(self):
        self.p.terminate()
        self.p.join()
        print("terminated loss-plot")
        # self.handle_loss.close()

    def visualize(self):
        def getcolor(loss_name):
            if loss_name == 'loss_src':
                return 'r'
            else:
                return 'g'
        plt.figure(1)
        bool_legend = True
        while True:
            # with self.lock:
            if not self.s2c.empty():
                loss_dict = self.s2c.get()
                plt.title(self.title)
                plt.xlabel(self.xlabel)
                plt.ylabel(self.ylabel)
                for loss_name, loss_arr in loss_dict.items():
                    plt.plot(np.arange(len(loss_dict[loss_name])),
                             np.array(loss_dict[loss_name]), getcolor(loss_name), label=loss_name)
                if bool_legend:
                    plt.legend(loc='upper right')
                    bool_legend = False
                plt.pause(0.1)
        self.kill()


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T
