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
