import torch
import torch.nn as nn


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()

#         # self.encoder = nn.Sequential(
#         #     nn.Conv2d(3, 128, 5, stride=2, padding=2),
#         #     nn.LeakyReLU(0.1),  # in_place = True or False

#         #     nn.Conv2d(128, 256, 5, stride=2, padding=2),
#         #     nn.LeakyReLU(0.1),  # in_place = True or False

#         #     nn.Conv2d(256, 512, 5, stride=2, padding=2),
#         #     nn.LeakyReLU(0.1),  # in_place = True or False

#         #     nn.Conv2d(512, 1024, 5, stride=2, padding=2),
#         #     nn.LeakyReLU(0.1),  # in_place = True or False

#         #     nn.Linear(1024*16*16, 512),

#         #     nn.Linear(512, 8*8*512),
#         #     nn.Conv2d(512, 512*4, 3, stride=1, padding=1),
#         #     nn.LeakyReLU(0.1),
#         #     nn.PixelShuffle(2)
#         # )
#         # self.decoder = nn.Sequential(
#         #     nn.Conv2d(512, 512*4, 3, stride=1, padding=1),
#         #     nn.LeakyReLU(0.1),
#         #     nn.PixelShuffle(2),
#         #     nn.Conv2d(512, 256*4, 3, stride=1, padding=1),
#         #     nn.LeakyReLU(0.1),
#         #     nn.PixelShuffle(2),
#         #     nn.Conv2d(256, 128*4, 3, stride=1, padding=1),
#         #     nn.LeakyReLU(0.1),
#         #     nn.PixelShuffle(2),
#         #     nn.Conv2d(128, 3, 5, padding=2)
#         # )

#         self.downscale_128 = nn.Sequential(
#             nn.Conv2d(3, 128, 5, stride=2, padding=2),
#             nn.LeakyReLU(0.1)  # in_place = True or False
#         )
#         self.downscale_256 = nn.Sequential(
#             nn.Conv2d(128, 256, 5, stride=2, padding=2),
#             nn.LeakyReLU(0.1)  # in_place = True or False
#         )
#         self.downscale_512 = nn.Sequential(
#             nn.Conv2d(256, 512, 5, stride=2, padding=2),
#             nn.LeakyReLU(0.1)  # in_place = True or False
#         )
#         self.downscale_1024 = nn.Sequential(
#             nn.Conv2d(512, 1024, 5, stride=2, padding=2),
#             nn.LeakyReLU(0.1)  # in_place = True or False
#         )

#         self.dense_512 = nn.Linear(1024*16*16, 512)
#         self.dense_8_8_512 = nn.Linear(512, 8*8*512)
#         self.upscale = nn.Sequential(
#             # nn.Conv2d(512, 512, 3, stride=1, padding=1),
#             # nn.LeakyReLU(0.1),
#             # # nn.PixelShuffle(2)
#             nn.ConvTranspose2d(512, 512, 3, stride=2, output_padding=1,
#                                padding=1),  # b, 512, 16, 16
#         )

#         self.upscale_512 = nn.Sequential(
#             # nn.Conv2d(512, 512, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.1),
#             # nn.PixelShuffle(2)
#             nn.ConvTranspose2d(512, 512, 3, stride=2, output_padding=1,
#                                padding=1),  # b, 512, 32, 32
#         )

#         self.upscale_256 = nn.Sequential(
#             # nn.Conv2d(512, 256, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.1),
#             # nn.PixelShuffle(2)
#             nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1,
#                                padding=1),  # b, 256, 64,64
#         )
#         self.upscale_128 = nn.Sequential(
#             # nn.Conv2d(256, 128, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.1),
#             # nn.PixelShuffle(2)
#             nn.ConvTranspose2d(256, 64, 3, stride=2, output_padding=1,
#                                padding=1),  # b, 64, 128, 128
#         )

#         self.conv2d = nn.Conv2d(64, 3, 5, padding=2)
#         self.out_layer = nn.Sigmoid()

#     def forward(self, x):
#         # x = self.encoder(x)
#         # x = self.decoder(x)
#         x = self.downscale_128(x)
#         x = self.downscale_256(x)
#         x = self.downscale_512(x)
#         x = self.downscale_1024(x)
#         x = x.view(x.size(0), -1)
#         x = self.dense_512(x)
#         x = self.dense_8_8_512(x)
#         x = x.view(x.size(0), 512, 8, 8)
#         x = self.upscale(x)
#         x = self.upscale_512(x)
#         x = self.upscale_256(x)
#         x = self.upscale_128(x)
#         x = self.conv2d(x)
#         x = self.out_layer(x)
#         return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.downscale_128 = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        self.downscale_256 = nn.Sequential(
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        self.downscale_512 = nn.Sequential(
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        self.downscale_1024 = nn.Sequential(
            nn.Conv2d(512, 1024, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )

        self.dense_512 = nn.Linear(1024*16*16, 512)
        self.dense_8_8_512 = nn.Linear(512, 8*8*512)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, stride=2, output_padding=1,
                               padding=1),  # b, 512, 16, 16
        )

    def forward(self, x):
        x = self.downscale_128(x)
        x = self.downscale_256(x)
        x = self.downscale_512(x)
        x = self.downscale_1024(x)
        x = x.view(x.size(0), -1)
        x = self.dense_512(x)
        x = self.dense_8_8_512(x)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.upscale(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upscale_512 = nn.Sequential(
            # nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            # nn.PixelShuffle(2)
            nn.ConvTranspose2d(512, 512, 3, stride=2, output_padding=1,
                               padding=1),  # b, 512, 32, 32
        )

        self.upscale_256 = nn.Sequential(
            # nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            # nn.PixelShuffle(2)
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1,
                               padding=1),  # b, 256, 64,64
        )
        self.upscale_128 = nn.Sequential(
            # nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            # nn.PixelShuffle(2)
            nn.ConvTranspose2d(256, 64, 3, stride=2, output_padding=1,
                               padding=1),  # b, 64, 128, 128
        )

        self.conv2d = nn.Conv2d(64, 3, 5, padding=2)
        self.out_layer = nn.Sigmoid()

        self.upscale_512_mask = nn.Sequential(
            # nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            # nn.PixelShuffle(2)
            nn.ConvTranspose2d(512, 512, 3, stride=2, output_padding=1,
                               padding=1),  # b, 512, 32, 32
        )

        self.upscale_256_mask = nn.Sequential(
            # nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            # nn.PixelShuffle(2)
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1,
                               padding=1),  # b, 256, 64,64
        )
        self.upscale_128_mask = nn.Sequential(
            # nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            # nn.PixelShuffle(2)
            nn.ConvTranspose2d(256, 64, 3, stride=2, output_padding=1,
                               padding=1),  # b, 64, 128, 128
        )

        self.conv2d_mask = nn.Conv2d(64, 1, 5, padding=2)
        self.out_layer_mask = nn.Sigmoid()

    def forward(self, x):
        input_ = x
        out = self.upscale_512(input_)
        out = self.upscale_256(out)
        out = self.upscale_128(out)
        out = self.conv2d(out)
        out = self.out_layer(out)

        out_mask = self.upscale_512_mask(input_)
        out_mask = self.upscale_256_mask(out_mask)
        out_mask = self.upscale_128_mask(out_mask)
        out_mask = self.conv2d_mask(out_mask)
        out_mask = self.out_layer_mask(out_mask)
        return out, out_mask
