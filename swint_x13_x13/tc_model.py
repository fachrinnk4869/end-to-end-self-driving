import unittest
import warnings
import torch
from model import swint_x13_x13
from config import GlobalConfig as Config
import os
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class TestVIT_bb_ss(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.device = torch.device("cuda:0")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # visible_gpu #"0" "1" "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.model = swint_x13_x13(
            self.config, self.device).float().to(self.device)
        self.w = self.config.input_resolution
        self.h = self.w

    # def test_init(self):
    #     self.assertIsInstance(self.model, swint_x13_x13)
    #     self.assertEqual(self.model.config, self.config)
    #     self.assertEqual(self.model.gpu_device, self.device)

    # def test_rgb_encoder(self):
    #     # Assuming RGB_Encoder is a method of xr14
    #     input_tensor = torch.randn(
    #         self.config.batch_size, 3, self.h, self.w).to(
    #         self.device, dtype=torch.float)  # Example input
    #     features = self.model.RGB_encoder(input_tensor)

    #     # if isinstance(features, list):
    #     # print(features.shape)
    #     # features = torch.cat(features, dim=1)  # Concatenate feature maps
    #     # Print shapes of the feature maps
    #     # for i, feat in enumerate(features):
    #     #     print(f"Stage {i+1} feature map shape: {feat.shape}")
    #     # features = torch.cat([features[-1], features[-2]], dim=1)
    #     # Combine features from Stage 1 and Stage 4
    #     local_features = features[0]  # Stage 1 (1/4 resolution)
    #     global_features = features[3]  # Stage 4 (1/32 resolution)

    #     # Resize global_features to match local_features
    #     global_features_resized = F.interpolate(
    #         global_features, size=local_features.shape[2:], mode='bilinear', align_corners=False)

    #     # Combine features
    #     features = torch.cat(
    #         [local_features, global_features_resized], dim=1)
    # # features = torch.cat([features[i] for i in [-2, -1]], dim=1)
    #     self.assertIsInstance(features, torch.Tensor)
    # Add more assertions based on expected output shape and values

    def test_forward(self):
        batch_size = 1
        rgbs = torch.randn(batch_size, 3, self.h, self.w).to(
            self.device, dtype=torch.float)
        depth = torch.randn(batch_size, self.h, self.w).to(
            self.device, dtype=torch.float)
        target_point = torch.randn(batch_size, 2).to(
            self.device, dtype=torch.float)
        velo_in = torch.randn(1).to(self.device, dtype=torch.float)

        segs_f, pred_wp, steering, throttle, brake, red_light, stop_sign, sdcs = self.model(
            rgbs, depth, target_point, velo_in)

        assert len(segs_f) == self.config.seq_len
        # is contigous
        for seg in segs_f:
            assert seg.is_contiguous()
        assert pred_wp.shape == (
            batch_size, self.config.pred_len, 2)
        assert isinstance(steering, torch.Tensor)
        assert isinstance(throttle, torch.Tensor)
        # print(sdcs)
        assert len(sdcs) == self.config.seq_len

    # def test_sc_encoder(self):
    #     # Assuming SC_encoder is a method of xr14
    #     input_tensor = torch.randn(
    #         1, 23, self.h, self.w).to(
    #         self.device, dtype=torch.float)  # Example input
    #     output = self.model.SC_encoder(input_tensor)
    #     self.assertIsInstance(output, torch.Tensor)
    # Add more assertions based on expected output shape and values


if __name__ == '__main__':
    unittest.main()
