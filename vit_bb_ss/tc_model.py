import unittest
import torch
from model import vit_bb_ss
from config import GlobalConfig as Config
import os


class TestVIT_bb_ss(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.device = torch.device("cuda:0")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # visible_gpu #"0" "1" "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.model = vit_bb_ss(
            self.config, self.device).float().to(self.device)
        self.w = self.config.input_resolution
        self.h = self.w

    def test_init(self):
        self.assertIsInstance(self.model, vit_bb_ss)
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.gpu_device, self.device)

    def test_rgb_encoder(self):
        # Assuming RGB_Encoder is a method of xr14
        input_tensor = torch.randn(
            self.config.batch_size, 3, self.h, self.w).to(
            self.device, dtype=torch.float)  # Example input
        output = self.model.RGB_encoder(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
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

    def test_sc_encoder(self):
        # Assuming SC_encoder is a method of xr14
        input_tensor = torch.randn(
            1, 23, self.h, self.w).to(
            self.device, dtype=torch.float)  # Example input
        output = self.model.SC_encoder(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        # Add more assertions based on expected output shape and values


if __name__ == '__main__':
    unittest.main()
