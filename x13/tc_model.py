import unittest
import torch
from model import x13
from config import GlobalConfig as Config


class TestXR14(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.device = torch.device('cpu')
        self.model = x13(self.config, self.device)
        self.w = self.config.input_resolution
        self.h = self.w

    def test_init(self):
        self.assertIsInstance(self.model, x13)
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.gpu_device, self.device)

    def test_rgb_encoder(self):
        # Assuming RGB_Encoder is a method of xr14
        input_tensor = torch.randn(
            self.config.batch_size, 3, self.h, self.w)  # Example input
        output = self.model.RGB_encoder(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        # Add more assertions based on expected output shape and values

    def test_forward(self):
        batch_size = 1
        rgbs = torch.randn(batch_size, 3, self.h, self.w).to(self.device)
        depth = torch.randn(batch_size, self.h, self.w).to(self.device)
        target_point = torch.randn(batch_size, 2).to(self.device)
        velo_in = torch.randn(1).to(self.device)

        segs_f, pred_wp, steering, throttle, brake, red_light, stop_sign, sdcs = self.model(
            rgbs, depth, target_point, velo_in)
        # params = list(
        #     filter(lambda p: p.requires_grad, self.model.parameters()))
        # for idx, param in enumerate(params):
        #     print(
        #         f"Index: {idx}, Shape: {param.shape}, Name: {param.name if hasattr(param, 'name') else 'Unnamed'}")
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
            1, 23, self.h, self.w)  # Example input
        output = self.model.SC_encoder(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        # Add more assertions based on expected output shape and values


if __name__ == '__main__':
    unittest.main()
