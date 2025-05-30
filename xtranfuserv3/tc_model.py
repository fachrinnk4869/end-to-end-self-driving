import unittest
import torch
from model import x13
from config import GlobalConfig as Config
from transfuser.model import TransFuser
import os


class TestXR14(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.device = torch.device('cuda')
        self.model = x13(self.config, self.device).float().to(self.device)
        self.modelTransfuser = TransFuser(
            self.config, self.device).float().to(self.device)
        state_dict = torch.load(
            os.path.join('transfuser/log', 'best_model.pth'))
        # Print available layers (keys in state_dict)
        # for key in state_dict.keys():
        #     print(key)
        # print(self.modelTransfuser)
        # print(model)

        self.modelTransfuser.load_state_dict(state_dict)
        # self.w = self.config.input_resolution
        self.w = 256
        self.h = self.w

    def test_init(self):
        self.assertIsInstance(self.model, x13)
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.gpu_device, self.device)

    def test_run_transfuser(self):
        # Assuming TransFuser is a method of xr14
        input_tensor = torch.randn(
            self.config.batch_size, 3, self.h, self.w).to(self.device)
        # image_list torch.Size([20, 3, 256, 256])
        image_list = [torch.randn(
            20, 3, self.h, self.w).to(self.device)]
        # lidar_list torch.Size([20, 2, 256, 256])
        lidar_list = [torch.randn(
            20, 2, self.h, self.w).to(self.device)]
        # target_point torch.Size([20, 2])
        target_point = torch.randn(
            20, 2).to(self.device)
        # velocity torch.Size([20])
        velocity = torch.randn(20).to(self.device)

        pred_wp = self.modelTransfuser(
            image_list, lidar_list, target_point, velocity)
        print(pred_wp.shape)

        self.assertIsInstance(pred_wp, torch.Tensor)
        # Add more assertions based on expected output shape and values

    def test_forward(self):
        batch_size = 1
        rgbs = torch.randn(batch_size, 3, self.h, self.w).to(self.device)
        depth = torch.randn(batch_size, self.h, self.w).to(self.device)
        target_point = torch.randn(batch_size, 2).to(self.device)
        velo_in = torch.randn(1).to(self.device)

        segs_f, pred_wp, steering, throttle, brake, red_light, stop_sign, sdcs = self.model(
            rgbs, depth, target_point, velo_in)
        params = list(
            filter(lambda p: p.requires_grad, self.model.parameters()))
        for idx, param in enumerate(params):
            print(
                f"Index: {idx}, Shape: {param.shape}, Name: {param.name if hasattr(param, 'name') else 'Unnamed'}")

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


if __name__ == '__main__':
    unittest.main()
