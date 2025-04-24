import unittest
import warnings
import torch
from model import xintra
from config import GlobalConfig as Config
import os
import torch.nn.functional as F
from interfuser.timm.models.interfuser import Interfuser


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
        self.model = xintra(
            self.config, self.device).float().to(self.device)
        self.interfuser_model = Interfuser(
            enc_depth=6,
            dec_depth=6,
            embed_dim=256,
            rgb_backbone_name="r50",
            lidar_backbone_name="r18",
            waypoints_pred_head="gru",
            use_different_backbone=True,
        ).float().to(self.device)
        self.w = self.config.input_resolution
        self.h = self.w

    def test_interfuser(self):
        batch_size = 1
        rgbs = torch.randn(batch_size, 3, self.h, self.w).to(
            self.device, dtype=torch.float)
        seg_image = torch.randn(batch_size, 3, self.h, self.w).to(
            self.device, dtype=torch.float)
        sdc = torch.randn(batch_size, 3, self.h, self.w).to(
            self.device, dtype=torch.float)
        target_point = torch.randn(batch_size, 2).to(
            self.device, dtype=torch.float)
        velo_in = torch.randn(1).to(self.device, dtype=torch.float)
        x = {
            'rgb': rgbs,
            'rgb_center': rgbs,
            'target_point': target_point,
            'seg_image': seg_image,
            'sdc': sdc,
            'measurements': velo_in
        }
        hx, pred_wp, red_light, stop_sign = self.interfuser_model(
            x)
        # print("red_light", red_light.shape)
        # assert segs_f.shape == (
        #     batch_size, 23, self.h, self.w)
        # assert sdcs.shape == (
        #     batch_size, 23, self.h, self.w)
        assert pred_wp.shape == (
            batch_size, self.config.pred_len, 2)
        # assert steering.shape == (1, )
        # assert throttle.shape == (1,)
        # assert brake.shape == (1,)
        assert red_light.shape == (1,)
        assert stop_sign.shape == (1,)
        # assert len(segs_f) == self.config.seq_len
        # is contigous
        # for seg in segs_f:
        #     assert seg.is_contiguous()
        assert pred_wp.shape == (
            batch_size, self.config.pred_len, 2)
        # assert isinstance(steering, torch.Tensor)
        # assert isinstance(throttle, torch.Tensor)
        # assert len(sdcs) == self.config.seq_len

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
        # params = list(
        #     filter(lambda p: p.requires_grad, self.model.parameters()))
        # for idx, param in enumerate(params):
        #     print(
        #         f"Index: {idx}, Shape: {param.shape}, Name: {param.name if hasattr(param, 'name') else 'Unnamed'}")

        assert segs_f.shape == (
            batch_size, 23, self.h, self.w)
        assert sdcs.shape == (
            batch_size, 23, self.h, self.w)
        assert pred_wp.shape == (
            batch_size, self.config.pred_len, 2)
        assert steering.shape == (1, )
        assert throttle.shape == (1,)
        assert brake.shape == (1,)
        assert red_light.shape == (1,)
        assert stop_sign.shape == (1,)
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
