from mmseg.apis import init_model
import torch

# Load the config for SETR
# Adjust path if needed
config_file = './mmsegmentation/configs/swin/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py'
checkpoint_file = './upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth'

# Load model config
model = init_model(config_file, checkpoint_file)
encoder = model.backbone

# Test the encoder with a random input tensor


def to_device(data, device):
    """Move tensor(s) to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
# model = encoder.to(device)

input_tensor = torch.randn(1, 3, 768, 768)
# Move the input tensor to the device
input_tensor = to_device(input_tensor, device)

# Example input (batch_size, channels, height, width)

features = encoder(input_tensor)
print([f.shape for f in features])  # Print shapes of feature maps
print(model)
