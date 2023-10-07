import torch
import paddle
model = torch.load("s3fd-619a316812.pth")

model_path="s3fd-619a316812.pth"
paddle_model_path="paddle_s3fd-619a316812.pth"
model_path="2DFAN4-11f355bf06.pth.tar"
paddle_model_path="paddle_2DFAN4-11f355bf06.pth.tar"
model_path="3DFAN4-7835d9f11d.pth.tar"
paddle_model_path="paddle_3DFAN4-7835d9f11d.pth.tar"
checkpoint = torch.load(model_path)

paddle_checkpoint = {}
for k, v in checkpoint.items():
    paddle_checkpoint[k] = paddle.to_tensor(v.cpu().numpy())

paddle.save(paddle_checkpoint, paddle_model_path)
print("Convert finish.")
