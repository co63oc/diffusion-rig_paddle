import torch
import paddle

model_path="data/deca_model.tar"
paddle_model_path="data/paddle_deca_model.tar"
checkpoint = torch.load(model_path)

paddle_checkpoint = {}
for index_name in ["E_flame", "E_detail", "D_detail"]:
    paddle_checkpoint[index_name] = {}
    for k, v in checkpoint[index_name].items():
        paddle_checkpoint[index_name][k] = paddle.to_tensor(v.cpu().numpy())

paddle.save(paddle_checkpoint, paddle_model_path)
print("Convert finish.")