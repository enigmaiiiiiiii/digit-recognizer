from DataTransformer import predict_data

import  torch
predict_loader = predict_data(batch_size=100)

model = torch.load('Net.pth')
model.eval()

submission_tensor = torch.Tensor()
for (predict_data,) in predict_loader:
    output = model(predict_data)
    submission_tensor = torch.cat(submission_tensor,output)


submission_pred = submission_tensor.numpy()



