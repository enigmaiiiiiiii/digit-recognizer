from DataTransformer import predict_data
import torch
import pandas as pd

predict_loader = predict_data(batch_size=50)

model = torch.load('Net.pth')
model.eval()

submission_tensor = torch.LongTensor()
for (predict_data,) in predict_loader:
    output = model(predict_data)
    pred = output.data.max(1, keepdim=True)[1]
    submission_tensor = torch.cat((submission_tensor, pred), dim=0)

submission_pred = submission_tensor.detach().numpy().squeeze()

sub = pd.read_csv(r'D:/JupyterProject/digit-recognizer/sample_submission.csv')
sub['Label'] = submission_pred
sub.to_csv(r'D:/JupyterProject/digit-recognizer/sample_submission2.csv')
