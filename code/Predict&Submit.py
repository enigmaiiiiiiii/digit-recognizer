from DataTransformer import data_factory
import torch
import pandas as pd

train_data_path = r'D:/JupyterProject/digit-recognizer/train.csv'
test_data_path = r'D:/JupyterProject/digit-recognizer/test.csv'
digit_data = data_factory(train_data_path,test_data_path)

predict_loader = digit_data.predict_dataloader(batch_size=100)

model = torch.load('.\\TrainedModels\\TrainWithAllData.pth')
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
