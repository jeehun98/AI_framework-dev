import torch
import torch.nn as nn
import torch.optim as optim

# XOR 입력과 정답
x = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

# 모델 정의
class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 모델, 손실 함수, 옵티마이저
model = XORModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습 루프
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 예측 결과
preds = model(x).detach().numpy()
print("\n🔍 XOR 예측 결과:")
print("====================================")
print("  입력         |  정답  |  예측값")
print("---------------|--------|----------")
for i in range(4):
    print(f"  {x[i].tolist()}  |   {y[i].item():.1f}   |  {preds[i][0]:.4f}")
print("====================================")
