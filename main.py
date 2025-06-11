import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# --- モデル定義 ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], -1)
        h = self.fc1(x_reshaped)
        z = torch.sigmoid(h)
        y_hat = self.fc2(z)
        return y_hat

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
loaded_model = SimpleMLP().to(device)
loaded_model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device, weights_only=True))
loaded_model.eval()

st.title("手書き数字認識 (SimpleMLP)")
st.write("キャンバスに数字を書いて予測を実行します。")

# --- 描画キャンバス ---
canvas_result = st_canvas(
    fill_color="#000000",  # 塗りつぶし色
    stroke_width=10,       # ペンの太さ
    stroke_color="#FFFFFF",  # 白で描画
    background_color="#000000",  # 黒背景
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# --- 予測処理 ---
if canvas_result.image_data is not None:
    # キャンバス画像 (numpy array) を取得
    img = canvas_result.image_data

    # 画像前処理
    img = Image.fromarray((img[:, :, 0]).astype('uint8'))  # グレースケール化
    img = img.resize((28, 28))
    img = ImageOps.invert(img)  # 白黒反転（白背景・黒文字に変換）

    # Streamlitで表示
    st.image(img, caption="前処理後の画像 (28x28)", width=150)

    # 正規化とテンソル化
    image_np = np.array(img) / 255.0
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 予測
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = loaded_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    st.write("**予測されたクラス:**", predicted_class)
    st.write("**各クラスの確率:**", probabilities.cpu().numpy())

    # 可視化
    fig, ax = plt.subplots()
    ax.imshow(image_tensor.squeeze().cpu().numpy(), cmap="gray")
    ax.set_title(f"Prediction: {predicted_class}")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("キャンバスに数字を書いてください。")
