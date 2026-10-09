# 拓扑霍尔效应（THE）数据分析工具

基于 Python + Streamlit 的多温度输运（ETO）/ 磁性（VSM）数据分析应用，用于提取拓扑霍尔电阻率 ρxy_T(H)。

## 目录结构

```
deploy/
├── app.py                  # Streamlit 应用入口
├── requirements.txt        # Python 依赖
├── .streamlit/
│   └── config.toml         # 主题与服务器配置
└── README.md
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 将本目录内容推送到一个 GitHub 仓库根目录。
2. 打开 <https://share.streamlit.io>，用 GitHub 账号登录。
3. 点击 **New app**，选择仓库、分支（main）、入口文件 `app.py`。
4. 点击 **Deploy**。

## 部署到 Hugging Face Spaces

1. 在 <https://huggingface.co/spaces> 新建 Space，SDK 选择 **Streamlit**。
2. 上传 `app.py`、`requirements.txt`、`.streamlit/config.toml`。
3. 等待自动构建完成。

## 说明

- 应用通过 `st.file_uploader` 上传数据，无需数据库或持久化存储。
- 会话中的分析结果保存在内存中，长期保存请使用应用内的“导出会话 JSON”。
