# Restful API & Ajax

## 提問
- 通則
  - 「結業前」可提問、討論，要把多餘時間和資源，留給當前上課的學員。
- 寫信
	- E-mail: `darren@darreninfo.cc`
	- 信件標題寫上你的**班別和姓名**，或是在哪裡參與我的課程，例如 `[資展 BDSEXX / 臺大計中 / 聯成]` 你的主旨 ○○○。
	- 提問的內容要與本專案有關，**其它課程的部分，去請益原本授課的老師**。
	- **不要把程式碼寄給我**，可能沒時間看，討論儘量以解決問題的方向為主。
	- 不符合以上幾點，將**直接刪除**，敬請見諒。

## 安裝套件
```bash
pip install -r requirements.txt
```

## 作業
- 僅限授課學員。
- 同學之間可以互相討論，但千萬不要抄襲。
- 在 `titanic_restful_project` 專案基礎上，完成以下功能（如果覺得範例不夠完整，可以自行調整或重構）：
  - 新增一個或多個頁面，與機器學習有關。
  - 能夠一鍵將 `titanic` 資料表的資料進行機器學習模型訓練，要調整超參數（參數數量自行選擇），並顯示最佳的超參數資訊。
  - 頁面要能夠觀察/知道模型是否訓練完成，並且將訓練好的模型儲存起來。
  - 能夠讓人輸入資料（或上傳 csv 進行批次處理），預測該資料所代表的乘客是否存活，以及生存機率。
- `80` 分條件
  - 不用給我程式碼。
  - 錄成影片，說明與展示你所完成的功能，以及最後的成果。
  - 提供影片連結給我，可以用 YouTube 或是 Google Drive 的公開連結。
  - 沒有限制影片時間。
- `100` 分條件 (基於 `80` 分條件)
    - 使用 `GitHub` 平台來提交作業，並且將 `github repo 連結` 以及 `影片連結` 連結寄給我。
      - [程式與網頁開發者必備技能！Git 和 GitHub 零基礎快速上手，輕鬆掌握版本控制的要訣！](https://www.youtube.com/watch?v=FKXRiAiQFiY)
      - [How to use Markdown for writing technical documentation](https://experienceleague.adobe.com/en/docs/contributor/contributor-guide/writing-essentials/markdown)
    - 上傳大型檔案到 github 上，請參考：
      - [Git Large File Storage - An open source Git extension for versioning large files](https://git-lfs.com/)
      - [我如何使用 Git LFS 來託付大型 Git 檔案？](https://www.webdong.dev/zh-tw/post/how-i-use-git-lfs-to-manage-large-git-files/)
    - 檔案放置結構要清楚，建議如下：
      ```
      models/
      titanic_restful_project/
      README.md
      ```
    - 可以使用以下方式儲存模型：
      ```python
      import joblib

      # 儲存模型
      joblib.dump(model, 'models/your_model_name.joblib')

      # 載入模型
      model = joblib.load('models/your_model_name.joblib')

      # 預測模型
      y_pred = loaded_model.predict(X_test)
      y_prob = loaded_model.predict_proba(X_test)
      ```
    - `README.md` 要有說明，例如:
        ```markdown
        # 使用 Restful API 與 Ajax 來進行機器學習模型訓練與預測

        ## 安裝套件
        - Flask==3.1.3
        - pandas==3.0.3
        - scikit-learn==1.9.0
        (版本號可用 pip list，或是 conda list 來檢視，上面是範例，你的可能跟我不一樣，請自行調整)
        ...

        ## 執行方法
        python init_db.py
        python app.py
        (或是其它你覺得比較好的執行方式，請自行調整)

        ## 說明
        (介紹你使用的模型，理想的超參數是哪些，剩下可以自由發揮)

        ## 成果
        ![](執行過程的擷圖或說明圖片)
        ...
        [影片名稱或其它標題](你的影片連結)
        ...

        ## 其它你想要補充的資訊
        ...
        ```
- 沒交：`?` 分。
- 繳交時間
  - 原則上最後一堂課結束後 2 週內，準確時間上課說明。

## 功能說明
- [Ollama](./ollama/README.md)
- [Leaflet](./leaflet/README.md)
- [E-book](./ebook/README.md)
- [Titanic RESTful API + Ajax 機器學習訓練與預測](./titanic_restful_project/README.md)
