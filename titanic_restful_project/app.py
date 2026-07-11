import sqlite3
import os
import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

app = Flask(__name__)

# ============================================================
# 1. 全域讀取資料庫
# ============================================================

DATABASE = "my_db.db"

# 這裡我們直接在全域讀取資料庫，這樣在每個 route 就可以直接使用 db 來存取資料庫了。
db = sqlite3.connect(DATABASE, check_same_thread=False)

# 讓我們在讀取資料庫時，可以直接用 row["欄位名稱"] 的方式來存取資料，
# 而不是 row[0]、row[1] 這樣的 index。
db.row_factory = sqlite3.Row


# ============================================================
# 2. 小工具：把 SQLite Row 轉成 dict
# ============================================================

def row_to_dict(row):
    return dict(row)


# ============================================================
# 3. 前端頁面 Routes
# ============================================================

# 首頁
@app.route("/")
def index_page():
    return render_template("index.html")

# 新增乘客頁面
@app.route("/passengers/new")
def new_passenger_page():
    return render_template("new.html")

# 編輯乘客頁面
@app.route("/passengers/<int:passenger_id>/edit")
def edit_passenger_page(passenger_id):
    return render_template("edit.html", passenger_id=passenger_id)


# ============================================================
# 4. API：取得全部乘客資料，包含簡單分頁
# GET /api/passengers?page=1&per_page=20
# ============================================================

@app.route("/api/passengers", methods=["GET"])
def get_passengers():
    # 讀取 query string 的 page 和 per_page 參數，並設定預設值
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    # 搜尋姓名
    search = request.args.get("search", "")

    # 計算 SQL 查詢的 offset，用於分頁
    offset = (page - 1) * per_page

    # 根據是否有搜尋關鍵字，執行不同的 SQL 查詢
    if search != "":
        # 有輸入搜尋關鍵字：只查詢姓名符合的資料
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            WHERE Name LIKE ?
            """,
            (f"%{search}%",)
        ).fetchone()

        rows = db.execute(
            """
            SELECT *
            FROM titanic
            WHERE Name LIKE ?
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (f"%{search}%", per_page, offset)
        ).fetchall()

    else:
        # 沒有輸入搜尋關鍵字：查詢全部資料
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            """
        ).fetchone()

        # 根據 page 和 per_page 的值，從資料庫查詢對應的資料列，
        # 並按照 PassengerId 排序。
        rows = db.execute(
            """
            SELECT *
            FROM titanic
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (per_page, offset)
        ).fetchall()

    # 總共有多少筆資料
    total = total_row["total"]

    # 最後回傳 JSON 格式的資料，包含 items（資料列表）、page、per_page 和 total。
    return jsonify({
        "message": "ok",
        "items": [row_to_dict(row) for row in rows],
        "page": page,
        "per_page": per_page,
        "total": total
    }), 200


# ============================================================
# 5. API：取得單一乘客
# GET /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["GET"])
def get_passenger(passenger_id):
    # 根據 passenger_id 查詢資料庫，看看有沒有這個乘客的資料。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    # 如果 row 是 None，代表資料庫裡沒有這個 passenger_id 的資料，我們就回傳 404 Not Found 的錯誤訊息。
    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    # 如果有找到資料，我們就把這筆資料轉成 dict，然後回傳 JSON 格式的資料。
    return jsonify({
        "message": "ok", 
        "item": row_to_dict(row)}
    ), 200


# ============================================================
# 6. API：新增乘客
# POST /api/passengers
# ============================================================

@app.route("/api/passengers", methods=["POST"])
def create_passenger():
    # 從 request 的 JSON body 讀取資料
    data = request.get_json()

    # 執行 SQL INSERT 語句，把新的乘客資料新增到 titanic 資料表中。
    cursor = db.execute(
        """
        INSERT INTO titanic (
            Survived, Pclass, Name, Sex, Age,
            SibSp, Parch, Ticket, Fare, Cabin,
            Embarked
        )
        VALUES (
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?
        )
        """,
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"]
        )
    )

    # 執行 commit()，把剛剛的 INSERT 操作真正寫入資料庫。
    db.commit()

    # cursor.lastrowid 會回傳剛剛 INSERT 的那筆資料的自動增加的 ID，
    # 也就是 PassengerId。
    new_id = cursor.lastrowid

    # 根據 new_id 查詢剛剛新增的那筆資料，這樣我們就可以把完整的資料回傳給前端了。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (new_id,)
    ).fetchone()

    # 最後回傳 JSON 格式的資料，包含 message 和 item（剛剛新增的那筆資料）。
    return jsonify({
        "message": "created",
        "item": row_to_dict(row)
    }), 201


# ============================================================
# 7. API：修改乘客
# PUT /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["PUT"])
def update_passenger(passenger_id):
    # 從 request 的 JSON body 讀取資料
    data = request.get_json()

    # 執行 SQL UPDATE 語句，根據 passenger_id 把對應的資料更新成新的值。
    cursor = db.execute(
        """
        UPDATE titanic
        SET
            Survived = ?,
            Pclass = ?,
            Name = ?,
            Sex = ?,
            Age = ?,
            SibSp = ?,
            Parch = ?,
            Ticket = ?,
            Fare = ?,
            Cabin = ?,
            Embarked = ?
        WHERE PassengerId = ?
        """,
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"],
            passenger_id
        )
    )

    # 執行 commit()，把剛剛的 UPDATE 操作真正寫入資料庫。
    db.commit()

    # 如果沒有更新任何資料，則回傳 404 Not Found 的錯誤訊息。
    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    # 根據 passenger_id 查詢剛剛更新的那筆資料，這樣我們就可以把完整的資料回傳給前端了。
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    # 如果 row 是 None，代表資料庫裡沒有這個 passenger_id 的資料，我們就回傳 404 Not Found 的錯誤訊息。
    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    # 最後回傳 JSON 格式的資料，包含 message 和 item（剛剛更新的那筆資料）。
    return jsonify({
        "message": "updated",
        "item": row_to_dict(row)
    }), 200


# ============================================================
# 8. API：刪除乘客
# DELETE /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["DELETE"])
def delete_passenger(passenger_id):
    # 執行 SQL DELETE 語句，根據 passenger_id 把對應的資料從 titanic 資料表中刪除。
    cursor = db.execute(
        "DELETE FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    )

    # 執行 commit()，把剛剛的 DELETE 操作真正寫入資料庫。
    db.commit()

    # 如果沒有刪除任何資料，則回傳 404 Not Found 的錯誤訊息。
    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    # 最後回傳 JSON 格式的資料，包含 message，告訴前端這筆資料已經被刪除了。
    return jsonify({
        "message": "deleted"
    }), 200 # 你也可以設定 204，但不會有 response body，前端無法判斷成功還是失敗


# ============================================================
# 9. 機器學習：模型訓練與預測
# ============================================================

# 頁面：模型訓練
@app.route("/ml/train")
def ml_train_page():
    return render_template("ml_train.html")

# 頁面：乘客生存預測
@app.route("/ml/predict")
def ml_predict_page():
    return render_template("ml_predict.html")


# 用來訓練與預測的特徵欄位
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
PASSTHROUGH_FEATURES = ["Pclass", "SibSp", "Parch"]


def build_preprocessor():
    """建立前處理流程：數值欄位補中位數，類別欄位補眾數後做 One-Hot Encoding。"""
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
        ("pass", SimpleImputer(strategy="most_frequent"), PASSTHROUGH_FEATURES),
    ])


# 兩個候選模型與各自要調整的超參數網格（預設值，前端可覆寫）
MODEL_CONFIGS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
        },
    },
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__class_weight": [None, "balanced"],
        },
    },
}

# 每個超參數欄位的型別與限制，用於解析、驗證前端傳來的自訂候選值
PARAM_FIELD_SPECS = {
    "RandomForest": {
        "n_estimators": {"type": "int", "min": 1},
        "max_depth": {"type": "int_or_none"},
        "min_samples_split": {"type": "int", "min": 2},
    },
    "LogisticRegression": {
        "C": {"type": "float", "min": 0.0001},
        "class_weight": {"type": "choice", "choices": [None, "balanced"]},
    },
}

# 網格搜尋組合數上限，避免使用者選了太多候選值導致訓練時間過長
MAX_GRID_COMBINATIONS = 60


def parse_param_values(raw, spec):
    """把前端傳來的單一超參數候選值（逗號分隔字串或陣列）解析成正確型別的清單。"""
    if raw is None or raw == "":
        return None

    parts = raw if isinstance(raw, list) else str(raw).split(",")
    parts = [str(p).strip() for p in parts if str(p).strip() != ""]
    if not parts:
        return None

    values = []
    for p in parts:
        if spec["type"] == "int":
            v = int(p)
            if "min" in spec and v < spec["min"]:
                raise ValueError(f"數值 {v} 小於允許的最小值 {spec['min']}")
            values.append(v)
        elif spec["type"] == "int_or_none":
            values.append(None if p.lower() == "none" else int(p))
        elif spec["type"] == "float":
            v = float(p)
            if "min" in spec and v < spec["min"]:
                raise ValueError(f"數值 {v} 小於允許的最小值 {spec['min']}")
            values.append(v)
        elif spec["type"] == "choice":
            choice = None if p.lower() == "none" else p
            if choice not in spec["choices"]:
                raise ValueError(f"不支援的值：{p}")
            values.append(choice)

    # 去除重複值，並保留原本的順序
    unique_values = []
    for v in values:
        if v not in unique_values:
            unique_values.append(v)

    return unique_values


def build_param_grid(model_name, raw_fields):
    """根據前端傳來的自訂欄位，組出 GridSearchCV 用的 param_grid；沒有提供或無效的欄位則使用預設值。"""
    default_grid = MODEL_CONFIGS[model_name]["param_grid"]
    field_specs = PARAM_FIELD_SPECS[model_name]
    raw_fields = raw_fields or {}

    param_grid = {}
    for field, spec in field_specs.items():
        key = f"clf__{field}"
        values = parse_param_values(raw_fields.get(field), spec)
        param_grid[key] = values if values is not None else default_grid[key]

    return param_grid

# 模型存放路徑：repo 根目錄下的 models/ 資料夾
MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# 目前記憶體中可用的模型（訓練完成後或伺服器啟動時載入）
CURRENT_MODEL = None

# 保護 TRAIN_STATE 讀寫的鎖，避免背景訓練執行緒與 API 讀取同時衝突
TRAIN_LOCK = threading.Lock()

TRAIN_STATE = {
    "status": "idle",  # idle | training | done | error
    "progress": "",
    "started_at": None,
    "finished_at": None,
    "candidates": {},
    "best_model_name": None,
    "best_params": None,
    "cv_score": None,
    "test_accuracy": None,
    "precision": None,
    "recall": None,
    "f1": None,
    "auc": None,
    "confusion_matrix": None,
    "roc_points": None,
    "n_samples": None,
    "error": None,
}


def load_saved_model():
    """伺服器啟動時，如果先前已經訓練過模型，直接載入，讓頁面與預測功能立刻可用。"""
    global CURRENT_MODEL

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        CURRENT_MODEL = joblib.load(MODEL_PATH)

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        TRAIN_STATE.update(meta)
        TRAIN_STATE["status"] = "done"
        TRAIN_STATE["progress"] = ""


load_saved_model()


def compute_roc_points(y_true, y_proba, max_points=60):
    """計算 ROC 曲線座標點，並取樣到最多 max_points 個點，避免傳給前端的資料過大。"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    idx = sorted(set(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int).tolist()))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def run_training(param_grids):
    """在背景執行緒執行：讀取 titanic 資料表 -> 對兩個候選模型做 GridSearchCV -> 挑最佳者存檔。

    param_grids: {模型名稱: param_grid}，由呼叫端（API route）先解析並驗證好。
    """
    global CURRENT_MODEL

    with TRAIN_LOCK:
        TRAIN_STATE["status"] = "training"
        TRAIN_STATE["progress"] = "讀取資料中..."
        TRAIN_STATE["started_at"] = datetime.now().isoformat(timespec="seconds")
        TRAIN_STATE["finished_at"] = None
        TRAIN_STATE["error"] = None
        TRAIN_STATE["candidates"] = {}

    try:
        df = pd.read_sql("SELECT * FROM titanic", db)

        X = df[FEATURE_COLUMNS]
        y = df["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        candidates = {}
        best_name = None
        best_test_acc = -1.0
        best_pipeline = None
        best_params = None
        best_cv_score = None
        best_precision = None
        best_recall = None
        best_f1 = None
        best_auc = None
        best_confusion_matrix = None
        best_roc_points = None

        for name, config in MODEL_CONFIGS.items():
            with TRAIN_LOCK:
                TRAIN_STATE["progress"] = f"正在訓練 {name}（GridSearchCV 超參數調整中）..."

            pipeline = Pipeline([
                ("prep", build_preprocessor()),
                ("clf", config["estimator"]),
            ])

            param_grid = param_grids[name]

            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=1,
            )
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            y_proba = grid.predict_proba(X_test)[:, 1]

            test_acc = float(accuracy_score(y_test, y_pred))
            cv_score = float(grid.best_score_)
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            auc = float(roc_auc_score(y_test, y_proba))
            cm = confusion_matrix(y_test, y_pred).tolist()
            roc_points = compute_roc_points(y_test, y_proba)

            candidates[name] = {
                "param_grid": param_grid,
                "best_params": grid.best_params_,
                "cv_score": cv_score,
                "test_accuracy": test_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "confusion_matrix": cm,
                "roc_points": roc_points,
            }

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_name = name
                best_pipeline = grid.best_estimator_
                best_params = grid.best_params_
                best_cv_score = cv_score
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_auc = auc
                best_confusion_matrix = cm
                best_roc_points = roc_points

        joblib.dump(best_pipeline, MODEL_PATH)

        meta = {
            "candidates": candidates,
            "best_model_name": best_name,
            "best_params": best_params,
            "cv_score": best_cv_score,
            "test_accuracy": best_test_acc,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "auc": best_auc,
            "confusion_matrix": best_confusion_matrix,
            "roc_points": best_roc_points,
            "n_samples": len(df),
        }

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        CURRENT_MODEL = best_pipeline

        with TRAIN_LOCK:
            TRAIN_STATE.update(meta)
            TRAIN_STATE["status"] = "done"
            TRAIN_STATE["progress"] = ""
            TRAIN_STATE["finished_at"] = datetime.now().isoformat(timespec="seconds")

    except Exception as e:
        with TRAIN_LOCK:
            TRAIN_STATE["status"] = "error"
            TRAIN_STATE["progress"] = ""
            TRAIN_STATE["error"] = str(e)
            TRAIN_STATE["finished_at"] = datetime.now().isoformat(timespec="seconds")


# API：一鍵開始訓練（背景執行，立即回應）
# POST /api/ml/train
# body（可省略，省略則使用預設超參數網格）：
# {
#   "RandomForest": {"n_estimators": "100,200,300", "max_depth": "None,5,10", "min_samples_split": "2,5,10"},
#   "LogisticRegression": {"C": "0.01,0.1,1,10", "class_weight": "None,balanced"}
# }
@app.route("/api/ml/train", methods=["POST"])
def start_training():
    body = request.get_json(silent=True) or {}

    try:
        param_grids = {
            name: build_param_grid(name, body.get(name))
            for name in MODEL_CONFIGS
        }
    except ValueError as e:
        return jsonify({"error": f"超參數格式錯誤：{e}"}), 400

    for name, grid in param_grids.items():
        combo_count = 1
        for values in grid.values():
            combo_count *= len(values)
        if combo_count > MAX_GRID_COMBINATIONS:
            return jsonify({
                "error": f"{name} 的超參數組合數（{combo_count}）過多，請減少候選值數量（上限 {MAX_GRID_COMBINATIONS}）"
            }), 400

    with TRAIN_LOCK:
        if TRAIN_STATE["status"] == "training":
            return jsonify({"error": "已有訓練正在進行中，請稍候"}), 409

    thread = threading.Thread(target=run_training, args=(param_grids,), daemon=True)
    thread.start()

    return jsonify({"message": "started"}), 202


# API：查詢目前訓練狀態，前端用輪詢的方式呼叫
# GET /api/ml/train/status
@app.route("/api/ml/train/status", methods=["GET"])
def get_training_status():
    with TRAIN_LOCK:
        return jsonify(dict(TRAIN_STATE)), 200


def to_float_or_nan(value):
    if value in (None, ""):
        return float("nan")
    return float(value)


def build_feature_row(data):
    return pd.DataFrame([{
        "Pclass": int(data["Pclass"]),
        "Sex": data["Sex"],
        "Age": to_float_or_nan(data.get("Age")),
        "SibSp": int(data.get("SibSp") or 0),
        "Parch": int(data.get("Parch") or 0),
        "Fare": to_float_or_nan(data.get("Fare")),
        "Embarked": data.get("Embarked") or None,
    }])


# API：單筆預測
# POST /api/ml/predict
@app.route("/api/ml/predict", methods=["POST"])
def predict_single():
    if CURRENT_MODEL is None:
        return jsonify({"error": "尚未訓練模型，請先到模型訓練頁面完成訓練"}), 400

    data = request.get_json()
    X = build_feature_row(data)

    survived = int(CURRENT_MODEL.predict(X)[0])
    probability = float(CURRENT_MODEL.predict_proba(X)[0][1])

    return jsonify({
        "survived": survived,
        "probability": probability
    }), 200


# API：CSV 批次預測
# POST /api/ml/predict/batch
@app.route("/api/ml/predict/batch", methods=["POST"])
def predict_batch():
    if CURRENT_MODEL is None:
        return jsonify({"error": "尚未訓練模型，請先到模型訓練頁面完成訓練"}), 400

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "請上傳 CSV 檔案"}), 400

    df = pd.read_csv(file)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        return jsonify({"error": f"CSV 缺少欄位: {', '.join(missing_cols)}"}), 400

    X = df[FEATURE_COLUMNS]

    result_df = df.copy()
    result_df["PredictedSurvived"] = CURRENT_MODEL.predict(X)
    result_df["Probability"] = CURRENT_MODEL.predict_proba(X)[:, 1]

    # 透過 to_json 再 loads 回來，確保 numpy 型別與 NaN 都能正確轉成 JSON 可序列化的型別
    items = json.loads(result_df.to_json(orient="records"))

    return jsonify({"items": items}), 200


# ============================================================
# 10. 啟動 Flask
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )