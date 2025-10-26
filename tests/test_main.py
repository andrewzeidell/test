import os
import pandas as pd
import joblib
import pytest
import matplotlib.pyplot as plt
from io import BytesIO

from ML_Project import preprocess, train_cmd, process, evaluate, report


@pytest.fixture
def sample_data(tmp_path):
    # Create a simple CSV file
    data = pd.DataFrame({"x": [1,2,3,4,5], "y": [2,4,6,8,10]})
    data_path = tmp_path / "simple_data.csv"
    data.to_csv(data_path, index=False)
    return str(data_path)

def test_workflow(sample_data, tmp_path):
    """End-to-end workflow: preprocess, train, predict, evaluate, and report building."""
    # Step 1: Preprocess
    clean_path = data.preprocess_data(sample_data)
    assert os.path.exists(clean_path)

    # Step 2: Train
    model_path = clean_path.replace("_clean.csv", "_model.pkl")
    score = train.train_model(clean_path, model_path)
    assert os.path.exists(model_path)

    # Step 3: Predict
    model = joblib.load(model_path)
    df = pd.read_csv(clean_path)
    X = df[["x"]]
    preds = model.predict(X)
    df["pred"] = preds
    pred_path = clean_path.replace("_clean.csv", "_pred.csv")
    df.to_csv(pred_path, index=False)
    assert os.path.exists(pred_path)

    # Step 4: Evaluate
    y_true = df["y"]
    y_pred = df["pred"]
    mse = ((y_true - y_pred) ** 2).mean()
    assert mse < 1e-6  # Should be very close to zero for perfect fit

    # Step 5: Build report and check output
    fig_path = tmp_path / "eval_plot.png"
    # Use a dummy plot for test
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(fig_path)
    zip_bytes = report.build_report(pred_path, str(fig_path))
    assert isinstance(zip_bytes, BytesIO)
    assert zip_bytes.getbuffer().nbytes > 0