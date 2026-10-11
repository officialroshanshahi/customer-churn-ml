import io, json, httpx, gradio as gr
from PIL import Image
from src.v1.config import API_URL

# function for connecting to the server
def api_call(endpoint: str, method: str = "GET", token: str = "", json_data: dict = None, raw_response: bool = False):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        res = httpx.request(method, f"{API_URL}{endpoint}", json=json_data, headers=headers, timeout=30.0)
        if raw_response:
            return Image.open(io.BytesIO(res.content)) if res.status_code == 200 else None
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# authentication function
def handle_auth(action, user, pwd):
    res = api_call(f"/{action.lower()}", "POST", json_data={"username": user, "password": pwd})
    token = res.get("access_token", "") if action == "Login" and isinstance(res, dict) else ""
    return token, res


# for /predict and /explain as both of them take single dataset
def handle_single_inference(action, token, cid, age, gender, tenure, usage, support, delay, sub, contract, spend, last_int):
    payload = {
        "data": {
            "CustomerID": int(cid),
            "Age": int(age),
            "Gender": gender,
            "Tenure": int(tenure),
            "Usage Frequency": int(usage),
            "Support Calls": int(support),
            "Payment Delay": int(delay),
            "Subscription Type": sub,
            "Contract Length": contract,
            "Total Spend": float(spend),
            "Last Interaction": int(last_int)
        }
    }
    
    # if /predict
    if action == "Predict":
        return api_call("/predict", "POST", token=token, json_data=payload), None
    
    # if not
    img = api_call("/explain", "POST", token=token, json_data=payload, raw_response=True)
    
    return {"status": "image_generated" if img else "failed"}, img


# for handling batch inference
def handle_batch_inference(token, json_str):
    try:
        data_list = json.loads(json_str)
        payload = {"data": data_list} if isinstance(data_list, list) else data_list
        return api_call("/predict/batch", "POST", token=token, json_data=payload)
    except Exception as e:
        return {"error": f"Invalid JSON format: {str(e)}"}


# simple UI layout
with gr.Blocks(title="Customer Churn ML Platform") as demo:
    
    auth_token = gr.State("")

    gr.Markdown("Customer Churn ML Control Center")
    
    with gr.Tab("Inference & Authentication"):
        with gr.Row():
            # Auth & System Status
            with gr.Column(scale=1):
                gr.Markdown("Auth & System")
                user_in = gr.Textbox(label="Username")
                pass_in = gr.Textbox(label="Password", type="password")
                
                with gr.Row():
                    btn_login = gr.Button("Login", variant="primary")
                    btn_reg = gr.Button("Register")
                
                with gr.Row():
                    btn_status = gr.Button("/status")
                    btn_health = gr.Button("/health")
                    
                token_disp = gr.Textbox(label="Active Bearer Token", interactive=False, max_lines=1)
                sys_out = gr.JSON(label="System Response")

            # for handling single inference and SHAP
            with gr.Column(scale=2):
                gr.Markdown("Single Customer Inference & Explanation")
                with gr.Row():
                    cid = gr.Number(label="Customer ID", value=101)
                    age = gr.Slider(18, 100, value=30, label="Age")
                    gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
                    tenure = gr.Number(label="Tenure", value=12)
                with gr.Row():
                    usage = gr.Number(label="Usage Freq", value=15)
                    support = gr.Number(label="Support Calls", value=2)
                    delay = gr.Number(label="Pay Delay", value=5)
                    spend = gr.Number(label="Total Spend", value=450.50)
                with gr.Row():
                    sub = gr.Dropdown(["Basic", "Standard", "Premium"], value="Standard", label="Subscription")
                    contract = gr.Dropdown(["Monthly", "Quarterly", "Annual"], value="Annual", label="Contract")
                    last_int = gr.Number(label="Last Interaction", value=3)

                with gr.Row():
                    btn_pred = gr.Button("Predict (/predict)", variant="primary")
                    btn_exp = gr.Button("Explain (/explain)")

                pred_out = gr.JSON(label="Prediction Result")
                img_out = gr.Image(label="SHAP Explanation Plot", type="pil")

    with gr.Tab("Model Metrics & Evaluation"):
        gr.Markdown("Model Performance Analysis (/evaluate)")
        btn_eval = gr.Button("Fetch Evaluation Plots", variant="primary")
        eval_img_out = gr.Image(label="Performance Charts (Bar Chart, ROC, PR Curves)", type="pil")

    with gr.Tab("Batch Operations & Monitoring"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Batch Processing (/predict/batch)")
                batch_json = gr.TextArea(
                    label="Batch JSON Input",
                    value='[\n {\n    "CustomerID": 101, "Age": 30, "Gender": "Female", "Tenure": 12,\n    "Usage Frequency": 15, "Support Calls": 2, "Payment Delay": 5,\n    "Subscription Type": "Standard", "Contract Length": "Annual",\n    "Total Spend": 450.50, "Last Interaction": 3\n  }\n]',
                    lines=8
                )
                btn_batch = gr.Button("Run Batch Prediction")
                batch_out = gr.JSON(label="Batch Results")

            with gr.Column():
                gr.Markdown("### Ground Truth & Production Monitoring")
                with gr.Row():
                    req_id = gr.Textbox(label="Request ID")
                    churn_val = gr.Radio([0, 1], value=0, label="Actual Churn")
                btn_label = gr.Button("Submit Ground Truth (/label)")
                label_out = gr.JSON(label="Label Response")

                gr.Markdown("#### Data Drift & Retraining Check")
                btn_monitor = gr.Button("Run Production Monitoring (/monitor)", variant="primary")
                mon_out = gr.JSON(label="Monitoring Diagnostics")

    # for handling events
    btn_login.click(lambda u, p: handle_auth("Login", u, p), inputs=[user_in, pass_in], outputs=[auth_token, sys_out])
    btn_reg.click(lambda u, p: handle_auth("Register", u, p), inputs=[user_in, pass_in], outputs=[auth_token, sys_out])
    auth_token.change(lambda t: t, inputs=[auth_token], outputs=[token_disp])

    
    btn_status.click(lambda: api_call("/status"), outputs=[sys_out])
    btn_health.click(lambda t: api_call("/health", token=t), inputs=[auth_token], outputs=[sys_out])

    
    inf_inputs = [auth_token, cid, age, gender, tenure, usage, support, delay, sub, contract, spend, last_int]
    btn_pred.click(lambda *args: handle_single_inference("Predict", *args), inputs=inf_inputs, outputs=[pred_out, img_out])
    btn_exp.click(lambda *args: handle_single_inference("Explain", *args), inputs=inf_inputs, outputs=[pred_out, img_out])

    
    btn_eval.click(lambda t: api_call("/evaluate", "GET", token=t, raw_response=True), inputs=[auth_token], outputs=[eval_img_out])
    btn_batch.click(handle_batch_inference, inputs=[auth_token, batch_json], outputs=[batch_out])
    btn_label.click(lambda t, r, c: api_call("/label", "POST", token=t, json_data={"request_id": r, "churn": int(c)}), inputs=[auth_token, req_id, churn_val], outputs=[label_out])
    btn_monitor.click(lambda t: api_call("/monitor", "POST", token=t), inputs=[auth_token], outputs=[mon_out])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())