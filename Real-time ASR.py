import webview
from app import demo as gradio_app

# If set to True, the gradio app will not block and the gradio server will terminate as soon as the script finishes.
# https://www.gradio.app/4.44.1/docs/gradio/blocks
gradio_app.launch(prevent_thread_lock=True)

# https://pywebview.flowrl.com/api/#webview-create-window
webview.create_window(
    title="Real-time ASR", 
    url=gradio_app.local_url,
    width=1300,
    height=450,
    # 可调整大小
    resizable=True,
    # 始终位于其它窗口之上
    on_top=False,
    )
webview.start()