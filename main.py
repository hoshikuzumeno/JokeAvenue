import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tkinter as tk
import random
import threading
import io
import time
from PIL import Image, ImageTk

# SD-Turbo用のライブラリ
import torch
from diffusers import AutoPipelineForText2Image

# シュールなプロンプトを生成するためのダミー単語リスト（日本語, 英語）
SUBJECTS = [
    ("巨大な目玉焼き", "a giant fried egg"), 
    ("空飛ぶクジラ", "a flying whale"), 
    ("時計が溶けた世界", "melting clocks"), 
    ("喋るサボテン", "a talking cactus"), 
    ("逆さまの街", "an upside-down city"), 
    ("光るキノコ", "glowing mushrooms")
]
ENVIRONMENTS = [
    ("深海のような森", "a forest like the deep sea"), 
    ("宇宙空間の砂漠", "a desert in outer space"), 
    ("ゼリーでできた海", "an ocean made of jelly"), 
    ("永遠の夕暮れ", "an eternal sunset"), 
    ("鏡の迷宮", "a labyrinth of mirrors")
]
ACTIONS = [
    ("が踊っている", "dancing"), 
    ("が静かに燃えている", "burning quietly"), 
    ("がお茶会をしている", "having a tea party"), 
    ("が溶け出している", "melting away"), 
    ("が浮遊している", "floating in the air")
]
STYLES = [
    ("サイバーパンク風", "cyberpunk style"), 
    ("水彩画風", "watercolor painting"), 
    ("油絵のような", "like an oil painting"), 
    ("ドット絵風", "pixel art"), 
    ("ネオンカラーで", "in neon colors")
]

# 昔のインターネット風フォント設定
FONT_FAMILY = "MS PGothic"
TITLE_FONT = (FONT_FAMILY, 18, "bold")
LOADING_FONT = (FONT_FAMILY, 36, "bold") # 最大サイズ
PROMPT_FONT = (FONT_FAMILY, 16, "bold")

class JokeAvenueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jokeavenue - シュールな景色生成器 (ローカルSD-Turbo版)")
        self.root.geometry("800x600")
        self.root.config(bg="#000000") # アプリ全体の背景を黒
        
        self.current_image = None
        self.pipe = None # モデルを保持する変数

        # ウィンドウリサイズ時のレイアウト調整
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)

        # 1. プレビュー枠（キャンバスを使用）
        self.preview_frame = tk.Frame(self.root, bg="#000000", padx=10, pady=10)
        self.preview_frame.grid(row=0, column=0, sticky="nsew")
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.preview_frame, bg="#000000", bd=10, relief="ridge", highlightthickness=5, highlightbackground="#FFFF00", highlightcolor="#FFFF00")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.initial_text_drawn = False

        # 3. プロンプト確認用テキストエリア
        self.prompt_frame = tk.Frame(self.root, bg="#000000", padx=10, pady=5)
        self.prompt_frame.grid(row=1, column=0, sticky="ew")
        self.prompt_frame.columnconfigure(0, weight=1)
        
        self.prompt_label = tk.Label(self.prompt_frame, text="▼ 生成されたプロンプト ▼", bg="#000000", fg="#FF00FF", font=TITLE_FONT)
        self.prompt_label.grid(row=0, column=0, sticky="w")
        
        self.prompt_text = tk.Text(self.prompt_frame, height=3, wrap="word", state="disabled", 
                                   bg="#000000", fg="#00FF00", font=PROMPT_FONT, 
                                   insertbackground="#00FF00", bd=5, relief="sunken")
        self.prompt_text.grid(row=1, column=0, sticky="ew", pady=(5,0))

        # 2. 生成ボタン
        self.btn_frame = tk.Frame(self.root, bg="#000000", padx=10, pady=10)
        self.btn_frame.grid(row=2, column=0, sticky="ew")
        self.btn_frame.columnconfigure(0, weight=1)

        # 初回は無効化しておく
        self.generate_btn = tk.Button(self.btn_frame, text="新しい景色を生成する！", 
                                      command=self.start_generation, 
                                      font=TITLE_FONT, bg="#555555", fg="#FFFF00", 
                                      activebackground="#00FFFF", activeforeground="#FF0000",
                                      relief="raised", bd=8, cursor="hand2", state="disabled")
        self.generate_btn.grid(row=0, column=0, pady=10, ipadx=30, ipady=10)
        
        # 背景スレッドでSD-Turboモデルを読み込む
        threading.Thread(target=self.load_model_thread, daemon=True).start()

    def load_model_thread(self):
        try:
            # モデルロード中メッセージ
            self.root.after(0, self.display_message, "SD-Turboモデルを準備中デス！\n※初回ダウンロード時は数分かかります")
            
            # SD-Turboのロード処理
            self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32)
            self.pipe.to("cpu")
            
            # ロード完了時にUIを更新
            self.root.after(0, self.on_model_loaded)
        except Exception as e:
            self.root.after(0, self.display_error, f"モデル読み込みエラーデス！\n{e}")

    def on_model_loaded(self):
        self.display_message("モデルの準備完了！\n生成ボタンを押してください")
        self.generate_btn.config(state="normal", bg="#0000FF")

    def display_message(self, text):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # 起動直後などでサイズが取れない場合のフォールバック
        if canvas_width <= 1: canvas_width = 800
        if canvas_height <= 1: canvas_height = 400
        
        self.canvas.create_text(canvas_width/2, canvas_height/2, text=text, font=TITLE_FONT, fill="#FF00FF", justify="center", width=canvas_width-40)

    def on_canvas_configure(self, event):
        if not self.initial_text_drawn:
            # メッセージ描画は display_message 側に任せる
            self.initial_text_drawn = True
        elif self.current_image:
            self.canvas.delete("all")
            self.canvas.create_image(event.width/2, event.height/2, image=self.current_image, anchor=tk.CENTER)

    def start_generation(self):
        if not self.pipe:
            return
            
        self.generate_btn.config(state="disabled", bg="#555555") 
        
        self.canvas.config(bg="#000000")
        self.canvas.delete("all")
        self.current_image = None
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.canvas.create_text(canvas_width/2, canvas_height/2, text="ジョークアベニュー建設中デス！", font=LOADING_FONT, fill="#FF0000", justify="center", width=canvas_width-40)

        sub_jp, sub_en = random.choice(SUBJECTS)
        env_jp, env_en = random.choice(ENVIRONMENTS)
        act_jp, act_en = random.choice(ACTIONS)
        style_jp, style_en = random.choice(STYLES)
        
        prompt_jp = f"{env_jp}の大通りで、{sub_jp}{act_jp}様子。{style_jp}。"
        prompt_en = f"{sub_en} {act_en} on an avenue in {env_en}, {style_en}"
        
        self.prompt_text.config(state="normal")
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(tk.END, prompt_jp)
        self.prompt_text.config(state="disabled")

        threading.Thread(target=self.fetch_image_thread, args=(prompt_en,), daemon=True).start()

    def fetch_image_thread(self, prompt_en):
        try:
            # CPU環境でのSD-Turbo画像生成 (最速化)
            image = self.pipe(prompt=prompt_en, num_inference_steps=1, guidance_scale=0.0).images[0]
            
            self.root.after(0, self.display_image, image)
        except Exception as e:
            error_msg = f"生成エラーデス！\n{e}"
            self.root.after(0, self.display_error, error_msg)

    def display_image(self, pil_image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 0 and canvas_height > 0:
            pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
        self.current_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width/2, canvas_height/2, image=self.current_image, anchor=tk.CENTER)

        self.generate_btn.config(state="normal", bg="#0000FF")

    def display_error(self, error_msg):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.canvas.create_text(canvas_width/2, canvas_height/2, text=error_msg, font=PROMPT_FONT, fill="#FF0000", justify="center", width=canvas_width-40)
        
        self.generate_btn.config(state="normal", bg="#0000FF")

if __name__ == "__main__":
    root = tk.Tk()
    app = JokeAvenueApp(root)
    root.mainloop()
