# Crab Adventure RL 処理フロー

このドキュメントは、`game.html`（ブラウザゲーム）と `rl/`（強化学習）がどう連携し、
学習から実ゲームでの推論までどのような流れで処理が進むかをまとめたものです。

---

## 1. 全体の流れ

```
┌──────────────┐    物理ロジックを移植    ┌──────────────┐
│  game.html   │ ───────────────────────▶ │  rl/env.py   │
│ (ブラウザ)   │                          │ (Gym環境)    │
└──────────────┘                          └──────┬───────┘
       ▲                                         │ step / reset
       │                                         ▼
       │                                  ┌──────────────┐
       │                                  │ rl/train.py  │ ← rl/agent.py, rl/model.py
       │                                  │ (学習ループ) │
       │                                  └──────┬───────┘
       │                                         │ 保存
       │                                         ▼
       │                                  ┌──────────────────────┐
       │                                  │ rl/models/*.pth      │
       │                                  │ rl/logs/train_log.csv│
       │                                  └──────┬───────────────┘
       │                                         │
       │              ┌──────────────────────────┼──────────────────────┐
       │              ▼                          ▼                      ▼
       │     ┌────────────────┐        ┌──────────────────┐   ┌──────────────────┐
       │     │ rl/evaluate.py │        │ rl/export_weights│   │ rl/plot_training │
       │     │ (ε=0 で評価)   │        │ .py              │   │ .py              │
       │     └────────────────┘        └────────┬─────────┘   └──────────────────┘
       │                                        │ 生成
       │                                        ▼
       │                               ┌──────────────────────┐
       └─────────── 読み込み ───────── │ rl/model_weights.js  │
                                       └──────────────────────┘
```

| フェーズ | スクリプト | 入力 | 出力 |
|---|---|---|---|
| 学習 | `rl/train.py` | なし（環境は内蔵） | `models/best_model.pth`, `models/checkpoint_*.pth`, `logs/train_log.csv` |
| 評価 | `rl/evaluate.py` | `.pth` | コンソールにクリア率 |
| 可視化 | `rl/plot_training.py` | `logs/train_log.csv` | `logs/training_curve.png` |
| 書き出し | `rl/export_weights.py` | `.pth` | `rl/model_weights.js` |
| 実ゲーム推論 | `game.html` | `rl/model_weights.js` | ブラウザ上でAIがプレイ |

---

## 2. 環境（rl/env.py）の処理フロー

### 2.1 reset()

1. プレイヤーを初期位置 (x=80, y=328) に配置し、速度ゼロ・非接地状態にする
2. 敵10体をテンプレートからディープコピーして復元する
3. ステップカウンタ・累積報酬をゼロにする
4. 観測ベクトル（24次元）を返す

### 2.2 step(action)

```
step(action)
 ├─ x_before = プレイヤーのx座標を記録
 ├─ 4フレーム繰り返し（フレームスキップ）
 │    └─ _step_frame(action)
 │         ├─ 入力をvx/vyに反映（左右移動・ジャンプ）
 │         ├─ 重力を加算し、位置を更新
 │         ├─ 地面・足場との衝突解決（_resolve_solid）
 │         ├─ ブロックとの衝突解決（_resolve_block）
 │         ├─ 敵10体を移動・衝突解決
 │         ├─ プレイヤーと敵の接触判定
 │         │     ├─ 踏みつけ（トゲ以外） → 敵HP減少、プレイヤーが跳ねる
 │         │     └─ それ以外の接触         → 終了 (-100)
 │         ├─ ゴール旗接触 → 終了 (+500, cleared=True)
 │         └─ 画面下に落下 → 終了 (-100)
 │    （終了したら残りフレームはスキップ）
 ├─ 報酬計算
 │    ├─ 終了時: 終了報酬のみ
 │    └─ 継続時: +0.03 × Δx（前進した分のみ） − 0.01（生存ペナルティ）
 ├─ 10,000ステップ到達で truncated=True、−100 を追加
 └─ (obs, reward, terminated, truncated, {'cleared': bool}) を返す
```

### 2.3 観測ベクトル（_get_obs）

| インデックス | 内容 | 正規化 |
|---|---|---|
| 0 | プレイヤーx | ÷ 4000 |
| 1 | プレイヤーy | ÷ 400 |
| 2 | vx | [-3.5, 3.5] → [0, 1] |
| 3 | vy | [-11, 15] → [0, 1] |
| 4 | 接地フラグ | 0 / 1 |
| 5–14 | 前方地面マップ（10マス、32px刻み） | 0 / 1（メイン地面のみ、足場は含まない） |
| 15–22 | 前方敵マップ（8マス、32px刻み） | 0 / 1 |
| 23 | ゴールまでの距離 | ÷ 4000 |

### 2.4 行動空間

| ID | 行動 |
|---|---|
| 0 | 右移動 |
| 1 | 右移動 + ジャンプ |
| 2 | ジャンプのみ |
| 3 | 左移動 |

---

## 3. 学習（rl/train.py）の処理フロー

```
train(args)
 ├─ シード固定（random / numpy / torch）
 ├─ models/, logs/ ディレクトリ作成
 ├─ デバイス選択（cuda があれば cuda、なければ cpu）
 ├─ 環境と DQNAgent を生成
 ├─ logs/train_log.csv を新規作成（既存ログは上書き）
 │
 └─ while 総ステップ < 5,000,000:
      ├─ 50エピソードに1回だけ描画フラグを立てる（--no-render 時は常にオフ）
      ├─ env.reset()
      │
      ├─ while エピソード継続:
      │    ├─ 描画フラグが立っていれば HUD を更新して env.render()
      │    ├─ action = agent.select_action(obs)    … ε-greedy
      │    ├─ env.step(action)
      │    ├─ agent.push(...)                       … バッファ追加 + ε減衰
      │    ├─ agent.learn()                         … ネットワーク更新（後述）
      │    └─ terminated or truncated なら抜ける
      │
      ├─ 直近100エピソードのクリア率を計算
      ├─ 100エピソード溜まっていて過去最高を更新 → models/best_model.pth 保存
      ├─ 前回保存から50,000ステップ以上経過      → models/checkpoint_{step}.pth 保存
      ├─ CSV に1行追記（episode, total_steps, reward, length, cleared, epsilon）
      └─ 10エピソードごとにコンソール出力
```

### 3.1 コマンドライン引数

| 引数 | 意味 |
|---|---|
| `--no-render` | pygame 描画を完全に無効化（高速学習） |
| `--seed N` | 乱数シード（デフォルト 42） |

### 3.2 注意点

- 学習の途中再開機能はない。停止したら最初からになる。
- `train_log.csv` は起動時に上書きされる。
- チェックポイントの保存判定はエピソード終了時に行うため、ファイル名のステップ数は 50,000 の倍数ちょうどにはならない。

---

## 4. エージェント（rl/agent.py）の処理フロー

### 4.1 select_action(state)

```
乱数 < ε ?
 ├─ Yes → ランダムに 0〜3 を返す（探索）
 └─ No  → オンラインネットに state を入力し、Q値最大の行動を返す（活用）
```

### 4.2 push(state, action, reward, next_state, done)

1. 遷移をリプレイバッファ（容量 100,000 の deque）に追加
2. 総ステップ数を +1
3. ε を線形減衰：`ε = 1.0 → 0.05` を最初の 100,000 ステップで

### 4.3 learn()

```
learn()
 ├─ バッファが 1,000 件未満 → 何もせず None を返す（ウォームアップ）
 ├─ バッファから 64 件をランダムサンプリング
 │
 ├─ 目標Q値の計算（Double DQN、勾配なし）
 │    ├─ next_acts = argmax( online_net(next_states) )     … 行動選択はオンライン側
 │    ├─ next_q    = target_net(next_states)[next_acts]    … 評価はターゲット側
 │    └─ target_q  = reward + γ(0.99) × next_q × (1 − done)
 │
 ├─ current_q = online_net(states)[actions]
 ├─ loss = SmoothL1Loss(current_q, target_q)   … Huber損失
 │
 ├─ optimizer.zero_grad()
 ├─ loss.backward()
 ├─ 勾配ノルムを 10 でクリップ
 ├─ optimizer.step()                            … Adam, lr=1e-4
 │
 ├─ 更新回数 +1、1,000 回ごとに target_net ← online_net をコピー
 └─ loss 値を返す
```

### 4.4 ネットワーク構造（rl/model.py）

```
入力 24
 → Linear(24, 128) + ReLU
 → Linear(128, 128) + ReLU
 → Linear(128, 64) + ReLU
 → Linear(64, 4)          … 各行動のQ値
```

オンラインネットとターゲットネットは同じ構造で、ターゲット側は一定間隔でコピーされる遅れたレプリカ。

---

## 5. 評価（rl/evaluate.py）の処理フロー

1. 環境と DQNAgent を生成し、指定した `.pth` を読み込む
2. `agent.eps = 0.0` にして完全 greedy にする
3. 指定エピソード数（デフォルト 20）を実行し、各エピソードの CLEAR / fail と報酬を表示
4. 最後にクリア率を表示

| 引数 | 意味 |
|---|---|
| `--model PATH` | 評価するモデル（デフォルト `rl/models/best_model.pth`） |
| `--episodes N` | エピソード数（デフォルト 20） |
| `--no-render` | 描画なし |

---

## 6. 重みの書き出し（rl/export_weights.py）

1. `.pth` を読み込んで DQN を復元
2. 4つの Linear 層それぞれの `weight`（2次元リスト）と `bias`（1次元リスト）を取り出す
3. `rl/model_weights.js` に以下の形式で書き出す

```js
const RL_MODEL_WEIGHTS = {"layers":[{"weight":[[...]],"bias":[...]}, ...]};
```

---

## 7. 実ゲームでの推論（game.html）

### 7.1 起動時

- `<script src="rl/model_weights.js">` で重みを読み込む
- 読み込めれば `rlModel` に格納、失敗時は `null`（ルールベースAIにフォールバック）

### 7.2 F キーで AI モード切り替え

```
computeAIInput()   … update() から毎フレーム呼ばれる
 ├─ start / gameover / clear 画面なら 60 フレーム後に自動リトライ
 │
 ├─ rlModel あり（DQN モード）
 │    ├─ rlSkipCount == 0 のときだけ rlForward() を実行  … 4フレームに1回、学習時のフレームスキップを再現
 │    │    ├─ rlGetObs()   … env.py の _get_obs と同じ 24 次元を作る
 │    │    ├─ 4層の全結合を JS で順伝播（ReLU は Math.max(0, v)）
 │    │    └─ argmax を返す
 │    └─ 行動 ID をキー入力に変換
 │         0 → 右    1 → 右 + Space    2 → Space    3 → 左
 │
 └─ rlModel なし（ルールベース）
      ├─ 常に右へ進む
      ├─ 前方 50〜220px に地面がなければジャンプ
      └─ 敵が近づいたらジャンプ（トゲは 190px、他は 110px 手前）
```

### 7.3 学習環境との一致点

| 項目 | game.html | env.py |
|---|---|---|
| 物理定数（重力 0.5、ジャンプ −11、歩行 3.5、最大落下 15） | 同じ | 同じ |
| 衝突解決ロジック | 同じ（角かすめ対策込み） | 同じ |
| 敵の初期配置・速度 | 同じ | 同じ |
| 観測ベクトルの作り方 | `rlGetObs()` | `_get_obs()` |
| フレームスキップ | 4フレームに1回推論 | 4フレームに1回 step |
| ダッシュ | あり（Shift） | なし |
| コイン・スコア | あり | なし（報酬に含まれない） |

---

## 8. 典型的な作業手順

```powershell
# 1. 学習（画面なし・高速）
python rl/train.py --no-render

# 2. 学習曲線を確認
python rl/plot_training.py

# 3. 評価（greedy）
python rl/evaluate.py --no-render --episodes 100

# 4. 重みをブラウザ用に書き出し
python rl/export_weights.py

# 5. ローカルサーバーで起動し、ブラウザで F キーを押す
python -m http.server 8000
# → http://localhost:8000/game.html
```

---

## 9. 要件定義書（requirements_rl.md）との対応

要件書は 2026-09-30 に実装へ合わせて更新済み。当初の値から変更された項目は以下のとおり。

| 項目 | 当初の要件 | 現在の要件・実装 |
|---|---|---|
| 前進報酬 | 0.01 × Δx | 0.03 × Δx |
| リプレイバッファ容量 | 50,000 | 100,000 |
| ε 減衰期間 | 50,000 ステップ | 100,000 ステップ |
| 総学習ステップ上限 | 500,000 | 5,000,000 |

いずれも学習を安定させる方向の調整。
