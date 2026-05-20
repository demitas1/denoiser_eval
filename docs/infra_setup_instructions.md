# AWS スポットインスタンスによる ML トレーニング環境セットアップ指示書

このドキュメントは Claude Code 向けの指示書です。既存の機械学習リポジトリに、AWS スポットインスタンスを利用した低コストなトレーニング環境を構築します。

## ゴール

以下のフローを自動化・再利用可能にする:

1. 学習用 AMI を事前に作成しておく
2. 学習開始時に AMI からスポットインスタンスを起動
3. 学習データとコードを S3 経由でインスタンスに同期
4. 学習開始(Slack に通知)
5. 学習終了を監視し、終了後にモデルを自動ダウンロード
6. インスタンスは自動削除

## 技術スタック

- **AWS**: EC2 (スポットインスタンス、GPU)、S3、IAM、Systems Manager Parameter Store
- **Terraform**: 永続リソース(S3、IAM、SG、起動テンプレート)を管理
- **Packer**: 学習用 AMI のビルド
- **PyTorch**: 学習フレームワーク
- **Slack Incoming Webhook**: 学習開始/完了/失敗の通知

## 想定スペック

- VRAM: 最大 32GB(基本は `g5.xlarge` / `g6.xlarge` の 24GB を想定)
- 学習データ: 最大 1GB
- ストレージ: EBS gp3 50GB

---

## 事前準備(ユーザーが実施する作業)

Claude Code 側でセットアップを始める前に、以下をユーザー側で済ませてもらう必要があります。指示書冒頭でユーザーに案内してください。

### 1. AWS アカウント準備

- AWS アカウントの作成(未作成の場合)
- IAM ユーザー作成と、ローカルでの `aws configure` 完了
  - 必要権限: EC2, S3, IAM, SSM, VPC のフルアクセス相当(初期セットアップ用)
- 利用リージョンの決定(例: `ap-northeast-1`)
- EC2 用 SSH キーペアの作成(AWS コンソール → EC2 → Key Pairs)
- GPU インスタンスのクォータ確認・引き上げ申請(`G and VT Spot Instance Requests` の vCPU 数)

### 2. Slack 準備

- Slack ワークスペース(個人専用で可、無料プランで十分)
- https://api.slack.com/apps で新規アプリ作成 → Incoming Webhook を有効化
- 通知先チャンネルを選択し、Webhook URL を取得
- 取得した URL は後ほど Parameter Store に登録するため控えておく

### 3. ローカル環境

- Terraform >= 1.5
- Packer >= 1.9
- AWS CLI v2
- jq(監視スクリプトで使用)

---

## 作成するファイル一覧

リポジトリ直下に `infra/` ディレクトリを切り、その配下に全てを配置する想定です。学習コード本体(`train.py` など)は既存リポジトリのものを使うため、ここでは作成しません。

### ルートに追加するもの

- `infra/README.md` — `infra/` 配下の使い方ガイド
- `.gitignore` への追記 — Terraform state、`.tfvars`、ローカル結果ディレクトリ、認証情報を除外

### `infra/packer/` — AMI ビルド

- `ml-base.pkr.hcl` — Packer テンプレート。Deep Learning AMI(Ubuntu, PyTorch)をベースに、`requirements.txt` を追加インストール
- `requirements.txt` — AMI に焼き込む Python パッケージ一覧
- `provisioners/setup.sh` — AMI ビルド時に実行するシェルスクリプト(追加パッケージインストール、ディレクトリ作成等)

### `infra/terraform/` — 永続リソース定義

- `main.tf` — provider 設定、共通 data sources(AMI 参照、VPC、caller identity)
- `variables.tf` — 入力変数(region, project_name, instance_type, key_name, allowed_ssh_cidr など)
- `outputs.tf` — bucket name, launch template ID, 各種 ARN を出力
- `s3.tf` — トレーニング用バケット(code/, data/, results/ プレフィックス)、ライフサイクルポリシー
- `iam.tf` — EC2 インスタンス用ロール、S3 と SSM Parameter Store へのアクセス権限、インスタンスプロファイル
- `security.tf` — SSH 用セキュリティグループ
- `launch_template.tf` — スポットインスタンス用起動テンプレート(AMI、インスタンスタイプ、IAM プロファイル、user_data、自動シャットダウン設定)
- `versions.tf` — Terraform / provider のバージョン制約
- `backend.tf.example` — S3 バックエンド設定のサンプル(ユーザーが必要に応じてコピーして使う)
- `terraform.tfvars.example` — 変数値のサンプル

### `infra/scripts/` — 運用 CLI

- `user-data.sh.tpl` — インスタンス起動時に実行されるスクリプトのテンプレート。S3 からコード/データ取得、Slack 通知、学習実行、結果アップロード、自動シャットダウンまで
- `train.sh` — ローカルから実行する学習開始スクリプト。S3 へのコード/データ同期 → インスタンス起動 → IP 表示
- `monitor.sh` — インスタンス状態を監視し、終了検知後に結果を自動ダウンロード
- `cleanup.sh` — 万一の取り残しインスタンスを検出・終了させる安全装置
- `setup-slack.sh` — Slack Webhook URL を Parameter Store に登録するヘルパー
- `lib/common.sh` — 各スクリプトで共有する関数(色付きログ出力、AWS CLI ラッパー等)

### `infra/training/` — 学習コード側の支援ライブラリ

学習コード本体は触らない方針だが、Slack 進捗通知などの便利モジュールは提供する:

- `notify.py` — `SLACK_WEBHOOK` 環境変数を読んで通知を送るユーティリティ。学習コードから `from notify import notify_slack` で使う想定
- `checkpoint.py` — スポット中断耐性のためのチェックポイント保存/復元ヘルパー(S3 への定期アップロード対応)
- `README.md` — これらのモジュールの組み込み方サンプル

### ドキュメント

- `infra/docs/SETUP.md` — 初回セットアップ手順(AWS 認証、Packer ビルド、Terraform apply、Slack 登録)
- `infra/docs/USAGE.md` — 日常運用の使い方(`train.sh` / `monitor.sh` の使い方、結果の取得、複数実験の並行実行)
- `infra/docs/TROUBLESHOOTING.md` — よくある問題(スポット枯渇、クォータ不足、Webhook 失敗、AMI が見つからない等)と対処
- `infra/docs/COST.md` — 想定コストとさらなる削減 Tips

---

## セットアップ作業の流れ(Claude Code 向け)

以下の順序で進めること。各ステップでユーザーに確認を取りながら進める。

### Step 1: リポジトリ調査

- 既存リポジトリの構造を把握する(`ls`, `tree`, 主要ファイルの確認)
- 既存の学習コード(`train.py` 相当)のエントリポイント、コマンドライン引数、データ読み込みパス、モデル保存パスを確認
- 既存の `requirements.txt` / `pyproject.toml` から依存パッケージを把握
- `.gitignore` の現状確認

把握した内容をユーザーに要約して提示し、認識のズレがないか確認する。

### Step 2: 設計判断のヒアリング

以下をユーザーに確認:

- AWS リージョン
- プロジェクト名(リソース命名のプレフィックス)
- 利用予定のインスタンスタイプ(デフォルト案: `g5.xlarge`)
- データを Git に含めるか/別途 S3 にアップロードするか
- Terraform state を S3 バックエンドで管理するか、ローカルで済ませるか
- 学習コードへの Slack 通知統合を行うか(`notify.py` の組み込み)

### Step 3: ファイル生成

上記「作成するファイル一覧」に従ってファイルを生成する。

- `infra/` 配下を作成
- 既存の学習コードのエントリポイントに合わせて `user-data.sh.tpl` の学習実行コマンドを調整
- `.gitignore` を更新

### Step 4: 動作確認手順の提示

ファイル生成完了後、以下の手順をユーザーに案内する(実行はユーザーが行う):

1. `infra/terraform/terraform.tfvars` を作成(`.example` をコピーして値を埋める)
2. `cd infra/packer && packer init . && packer build ml-base.pkr.hcl`
3. `cd infra/terraform && terraform init && terraform apply`
4. `./infra/scripts/setup-slack.sh` で Webhook URL を登録
5. 学習データを S3 にアップロード(初回のみ)
6. `./infra/scripts/train.sh` で学習開始
7. `./infra/scripts/monitor.sh` で監視・結果取得

### Step 5: README 更新

リポジトリのルート `README.md` に、`infra/` を使った学習方法へのリンクを追加する(既存 README の構成を尊重しつつ最小限の追記)。

---

## 設計上の注意点

Claude Code が実装する際に意識すべき点:

- **シークレットを絶対にリポジトリにコミットしない**。Slack Webhook URL、AWS 認証情報、`.tfvars` は `.gitignore` で除外
- **Terraform state はインスタンスを管理しない**。インスタンス起動は `aws ec2 run-instances --launch-template` で行い、state を汚さない
- **`instance_initiated_shutdown_behavior = "terminate"`** を必ず設定。User Data 内の `shutdown -h now` で自動削除されるようにする
- **コスト事故防止**: `cleanup.sh` および「想定より長く動いているインスタンスのアラート」を案内する
- **スポット中断耐性**: `checkpoint.py` を使ってチェックポイントを S3 に定期保存し、起動時に再開可能にする
- **User Data のテンプレート変数衝突**: Terraform の `templatefile()` の `${}` と Bash の `${}` が衝突するため、Bash 側は `$${}` でエスケープ
- **AMI 参照は data source 経由**で最新を取得(`most_recent = true`、owners = "self"、name filter で Packer 出力を引く)
- **複数実験の並行実行**を想定し、S3 の結果プレフィックスはタイムスタンプ + 実験名で衝突しないようにする
- **エラー時の通知**: User Data 内で `trap '... ERR'` を使い、失敗時も Slack に通知してからシャットダウン
- **学習コード本体は最小限の変更で済むようにする**。`notify.py` などは optional な import として、未設定環境でもエラーにならない作りにする

---

## 完了後にユーザーに渡すもの

- 生成ファイルの一覧と各ファイルの役割サマリ
- 次に実行すべきコマンドの順序(Step 4 の内容)
- 想定月額コストの概算
- トラブル時の問い合わせポイント(どのログ/どのドキュメントを見るか)
