#!/bin/bash

# --- 配置区域 ---
LOCAL_DIR="$(pwd)/" 
REMOTE_DIR="/home/wubintao.6/TLB_demo/"
SSH_PROXY='wsCli -a dt02.easyalgo.jd.com -t d3ViaW50YW8uNiZ3dWJpbnRhby11c2VyLXByb2ZpbGUmMDYyNThiMGVkNjEyNDAwZGE3MjRiNTBiYTRkMjgzOTA='

# --- 执行同步的函数 ---
do_sync() {
    # 使用 rsync 进行增量同步
    rsync -avz --delete \
        -e "ssh -o ProxyCommand='$SSH_PROXY'" \
        --exclude '.git/' \
        --exclude '.vscode/' \
        --exclude '.DS_Store' \
        --exclude '__pycache__/' \
        "$LOCAL_DIR" "wubintao.6@localhost:$REMOTE_DIR"
}

echo "--- 正在启动轮询同步模式 (每 2 秒检测一次) ---"
echo "--- 按下 Ctrl + C 停止同步 ---"

# 使用死循环代替 fswatch
while true; do
    # 也可以加上输出提示，但为了清爽建议只在同步时输出
    do_sync > /dev/null 2>&1 
    
    # 如果你想看到同步成功的日志，可以取消下面这一行的注释
    # echo "[$(date +%H:%M:%S)] 轮询同步中..."
    
    sleep 2
done