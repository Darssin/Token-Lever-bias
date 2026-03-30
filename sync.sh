#!/bin/bash

# --- 配置区域 ---
LOCAL_DIR="$(pwd)/" 
REMOTE_DIR="/home/wubintao.6/TLB_grc/"
SSH_PROXY='wsCli -a dt02.easyalgo.jd.com -t d3ViaW50YW8uNiZ3dWJpbnRhby11c2VyLXByb2ZpbGUmMDYyNThiMGVkNjEyNDAwZGE3MjRiNTBiYTRkMjgzOTA='

# --- 执行同步的函数 ---
do_sync() {
    # 核心修改：去掉了 --delete 参数
    # -a: 归档模式 (保留权限、时间戳等)
    # -v: 详细输出 (方便调试时观察)
    # -z: 压缩传输
    rsync -avz \
        -e "ssh -o ProxyCommand='$SSH_PROXY'" \
        --exclude '.git/' \
        --exclude '.vscode/' \
        --exclude '.DS_Store' \
        --exclude '__pycache__/' \
        "$LOCAL_DIR" "wubintao.6@localhost:$REMOTE_DIR"
}

echo "--- 正在启动增量同步模式（不删除远程冗余文件） ---"
echo "--- 按下 Ctrl + C 停止同步 ---"

while true; do
    # 执行同步
    do_sync > /dev/null 2>&1 
    
    # 轮询间隔
    sleep 2
done