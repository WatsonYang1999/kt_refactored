#!/bin/bash

# 检查是否提供了 commit message 参数
if [ -z "$1" ]; then
    echo "Error: Commit message is required."
    echo "Usage: ./git_push.sh \"Your commit message\""
    exit 1
fi

# 执行 Git 命令
git add .
git commit -m "$1"
git push github master
