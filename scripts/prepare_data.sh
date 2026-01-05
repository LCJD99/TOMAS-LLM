#!/bin/bash

###############################################################################
# TOMAS-LLM Phase 1: 数据准备脚本
# 功能: 一键处理 raw 数据生成训练数据
# 包含步骤:
#   1. 构造 Tool Registry (Phase 1.2)
#   2. 构造训练数据 (Phase 1.3)
###############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python 环境
PYTHON_ENV="python3"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  TOMAS-LLM Phase 1: 数据准备流程${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# 检查原始数据文件
echo -e "${YELLOW}[检查] 验证原始数据文件...${NC}"
if [ ! -f "data/raw/tools.json" ]; then
    echo -e "${RED}[错误] 缺少文件: data/raw/tools.json${NC}"
    exit 1
fi
if [ ! -f "data/raw/profiling.csv" ]; then
    echo -e "${RED}[错误] 缺少文件: data/raw/profiling.csv${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] 原始数据文件完整${NC}"
echo ""

# 创建输出目录
echo -e "${YELLOW}[准备] 创建输出目录...${NC}"
mkdir -p data/registry
mkdir -p data/processed
echo -e "${GREEN}[✓] 输出目录已就绪${NC}"
echo ""

###############################################################################
# Step 1: 构造 Tool Registry (Phase 1.2)
###############################################################################
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Step 1/2: 构造 Tool Registry${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo -e "${YELLOW}[执行] 运行 build_registry.py...${NC}"
$PYTHON_ENV src/data/build_registry.py \
    --tools data/raw/tools.json \
    --profiling data/raw/profiling.csv \
    --output data/registry/tool_registry.json

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[✓] Tool Registry 生成成功${NC}"
    echo -e "${GREEN}    输出文件: data/registry/tool_registry.json${NC}"
else
    echo -e "${RED}[×] Tool Registry 生成失败${NC}"
    exit 1
fi
echo ""

###############################################################################
# Step 2: 构造训练数据 (Phase 1.3)
###############################################################################
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Step 2/2: 构造训练数据${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo -e "${YELLOW}[执行] 运行 build_instruction_data.py...${NC}"
$PYTHON_ENV src/data/build_instruction_data.py \
    --registry data/registry/tool_registry.json \
    --output_dir data/processed/ \
    --show_examples 3

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[✓] 训练数据生成成功${NC}"
    echo -e "${GREEN}    输出文件: data/processed/train.jsonl${NC}"
    echo -e "${GREEN}    元数据: data/processed/metadata.json${NC}"
else
    echo -e "${RED}[×] 训练数据生成失败${NC}"
    exit 1
fi
echo ""