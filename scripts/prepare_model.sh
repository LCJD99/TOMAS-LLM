#!/bin/bash

###############################################################################
# TOMAS-LLM Phase 2: 模型准备脚本
# 功能: 扩充 Tokenizer 并初始化 Embedding
# 包含步骤:
#   1. Tokenizer 扩充 (Phase 2.1)
#   2. Embedding 初始化 (Phase 2.2)
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

# 基础模型
BASE_MODEL="Qwen2.5-7B"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  TOMAS-LLM Phase 2: 模型准备流程${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# 检查依赖文件
echo -e "${YELLOW}[检查] 验证依赖文件...${NC}"
if [ ! -f "data/registry/tool_registry.json" ]; then
    echo -e "${RED}[错误] 缺少文件: data/registry/tool_registry.json${NC}"
    echo -e "${RED}请先运行: bash scripts/prepare_data.sh${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] 依赖文件完整${NC}"
echo ""

# 创建输出目录
echo -e "${YELLOW}[准备] 创建输出目录...${NC}"
mkdir -p checkpoints/tokenizer_expanded
mkdir -p checkpoints/model_initialized
echo -e "${GREEN}[✓] 输出目录已就绪${NC}"
echo ""

###############################################################################
# Step 1: Tokenizer 扩充 (Phase 2.1)
###############################################################################
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Step 1/2: Tokenizer 扩充${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo -e "${YELLOW}[执行] 运行 expand_tokenizer.py...${NC}"
echo -e "${YELLOW}基础模型: ${BASE_MODEL}${NC}"
echo ""

$PYTHON_ENV src/tokenization/expand_tokenizer.py \
    --base_model "$BASE_MODEL" \
    --registry data/registry/tool_registry.json \
    --output checkpoints/tokenizer_expanded/ \
    --verify

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[✓] Tokenizer 扩充成功${NC}"
    echo -e "${GREEN}    输出目录: checkpoints/tokenizer_expanded/${NC}"
else
    echo -e "${RED}[×] Tokenizer 扩充失败${NC}"
    exit 1
fi
echo ""

###############################################################################
# Step 2: Embedding 初始化 (Phase 2.2)
###############################################################################
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Step 2/2: Embedding 初始化${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo -e "${YELLOW}[执行] 运行 init_embeddings.py...${NC}"
echo -e "${YELLOW}⚠️  注意: 此步骤需要 GPU 和足够的显存${NC}"
echo ""

$PYTHON_ENV src/tokenization/init_embeddings.py \
    --base_model "$BASE_MODEL" \
    --tokenizer checkpoints/tokenizer_expanded/ \
    --registry data/registry/tool_registry.json \
    --output checkpoints/model_initialized/ \
    --verify

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[✓] Embedding 初始化成功${NC}"
    echo -e "${GREEN}    输出目录: checkpoints/model_initialized/${NC}"
else
    echo -e "${RED}[×] Embedding 初始化失败${NC}"
    exit 1
fi
echo ""

###############################################################################
# 模型统计
###############################################################################
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  模型准备完成 - 统计信息${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# 统计 Tokenizer
if [ -d "checkpoints/tokenizer_expanded" ]; then
    echo -e "${GREEN}[Tokenizer]${NC}"
    echo "  - 输出目录: checkpoints/tokenizer_expanded/"
    if [ -f "checkpoints/tokenizer_expanded/tokenizer_config.json" ]; then
        echo "  - 配置文件: ✓"
    fi
fi
echo ""

# 统计模型
if [ -d "checkpoints/model_initialized" ]; then
    echo -e "${GREEN}[模型]${NC}"
    echo "  - 输出目录: checkpoints/model_initialized/"
    if [ -f "checkpoints/model_initialized/embedding_init_info.json" ]; then
        echo "  - 初始化信息:"
        cat checkpoints/model_initialized/embedding_init_info.json | grep -E '(num_new_tokens|num_initialized|embedding_dim)' | sed 's/^/    /'
    fi
fi
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}✓ Phase 2 模型准备流程完成！${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "下一步: 准备训练配置并开始 Stage 1 训练"
echo ""
