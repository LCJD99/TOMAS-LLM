import argparse

import torch

from src.v1.model.tool_encoder import SemanticEncoder
from src.v1.utils.config import load_config, merge_config_with_args
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_semantic_encoder(config: dict):

    semantic_encoder = SemanticEncoder(
        tool_registry_path="data/tool_registry/tools.json",
        device='cuda'
    )
    semantic_encoder.eval()
    tool_embed = semantic_encoder(torch.tensor(0, device='cuda')) #shape = (3584)
    print("Tool Embed Shape:", tool_embed.shape)

    model_name = "Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    embed_layer = model.get_input_embeddings() 

    prompt1 = "The tool message show as below:\n "
    input_ids1 = tokenizer(prompt1, return_tensors="pt").input_ids.to(model.device) 
    breakpoint()
    inputs_embeds1 = embed_layer(input_ids1) # shape = (1, seq_len, embed_dim)

    prompt2 = "\n describe the function of the tool and give an example to use it.\n Answer:"
    input_ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to(model.device) 
    inputs_embeds2 = embed_layer(input_ids2) # shape = (1, seq_len, embed_dim)

    # 3. 手动循环生成
    max_new_tokens = 20
    tool_embed = tool_embed.unsqueeze(0).unsqueeze(0).to(model.device).to(torch.bfloat16)  # shape = (1, 1, embed_dim)
    inputs_embeds = torch.cat([inputs_embeds1, tool_embed], dim=1)  # 在序列末尾添加 tool embed
    inputs_embeds = torch.cat([inputs_embeds, inputs_embeds2], dim=1)  # 在序列末尾添加 prompt2

    generated_ids = torch.cat([input_ids1, input_ids2], dim=-1)

    # 关闭梯度计算以节省显存
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 向前传播获取 Logits
            # outputs.logits 的维度是 [batch_size, sequence_length, vocab_size]
            outputs = model(inputs_embeds=inputs_embeds)
            
            # 我们只需要序列中最后一个 Token 产生的 Logits 来预测下一个 Token
            next_token_logits = outputs.logits[:, -1, :]
            
            # 基础处理：贪心搜索 (Greedy Search)
            # 也可以在这里加入 Temperature 或 Top-P 的逻辑
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            next_token_embeds = embed_layer(next_token_id)
            
            # 将新生成的 ID 拼接回序列
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            inputs_embeds = torch.cat([inputs_embeds, next_token_embeds], dim=1)

            token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(f"Generated Token: {token}")
            
            # 如果检测到 EOS Token，提前停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # 4. 解码输出
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Final Output: {result}")
