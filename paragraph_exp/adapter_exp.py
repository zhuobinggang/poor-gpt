from sector import *
from typing import List, Optional, Tuple, Union
from common import *
from transformers import apply_chunking_to_forward

class Adapter(nn.Module):
    def __init__(self, in_channel, hidden_channel, name = None):
        super().__init__()
        self.down = nn.Linear(in_channel, hidden_channel)
        self.gelu = nn.GELU()
        self.up = nn.Linear(hidden_channel, in_channel)
        self.name = name if name is not None else 'Default'
        self.cuda()
    def forward(self, x):
        h = self.down(x) 
        h = self.gelu(h)
        h = self.up(h)
        return x + h

def inject_feed_forward_chunk(self, adapter, attention_output):
    intermediate_output = self.intermediate(attention_output) # (1, 178, 3072)
    # NOTE: 使用Adapter
    intermediate_output = intermediate_output + adapter(intermediate_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output

def inject_output_forward(self, adapter, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states) # (1, 178, 768)
    # NOTE: 使用Adapter
    hidden_states = hidden_states + adapter(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

def c1(layer, adapter, idx):
    def closure(*args):
        return inject_feed_forward_chunk(layer, adapter, *args)
    return closure

def c2(output, adapter, idx):
    def closure(*args):
        return inject_output_forward(output, adapter, *args)
    return closure

class Sector_Adapter(Sector):
    def get_should_update(self):
        freeze(self) # 冻结参数
        adapters = []
        for idx, layer in enumerate(self.bert.encoder.layer):
            print(f'Initing {idx} layer adapter')
            adapter1 = Adapter(3072, 8, name = f'Layer {idx} 1st Adapter')
            adapters.append(adapter1)
            # adapters.append(layer.adapter)
            # 替换第1层前馈函数
            layer.feed_forward_chunk = c1(layer, adapter1, idx)
            # 替换第2层前馈函数
            output = layer.output
            adapter2 = Adapter(768, 8, name = f'Layer {idx} 2nd Adapter')
            adapters.append(adapter2)
            output.forward = c2(output, adapter2, idx)
        self.adapters = nn.ModuleList(adapters)
        # 解冻adapters
        for param in self.adapters.parameters():
            param.requires_grad = True
        return chain(self.adapters.parameters())

def script():
    m = Sector_Adapter()
    ld = loader.news.train()
    ss, ls = ld[99]
    out, tar = m(ss,ls)
    loss = m.loss(out, tar)
    loss.backward()
    check_grad(m.adapters[0])
    check_grad(m.adapters[5])
    check_grad(m.adapters[11])
    

def check_grad(adapter):
    return adapter.down.weight.grad


def script():
    m = ModelWrapper(Sector_Adapter(lr = 1e-4))
    train_save_eval_plot(m, 'Sector_Adapter', batch_size = 32, check_step = 500, total_step = 20000)

