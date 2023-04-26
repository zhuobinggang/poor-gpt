from sector import *
from typing import List, Optional, Tuple, Union
from common import *
from transformers import apply_chunking_to_forward

def normaled_linear(in_channel, hidden_channel):
    lin = nn.Linear(in_channel, hidden_channel)
    nn.init.normal_(lin.weight, std = 1e-3)
    nn.init.zeros_(lin.bias)
    return lin

class Adapter(nn.Module):
    def __init__(self, in_channel, hidden_channel, name = None):
        super().__init__()
        self.down = normaled_linear(in_channel, hidden_channel)
        self.gelu = nn.GELU()
        self.up = normaled_linear(hidden_channel, in_channel)
        self.name = name if name is not None else 'Default'
        self.cuda()
    def forward(self, x):
        h = self.down(x) 
        h = self.gelu(h)
        h = self.up(h)
        return x + h

def inject_output_FFW(self, adapter, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states) # (1, 178, 768)
    hidden_states = self.dropout(hidden_states)
    # NOTE: 使用Adapter在dropout之后
    hidden_states = adapter(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

def closure_create(layer, adapter, func):
    def closure(*args):
        return func(layer, adapter, *args)
    return closure

def inject_attention_output_FFW(self, adapter, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    # NOTE: 使用Adapter在dropout之后
    hidden_states = adapter(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class Sector_Adapter(Sector):
    def __init__(self, hidden_channel = 8, lr=1e-3):
        self.hidden_channel = hidden_channel
        super().__init__(lr)
    def init_adapters(self):
        adapters = []
        for idx, layer in enumerate(self.bert.encoder.layer):
            print(f'Initing {idx} layer adapter')
            adapter1 = Adapter(768, self.hidden_channel, name = f'Layer {idx} 1st Adapter')
            adapters.append(adapter1)
            # adapters.append(layer.adapter)
            # 替换第1层前馈函数
            # layer.feed_forward_chunk = closure_create(layer, adapter1, inject_feed_forward_chunk)
            layer.attention.output.forward = closure_create(
                    layer.attention.output, 
                    adapter1, 
                    inject_attention_output_FFW)
            # 替换第2层前馈函数
            adapter2 = Adapter(768, self.hidden_channel, name = f'Layer {idx} 2nd Adapter')
            adapters.append(adapter2)
            layer.output.forward = closure_create(layer.output, adapter2, inject_output_FFW)
        self.adapters = nn.ModuleList(adapters)
    def init_classfier(self):
        self.classifier = nn.Sequential(
            # Adapter(self.bert_size, self.hidden_channel),
            nn.Linear(self.bert_size, 2),
        )
    def init_hook(self):
        freeze(self.bert)
        self.init_adapters()
        self.init_classfier()
    def get_should_update(self):
        print('Sector_Adapter ONLY UPDATE ADAPTERS & CLASSIFIER')
        # 解冻adapters + classifier
        for param in chain(self.adapters.parameters(), self.classifier.parameters()):
            param.requires_grad = True
        return chain(self.adapters.parameters(), self.classifier.parameters())
    def reset_opter(self, lr):
        self.lr = lr
        self.opter = t.optim.AdamW(self.get_should_update(), lr)


class Sector_Adapter_Fulltune(Sector_Adapter):
    def get_should_update(self):
        print('Sector_Adapter_Fulltune UPDATE BERT, ADAPTERS, CLASSIFIER')
        for param in chain(self.bert.parameters(), self.adapters.parameters(), self.classifier.parameters()):
            param.requires_grad = True
        return chain(self.bert.parameters(), self.adapters.parameters(), self.classifier.parameters())

def script():
    m = Sector_Adapter()
    ld = loader.news.train()
    ss, ls = ld[99]
    out, tar = m(ss,ls)
    loss = m.loss(out, tar)
    loss.backward()
    check_gradient(m)

def script():
    m = ModelWrapper(Sector_Adapter_Fulltune(lr = 2e-5))
    train_save_eval_plot(m, 'Sector_Adapter_Fulltune', batch_size = 32, check_step = 500, total_step = 10000)

