import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, dk, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert (dk % n_heads == 0), \
            "dk must be divisible by n_heads"
        self.dk = dk
        self.n_heads = n_heads
        self.head_dim = dk // n_heads
        # torch.nn.Linear(inp_dim,out_dim)
        self.W_query = torch.nn.Linear(emb_dim, dk, bias=qkv_bias)
        self.W_key = torch.nn.Linear(emb_dim, dk, bias=qkv_bias)
        self.W_value = torch.nn.Linear(emb_dim, dk, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(dk, emb_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                      diagonal=1)
        )

    def forward(self,x):
        B, T, emb_dim = x.shape
        keys = self.W_key(x) # B,T,dk
        queries = self.W_query(x)
        values = self.W_value(x)

        # Expanding the last dimenstion from dk to n_head * head_dim
        # dk =  n_head * head_dim
        keys = keys.view(B, T, self.n_heads, self.head_dim) # (B,T,n_heads,head_dim)
        values = values.view(B, T, self.n_heads, self.head_dim)
        queries = queries.view(
            B, T, self.n_heads, self.head_dim
        )

        # TRANSPOSE from shape (B,T,n_heads,head_dim) to (B,n_heads,T,head_dim)
        keys = keys.transpose(1,2) # (B,n_heads,T,head_dim) 
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # (B,n_heads,T,head_dim) @ (B,n_heads,head_dim,T) ---> (B,n_heads,T,T)
        attn_scores = queries @ keys.transpose(2,3) # (B, n_heads, T, T)
        mask_bool = self.mask.bool()[:T,:T]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5 , dim=-1
        )
        attn_weights = self.dropout(attn_weights) # (B, n_heads, T, T)
        self.last_attn_weights = attn_weights

        # (B, n_heads, T, T) @ (B,n_heads,T,head_dim) --> (B, n_head, T, head_dim)
        context_vector = attn_weights @ values # (B, n_head, T, head_dim)
        context_vector = context_vector.transpose(1,2) # (B, T, n_head, head_dim)
        context_vector = context_vector.contiguous().view(
            B, T , self.dk                # (B, T, n_head, head_dim) ---> (B, T, dk)
        )

        context_vector = self.out_proj(context_vector) # (B, T, emb_dim)
        return context_vector
        
class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"] , 4*cfg["emb_dim"]),
            GELU(),
            torch.nn.Linear(4*cfg["emb_dim"] , cfg["emb_dim"])
        )
    def forward(self , x):
        return self.layers(x)

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        z = self.scale * norm_x + self.shift
        return z

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.MHA = MultiHeadAttention(
            emb_dim = cfg["emb_dim"],
            dk = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            n_heads = cfg["n_heads"],
            qkv_bias = cfg["kqv_bias"] 
        )
        self.ffw = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        inp = x
        x = self.norm1(x)
        x = self.MHA(x)
        x = self.dropout(x)
        x = x + inp

        inp = x
        x = self.norm2(x)
        x = self.ffw(x)
        x = self.dropout(x)
        x = inp + x

        return x     


class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])
        self.transformer_block = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        # Weight tying: share embedding weights with output projection
        # Saves ~19M params (vocab_size x emb_dim) and improves training
        self.out_head.weight = self.tok_emb.weight

    def forward(self, x):
        batch_size, context_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(context_len, device=x.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def createGPTModel(cfg):
    return GPTModel(cfg)