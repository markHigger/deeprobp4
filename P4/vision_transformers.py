"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from rob599.p4_helpers import *
from rob599 import Solver
from torch.nn import (Linear, 
                        Softmax, 
                        Dropout, 
                        GELU, 
                        Sequential, 
                        ModuleList,
                        Module, 
                        Parameter
                        )

def hello_vision_transformers():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from vision_transformers.py!')




def patchify(x, patch_size):
    """
        A helper function to convert input image(s) into a set of patches 
        for input to a transformer.

        Input:
        - x: Input data of shape (N, C, H, W)
        - patch_size: Size of desired output patches in terms of pixels
    
        This function requires H and W are multiples of patch_size

        Returns:
        - out: Output data of shape (N, H'*W', ph*pw, C) where H', W', ph, pw are given by
          H' = H // patch_size
          W' = W // patch_size
          ph = patch_size
          pw = patch_size
        """
    N, C, H, W = x.shape
    assert H % patch_size==0, "Height must be divisible by patch_size"
    assert W % patch_size==0, "Width must be divisible by patch_size"
    out = None
    ####################################################################
    # TODO: Implement the convolutional forward pass.                  #
    # Hint: you can use function torch.nn.functional.pad for padding.  #
    # You are NOT allowed to use anything in torch.nn in other places. #
    ####################################################################
    # Replace "pass" statement with your code
    H_prime = H // patch_size
    W_prime = W // patch_size

    patches = x.reshape(N, C, H_prime, patch_size, W_prime, patch_size)
    
    out = patches.permute(0, 2, 4, 3, 5, 1).reshape(N, H_prime * W_prime, patch_size * patch_size * C)
    #####################################################################
    #                          END OF YOUR CODE                         #
    #####################################################################
    
    return out



class Attention(Module):
    """
    Attention Layer: This model takes in batches of input embeddings 
    with shape `(N, *, D_{k,v})` and applies a single layer of scaled dot-product 
    attention in a sequential fashion of the form:

        Attention(q,k,v) = dropout( softmax(q * k.T / sqrt(D)) * v )
    
    See Equation (1) in Attention paper (https://arxiv.org/abs/1706.03762).

    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float=0.1):
        """
        Construct a new attention layer that applies each projection in sequence.

        Input:
        - embed_dim: Size of the expected input dimension
        - hidden_dim: Size of the dimension used within the self-attention
        - dropout: Dropout mask probability
        """
        super().__init__()

        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        #####################################################################
        # TODO: Initialize the parameters for the self attention layer      #
        #                                                                   #
        # Use the imported torch.nn layers (Linear, Softmax, etc.)          #
        # to initialize the projection and attention layers.                #
        #                                                                   #
        # IMPORTANT: initialize the values in the order they are defined    #
        #                                                                   #
        #####################################################################
        # Replace None statements with your code
        self.scale = 1 / torch.sqrt(torch.tensor(embed_dim))
        self.query = torch.nn.Linear(embed_dim, hidden_dim)
        self.key = torch.nn.Linear(embed_dim, hidden_dim)
        self.value = torch.nn.Linear(embed_dim, hidden_dim)
        self.attend = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(hidden_dim,embed_dim)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################
        
    
    def forward(self, query, key, value):
        """
        Calculate the attention output.

        Inputs:
        - query: Input query data, of shape (N, S, D_k)
        - key: Input key data, of shape (N, T, D_k)
        - value: Input value data to be used as the value, of shape (N, T, D_v)
        
        Assumes: 
        - D_k==D_v

        Returns:
        - out: Tensor of shape (N, S, D_v) giving the attention output after final projection
        - attention: Tensor of shape (N, S, D_v) giving the attention probability maps
        """
        out, attention = None, None
        N, S, D = query.shape
        N, T, D = key.shape
        assert key.shape==value.shape

        #####################################################################
        # TODO: Implement scaled dot-product attention                      #
        # Attention(x) = dropout( softmax(q * k.T / sqrt(D)) * v )          #
        #                                                                   #
        # Hint: use torch.matmul to perform a batched matrix-multiply       #
        # Hint: out should apply all layer operations while attention       #
        # should store the intermediate output of softmax(q * k.T / sqrt(D))#
        #####################################################################
        # Replace "pass" statement with your code
        q = self.query(query)  
        k = self.key(key)     
        v = self.value(value)  

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = self.attend(attn_scores) 
        out = torch.matmul(attention, v)
        out = self.dropout(out)
        out = self.out_proj(out)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        return out, attention


class MultiHeadAttention(Module):
    """
    MultiHeadAttention Layer: This model takes in batches of input embeddings 
    with shape `(N, *, D_{k,v})` and applies a single layer of multi-head scaled 
    dot-product attention in a **parallel** fashion of the form:

        MultiHeadAttention(q,k,v) = dropout( softmax((q*W_i^Q) * (k*W_i^K).T / sqrt(D/h)) * v*W_i^V )
                                    for i in range(Num_Heads)
    
    See Section 3.2.2 in Attention paper (https://arxiv.org/abs/1706.03762).

    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        """
        Construct a new multi-head attention layer that applies each projection head in parallel.

        Input:
        - embed_dim: Size of the expected input dimension
        - hidden_dim: Size of the dimension used within the self-attention
        - dropout: Dropout mask probability
        """
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.H=num_heads

        assert embed_dim%num_heads==0, "MHSA requires embedding dimension divisible by number of heads"

        #####################################################################
        # TODO: Initialize the parameters for the self attention layer      #
        #                                                                   #
        # Use the imported torch.nn layers (Linear, Softmax, etc.)          #
        # to initialize the projection and attention layers.                #
        #                                                                   #
        # Note: Your implementation should apply all the projection heads   #
        # in parallel. It may be helpful to write out the intermediate      #
        # shapes on paper before implementing.                              #
        #                                                                   #
        # Note: The autograder will expect the final linear projection to   # 
        # be applied on the concatenated output from each head, and after   #
        # dropout has been applied.                                         #
        #                                                                   #
        # IMPORTANT: initialize the values in the order they are defined    #
        #
        #####################################################################
        # Replace None statements with your code
        
        self.scale = 1 / torch.sqrt(torch.tensor(embed_dim //num_heads))         
        self.query = Linear(embed_dim, hidden_dim)         
        self.key = Linear(embed_dim, hidden_dim )         
        self.value = Linear(embed_dim, hidden_dim)
        self.attend = Softmax(-1)
        self.dropout = Dropout(dropout)
        self.out_proj = Linear(hidden_dim , embed_dim)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the attention output.

        Inputs:
        - query: Input query data, of shape (N, S, D_k)
        - key: Input key data, of shape (N, T, D_k)
        - value: Input value data to be used as the value, of shape (N, T, D_v)
        
        Assumes: 
        - D_k==D_v

        Returns:
        - out: Tensor of shape (N, S, D_v) giving the attention output after final projection
        - attention: Tensor of shape (N, S, D_v) giving the attention probability maps
        """
        out, attention = None, None
        N, S, D = query.shape
        N, T, D = key.shape
        assert key.shape==value.shape
        #####################################################################
        # TODO: Implement multi-head scaled dot-product attention           #
        #                                                                   #
        # Hint: use torch.matmul to perform a batched matrix-multiply       #
        #####################################################################
        # Replace "pass" statement with your code

        Q = self.query(query)  
        K = self.key(key)    
        V = self.value(value)  

        d_k = D // self.num_heads
        
        Q = Q.view(N, S, self.num_heads, d_k).transpose(1,2)
        K = K.view(N, T, self.num_heads, d_k).transpose(1,2)
        V = V.view(N, T, self.num_heads, d_k).transpose(1,2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = self.attend(attn_scores) 
        heads = torch.matmul(attention, V)
        heads = self.dropout(heads)
        heads_cat = heads.transpose(1,2).contiguous().view(N, S, D)
        print(heads_cat.shape)
        out = self.out_proj(heads_cat)



        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        return out, attention




class LayerNorm_fn(object):


    @staticmethod
    def forward(x, gamma, beta, ln_param):
        """
        Forward pass for layer normalization.

        During both training and testing, the mean and variance
        are computed from the features of each data point.

        Inputs:
        - x: Data of shape (*, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - ln_param: Dictionary with the following keys:
          - eps: Constant for numeric stability

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        eps = ln_param.get('eps', 1e-5)

        D = x.shape[-1]

        out, cache = None, None

        ##################################################################
        # TODO: Implement the training-time forward pass for layer norm. #
        # Use minibatch statistics to compute the mean and variance, use #
        # these statistics to normalize the incoming data, and scale and #
        # shift the normalized data using gamma and beta.                #
        #                                                                #
        # You should store the output in the variable out.               #
        # Any intermediates that you need for the backward pass should   #
        # be stored in the cache variable.                               #
        # Referencing the original paper                                 #
        # (https://arxiv.org/abs/1607.06450) might prove to be helpful.  #
        # Note: Unlike our BatchNorm implementation, this LayerNorm      #
        # should accept input of shape (*, D)                            #
        ##################################################################
        # Replace "pass" statement with your code
        mean = torch.mean(x, -1, keepdim=True)
        var = torch.var(x, -1, keepdim=True)
        diff = (x - mean) 
        vareps = var + eps
        sqrt = torch.sqrt(vareps)
        isqrt = 1 / sqrt
        div = isqrt * diff
        out = gamma * div + beta

        cache = x, gamma, beta, mean, var, diff, vareps, sqrt, isqrt, div

        ################################################################
        #                           END OF YOUR CODE                   #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for layer normalization.

        For this implementation, you should write out a
        computation graph for layer normalization to understand the
        gradients flow.

        Inputs:
        - dout: Upstream derivatives, of shape (*, D)
        - cache: Variable of intermediates from LayerNorm.forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (*, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        x, gamma, beta, mean, var, diff, vareps, sqrt, isqrt, div = cache
        D = x.shape[-1]
        #####################################################################
        # TODO: Implement the backward pass for layer normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1607.06450) #
        # might prove to be helpful.                                        #
        #                                                                   #
        # Note: Unlike our BatchNorm implementation, this LayerNorm should  #
        # accept input of shape (*, D)                                      #
        #####################################################################
        # Replace "pass" statement with your code
        # dsigma = dout * mean * gamma / var**2

        dgamma = torch.sum(torch.sum(dout * div, dim=0), dim=0)

        dbeta = torch.sum(torch.sum(dout,dim=0), dim=0)

        dout_norm = dout * gamma

        dvar_full = dout_norm * diff * -0.5 * vareps ** -1.5
        dvar = torch.sum(torch.sum(dvar_full, dim=0))
        dmean1 = torch.sum(torch.sum(dout_norm *-1 * isqrt, dim=0))
        dmean2 = dvar * -2 * diff
        dmean = dmean1 + dmean2

        dx = dout_norm*isqrt + dvar * 2 * diff/D + dmean/D

        
        #                      END OF YOUR CODE                         #
        #################################################################


        return dx, dgamma, dbeta



class TransformerEncoder(Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        use_mhsa: bool=True,
        num_heads: int=1,
        dropout: float=0.1
        ):
        super().__init__()

        assert num_heads>0, "Number of heads must be greater than zero"
        assert use_mhsa or num_heads==1, "Number of heads cannot be greater than one for Attention layer"

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_mhsa = use_mhsa
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = None
        self.norm2 = None
        self.attn = None
        self.mlp = None

        #####################################################################
        # TODO: Initialize the components of a transformer encoder layer    #
        #                                                                   #
        # The layer should have the following architecture:                 #
        # {layernorm1 - (multihead)self-attention - layernorm2 - mlp        #
        # with residual connections from input to attention output and from #
        # pre-layernorm2 to mlp output                                      #
        #                                                                   #
        # Referring to the original paper                                   #
        # (https://arxiv.org/abs/2010.11929) may be useful                  #
        #####################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################


    def forward(self, x):
        out, attention = None, None

        #####################################################################
        # TODO: implement forward pass of transformer encoder layer         #
        #                                                                   #
        # The layer should have the following architecture:                 #
        # {layernorm1 - (multihead)attention - layernorm2 - mlp             #
        # with residual connections from input to attention output and from #
        # pre-layernorm2 to mlp output                                      #
        #                                                                   #
        # Referring to the original paper                                   #
        # (https://arxiv.org/abs/2010.11929) may be useful                  #
        #####################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return out, attention


class VisionTransformer(Module):

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_mhsa: bool=True,
        num_heads: int=1,
        input_dims: tuple=(3,32,32),
        num_classes: int=10,
        patch_size: int=4,
        dropout: float=0.1,
        dtype: torch.dtype=torch.float,
        loss_fn: str='softmax_loss',
        ):
        super().__init__()

        assert num_heads>0, "Number of heads must be greater than zero"
        assert use_mhsa or num_heads==1, "Number of heads cannot be greater than one for Attention layer"

        C, H, W = input_dims
        assert H%patch_size==0, "Height must be divisible by patch_size"
        assert W%patch_size==0, "Width must be divisible by patch_size"

        self.num_patches = (H//patch_size) * (W//patch_size)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_mhsa = use_mhsa
        self.num_heads = num_heads
        self.input_dims = input_dims
        self.num_clases = num_classes
        self.patch_size = patch_size
        self.dropout = dropout
        self.dtype = dtype
        self.loss_fn = loss_fn

        self.cls_token = Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embedding = Parameter(torch.randn(1, 1+self.num_patches, self.embed_dim))

        self.in_proj = None
        self.transformer_layers = ModuleList()
        self.mlp_head = None

        #####################################################################
        # TODO: Initialize the vision transformer architecture layers.      #
        #                                                                   #
        # Referring to the original paper                                   #
        # (https://arxiv.org/abs/2010.11929) may be useful                  #
        #####################################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        self.params = self.parameters()

    def save(self, path):
        checkpoint = {
        'num_patches': self.num_patches,
        'embed_dim': self.embed_dim,
        'hidden_dim': self.hidden_dim,
        'num_layers': self.num_layers,
        'use_mhsa': self.use_mhsa,
        'num_heads': self.num_heads,
        'input_dims': self.input_dims,
        'num_clases': self.num_clases,
        'patch_size': self.patch_size,
        'dropout': self.dropout,
        'dtype': self.dtype,
        'loss_fn': self.loss_fn,
        'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.num_patches = checkpoint['num_patches']
        self.embed_dim = checkpoint['embed_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.use_mhsa = checkpoint['use_mhsa']
        self.num_heads = checkpoint['num_heads']
        self.input_dims = checkpoint['input_dims']
        self.num_clases = checkpoint['num_clases']
        self.patch_size = checkpoint['patch_size']
        self.dropout = checkpoint['dropout']
        self.dtype = checkpoint['dtype']
        self.loss_fn = checkpoint['loss_fn']

        self.load_state_dict(checkpoint['state_dict'])

        print("load checkpoint file: {}".format(path))
    
    def forward(self, x):
        """
        Calculate the ViT model output.

        Inputs:
        - x: Input image data, of shape (N, D, H, W)

        Returns:
        - out_cls: Tensor of shape (N, C) giving the attention output after final projection
        - out_tokens: Tensor of shape (N, S, D_v) giving the attention tokens of last transformer layer
        - out_attention_maps: List of tensors giving the attention probability tensors from *every* transformer layer
        """
        out_cls, out_tokens, out_attention_maps = None, None, None
        N, C, H, W = x.shape

        ###########################################################
        # TODO: Implement the vision transformer forward pass.    #
        #                                                         #
        # Referring to the original paper                         #
        # (https://arxiv.org/abs/2010.11929) may be useful        #
        ###########################################################
        # Replace "pass" statement with your code
        pass
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return out_cls, out_tokens, out_attention_maps


    def softmax_loss(self, X, y=None):
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        scores, tokens, attention = self.forward(X)

        if y is None:
            return scores

        loss, _ = softmax_loss(scores, y)

        return loss

    def loss(self, X, y=None):
        if self.loss_fn=='softmax_loss':
            return self.softmax_loss(X, y)
        else:
            raise NotImplementedError



def find_overfit_parameters():
    embed_dim = 1   # Experiment with this!
    hidden_dim = 1  # Experiment with this!
    num_layers = 1 # Experiment with this!
    use_mhsa = False # Experiment with this!
    num_heads = 1 # Experiment with this!
    weight_decay = 1e-10   # Experiment with this!
    learning_rate = 1e-10  # Experiment with this!
    ###########################################################
    # TODO: Change the hyperparameters defined above so your  #
    # model achieves 100% training accuracy within 50 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return embed_dim, hidden_dim, num_layers, use_mhsa, num_heads, weight_decay, learning_rate


def create_vit_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best VisionTransformer that you can   #
    # on PROPS within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    pass
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver





class LayerNorm(Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.gamma = Parameter(torch.ones(self.embed_dim))
        self.beta = Parameter(torch.zeros(self.embed_dim))

    def forward(self, x):
        return LayerNorm_fn.forward(x, self.gamma, self.beta, {})[0]
