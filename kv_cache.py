import torch
from iree.turbine import aot

class SimpleCache(torch.nn.Module):
    def __init__(self, cache_size, dtype=torch.float32):
        super().__init__()
        self.register_buffer("cache", torch.zeros(cache_size, dtype=dtype))

    def forward(self, input_pos, values):
        # input_pos: [S], values: [S, D]
        assert input_pos.shape[0] == values.shape[0]

        # Update the buffer at specified positions
        cache = torch.ops.aten.index_put_(self.cache, [input_pos], values)

        return cache

# Minimal Parameters
cache_size = 10
dtype = torch.float32

simple_cache = SimpleCache(cache_size, dtype)

# Inputs
S = 5
input_pos = torch.arange(S)
values = torch.randn(S, dtype=dtype)

# Example inputs for torch.export.export
initialize_inputs = (cache_size,)
inputs = (input_pos, values)

# Export the model using torch.export.export
fx_imported = torch.export.export(SimpleCache(*initialize_inputs), args=inputs)
print(fx_imported)

# Attempt to AOT export (note: this is just for illustrative purposes)
aot.export(fx_imported)
