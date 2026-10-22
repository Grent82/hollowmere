# Start llama.cpp's HTTP server for Hollowmere (config.toml expects http://127.0.0.1:8080/v1).
# Stop any other llama-cli / llama-server first: they would share the same 6 GB GPU.
#
# The model has to fit in RAM + VRAM. The 75 GB Qwen3.8-Flash-Next Q2_K does not
# (15 GB RAM + 6 GB VRAM): it streams from disk at about 1 token/s. Qwen3-8B Q4_K_M
# (4.7 GB) fits entirely on the GPU.

$LlamaDir = "D:\llama_cpp"
$Model    = "D:\Models\Qwen3-8B-Q4_K_M.gguf"
$Parallel = 2        # keep equal to [llm] max_concurrency in config.toml
$Context  = 8192     # split between the slots: 2 x 4096 tokens. A prompt (~2.2k) plus a reply (up to
                     # 1.5k) must fit in one slot, so more slots need more context, and 6 GB VRAM
                     # has no room for more. q8_0 keeps this cache at about 0.6 GB.

# ggml-cuda.dll needs cuBLAS 13, which this llama.cpp download lacks; without it the
# server silently runs on the CPU. The Strata bundle ships the two DLLs.
$CudaDlls = "D:\Strata\engine-bundle\0.1.46-windows-x86_64-cuda-cuda13"
if (-not (Test-Path "$LlamaDir\cublas64_13.dll")) { $env:PATH = "$CudaDlls;$env:PATH" }

& "$LlamaDir\llama-server.exe" `
    -m $Model `
    --host 127.0.0.1 --port 8080 `
    --jinja `
    --parallel $Parallel --ctx-size $Context `
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0
