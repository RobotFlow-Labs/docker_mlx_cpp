# Building a Mac GPU Model Runner That Any Docker Container Can Use

## What is actually possible on macOS

A “NVIDIA-style” setup—where arbitrary Linux containers get direct GPU device access (e.g., `--gpus all`)—does not translate cleanly to macOS. Docker Desktop runs Linux containers behind a VM boundary, and Docker’s own Model Runner team has described GPU passthrough across that boundary as “impossible or very flaky,” which is why they opted to run inference outside the VM and proxy calls back into containers. citeturn5view0

For Apple Silicon specifically, Metal GPU access requires direct hardware access. Docker’s February 26, 2026 announcement for the `vllm-metal` backend states plainly that “there is no GPU passthrough for Metal in containers,” so the backend runs natively on the host. citeturn4view0

So the feasible pattern on macOS is:

- Run GPU-accelerated inference as a **host-side service** on macOS (Metal).
- Let **any container** call that service over HTTP (loopback / Docker-internal networking).
- Standardize the interface with an OpenAI-compatible API (and optionally other API shapes), so existing tools “just work.”

This is exactly the architecture of entity["company","Docker","container platform"]’s Docker Model Runner design for Docker Desktop: host-executed inference engines, container-accessible APIs. citeturn5view0turn14view0

## What Docker Model Runner provides on Apple Silicon

Docker Model Runner (DMR) is designed to “pull, run, and serve” AI models locally from Docker Hub, any OCI-compliant registry, or entity["company","Hugging Face","ml model hub"], and to expose programmatic endpoints that are compatible with common client ecosystems. citeturn3search27turn14view0

### API surface you can build around

DMR provides multiple API formats from the same service:

- OpenAI-compatible endpoints at paths like `/engines/v1/chat/completions`, `/engines/v1/embeddings`, etc. citeturn14view0  
- Anthropic-compatible messages endpoints under `/anthropic/v1/messages`. citeturn16view1  
- Ollama-compatible endpoints as well. citeturn16view1  

From a **container**, the canonical base URL on Docker Desktop is `http://model-runner.docker.internal` (and then append `/engines/v1/...` for OpenAI-style calls). citeturn14view0  
From the **host**, the TCP base is `http://localhost:12434` (if host TCP access is enabled). citeturn14view0turn1view2

### Apple Silicon backends that matter for your goal

On Apple Silicon, DMR’s default inference engine is `llama.cpp`, using GGUF model files, and the official docs note that GPU support on Apple Silicon uses Metal with “automatic GPU acceleration.” citeturn1view1

In addition, as of Feb 26, 2026, Docker announced `vllm-metal`, which “brings vLLM inference to macOS using Apple Silicon’s Metal GPU.” It runs MLX-format models through vLLM with the same OpenAI-compatible API (and Anthropic-compatible API for tools like “Claude Code”), aiming to keep the same `docker model` workflow across platforms. citeturn4view1turn4view0

Critical implementation detail: the backend runs on the host, because Metal is not available inside containers; Docker Model Runner installs it by pulling an image that contains a self-contained Python environment and dependencies, then extracting it to `~/.docker/model-runner/vllm-metal/`, verifying it by importing the module, and launching a host-side server process on demand. citeturn4view0

If you also want to understand what `vllm-metal` itself supports and how it’s tuned: the upstream repo describes it as a plugin enabling vLLM on Apple Silicon using MLX as the primary compute backend, with optional experimental paged attention controlled by environment variables. citeturn17view0

## Reference architecture for “GPU from any container” on Mac

The architecture that matches your requirement (“a docker that connects to the model runner to use the GPU of the mac from any docker”) is:

- **Host layer (macOS):** Docker Desktop + DMR enabled. Inference engines run as host processes (Metal). citeturn5view0turn1view1turn4view0  
- **Container layer (Linux containers):** Any app container sends requests to DMR via `model-runner.docker.internal` (no GPU inside the container; the GPU work occurs on the host). citeturn14view0turn5view0  

A small but important nuance: for calling from containers you **do not** necessarily need host-side TCP enabled; Docker Desktop provides the special `model-runner.docker.internal` endpoint for containers. Host-side TCP is needed when host processes (outside Docker) should call via `localhost:12434`. citeturn14view0turn1view2turn5view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Docker Model Runner architecture diagram com.docker.backend model-runner.docker.internal","vllm-metal architecture diagram vLLM Metal Plugin MLX PyTorch", "Apple Silicon Metal GPU inference llama.cpp diagram","Docker Model Runner Models tab requests logs screenshot"],"num_per_query":1}

### Minimal proof that *any* container can reach host GPU inference

Below is a practical sequence that demonstrates the “GPU-from-any-container” pattern using DMR’s OpenAI-compatible API.

**Enable DMR + (optional) host TCP**
- Docker Desktop’s “AI” tab can enable Docker Model Runner and optionally “host-side TCP support.” citeturn1view2  
- The API docs explicitly call out that host TCP must be enabled to use `localhost:12434`. citeturn14view0  

**Pull and run a model**
```bash
docker model pull ai/smollm2:360M-Q4_K_M
docker model run ai/smollm2:360M-Q4_K_M "Hello from DMR"
```
DMR’s `docker model` workflow pulls and runs local models and serves them via the DMR API layer. citeturn1view2turn14view0

**Call it from *inside another container***
```bash
docker run --rm curlimages/curl:8.5.0 \
  -s http://model-runner.docker.internal/engines/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai/smollm2:360M-Q4_K_M",
    "messages": [{"role":"user","content":"Say hi from a container."}],
    "max_tokens": 64
  }'
```
This works because Docker Desktop publishes the DMR service to containers at `model-runner.docker.internal`, and the OpenAI-compatible endpoint is under `/engines/v1/chat/completions`. citeturn14view0

### Turning this into a reusable “connector container”

Even though `model-runner.docker.internal` is already reachable from containers, it’s often useful to introduce a dedicated “connector” container for:

- A stable internal DNS name inside your Compose network (e.g., `llm-gateway`), so apps never hardcode Docker Desktop’s special hostname.
- Centralized auth/rate-limiting/logging.
- Optional exposure to your LAN/VPN (while DMR itself may be loopback-only on the host side).

Conceptually:

- `llm-gateway` container listens on `0.0.0.0:PORT` (inside Docker).
- It proxies to `http://model-runner.docker.internal/engines/v1` (upstream).
- All other app containers call `http://llm-gateway:PORT`.

This is an architectural extension you build *on top of* the documented base URLs and endpoints. citeturn14view0

## Packaging and distributing models like “NVIDIA-style Docker assets”

What DMR distributes are “models as OCI artifacts,” not traditional runnable container images. Docker’s rationale is that packaging models as OCI artifacts lets teams reuse existing registry workflows, policy controls, and distribution infrastructure instead of inventing a new model delivery toolchain. citeturn12view0

### The model artifact format in brief

Docker’s model specification indicates:

- A model artifact uses an OCI-style manifest (`application/vnd.oci.image.manifest.v1+json`) but its layers represent model files (not a runnable filesystem). citeturn9view2  
- Layers “SHOULD contain the contents of a single file” and “SHOULD NOT be compressed.” citeturn9view2  
- A model config JSON (`application/vnd.docker.ai.model.config.v0.1+json`) records metadata and a file list; for GGUF it includes fields like architecture, parameter count, and quantization in an example. citeturn11view1turn9view2  

Separately, Docker’s OCI-artifacts blog explains an operational implication: at runtime DMR does not look for a GGUF file by a filesystem path; it identifies the desired file by its media type (e.g., `application/vnd.docker.ai.gguf.v3`) and fetches it from the model store. citeturn12view0

### How you package models into OCI artifacts

`docker model package` is the core CLI command. It can package:

- A GGUF file (`--gguf`)
- A Safetensors directory (`--safetensors-dir`)
- A Diffusers unified archive (`--dduf`)
- Or repackage an existing model (`--from`) citeturn1view3

Important packaging behavior details from the CLI reference:

- Sharded GGUF models: point `--gguf` at the first shard; all shards in the same directory with an indexed naming convention are discovered and packaged together. citeturn1view3  
- Safetensors: `--safetensors-dir` packages all files under the directory (including nested), and “each file is packaged as a separate OCI layer.” citeturn1view3  
- Optional extras include a license file (`--license`), a multimodal projector (`--mmproj`), and a chat template in Jinja format (`--chat-template`). citeturn1view3turn9view2  

Example:
```bash
docker model package \
  --gguf ./my-model.Q4_K_M.gguf \
  --context-size 8192 \
  --license ./LICENSE \
  --push myorg/my-model:8B-Q4_K_M
```
The CLI indicates that by default the packaged artifact is loaded into the local model store, and `--push` publishes it to a registry. citeturn1view3

### Choosing model formats for a Mac “GPU runner catalog”

For a Mac-focused catalog, the choice of format ties directly to the backend:

- **GGUF + llama.cpp** is the most “universal” Mac path, and official docs position llama.cpp as the default engine; Apple Silicon gets Metal acceleration automatically. citeturn1view1turn5view0  
- **MLX Safetensors + vllm-metal** is the Mac path when you want vLLM-style serving and you’re willing to use MLX-optimized models. Docker’s announcement states vllm-metal works with “safetensors models in MLX format,” and that DMR auto-routes MLX models to vllm-metal once installed (fallback to an MLX backend otherwise). citeturn4view0  

A practical implication for your “NVIDIA-like” experience: you can publish and version model artifacts in a registry the same way you would publish container images, but you should expect multiple “variants” per architecture depending on which backend you want (GGUF quantizations vs MLX-specific builds). citeturn12view0turn4view0turn1view3

## Mega plan to build a Mac GPU model-runner platform for all containers

This plan is based on how DMR is actually architected and exposed (host-executed inference + container-callable API), and it’s structured so you can evolve from a single-dev laptop to a team-wide platform while keeping the same API contracts. citeturn5view0turn14view0turn12view0

### Platform definition and success criteria

Define your platform as four contracts:

- **Inference contract:** OpenAI-compatible API at `/engines/v1/*` (your apps speak this). citeturn14view0  
- **Model distribution contract:** models are OCI artifacts in a registry; your platform can `pull`, `tag`, `push`, `package`. citeturn1view2turn12view0turn1view3  
- **Backend contract:** on macOS, inference runs host-side (Metal); containers do not get direct Metal GPU access. citeturn4view0turn5view0  
- **Developer UX contract:** “any container can call the local model” via either:
  - `model-runner.docker.internal` (simple), or
  - injected variables like `LLM_URL`/`LLM_MODEL` via Compose models (structured). citeturn14view0turn15view1turn15view0  

Success looks like:
- You can bring up any Compose stack and it can talk to local GPU inference **without** custom per-project networking hacks.
- Your organization can publish “known good” model artifacts with tags and metadata.
- You can swap models/backends without changing app code (only config).

### Foundation build

Install/enable:

- Enable Docker Model Runner in Docker Desktop (Settings → AI tab) and, if you want host-side access, enable host-side TCP support and choose a port. citeturn1view2turn14view0  
- For vLLM on macOS, Docker’s announcement says to update to Docker Desktop 4.62+ and install the backend via `docker model install-runner --backend vllm`. citeturn4view0turn4view1  

Verification checklist (pragmatic, not theoretical):
- `docker model version` works. citeturn1view2turn7view0  
- A model can be pulled from Docker Hub or Hugging Face reference formats described in the “get started” docs. citeturn1view2  
- The API endpoint is reachable:
  - from a container at `http://model-runner.docker.internal/engines/v1/...` citeturn14view0  
  - from host at `http://localhost:12434/engines/v1/...` if TCP enabled citeturn14view0turn1view2  

### Model catalog build

Create an internal “model catalog” policy:

- Decide naming/tagging conventions (e.g., `myorg/<family>:<params>-<quant>-<context>`).
- Require license inclusion where applicable: `docker model package` supports attaching explicit license files. citeturn1view3turn9view2  
- Record metadata: the model config JSON format captures packaging format and file list; GGUF-specific fields should correspond to GGUF standardized key-value pairs. citeturn11view1  

Implement two pipelines:

**GGUF pipeline (llama.cpp on Mac)**
1. Acquire or convert your model to GGUF (outside scope of DMR docs; DMR assumes you already have GGUF). citeturn12view1  
2. Package: `docker model package --gguf ... --context-size ... --push ...`. citeturn1view3turn1view1  
3. Validate:
   - Pull from registry into a clean machine profile.
   - Run a smoke test prompt via `/engines/v1/chat/completions`. citeturn14view0  

**MLX/vLLM pipeline (vllm-metal on Mac)**
1. Select MLX-format models (the Docker announcement references MLX models published by mlx-community). citeturn4view0  
2. Ensure vllm-metal backend is installed and verify routing behavior (DMR “automatically routes MLX models to vllm-metal when the backend is installed”). citeturn4view0  
3. Capture tuning knobs if needed: upstream `vllm-metal` exposes controls like `VLLM_METAL_USE_PAGED_ATTENTION` and memory fraction variables (useful later for your platform-level tuning). citeturn17view0  

### Standard “container connector” pattern

You want a repeatable way for *any* container to consume inference. Build around DMR’s documented base URLs and the OpenAI SDK sample pattern.

**Option that requires the least glue: set base URL directly**
- In containers, set `OPENAI_BASE_URL` (or whatever your app expects) to `http://model-runner.docker.internal/engines/v1`. citeturn14view0  
- Use any API key string if required by SDK validation; DMR doesn’t require an API key and ignores the Authorization header. citeturn16view1  

Example Python client (inside a container):
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

resp = client.chat.completions.create(
    model=os.environ["OPENAI_MODEL"],
    messages=[{"role": "user", "content": "Hello from a container."}],
    max_tokens=128,
)

print(resp.choices[0].message.content)
```
DMR’s API reference includes equivalent OpenAI SDK examples and notes that an API key is not needed. citeturn16view0turn16view1  

**Option that is the most “platform-like”: Compose models (preferred for teams)**

Compose models let you declare models as first-class dependencies and get endpoint injection automatically.

Docker’s docs describe:
- `models:` top-level definition and binding to services. citeturn15view1  
- Auto-injected environment variables like `LLM_URL` and `LLM_MODEL` (or custom variable names via long syntax). citeturn15view1turn15view0  
- Model settings like `context_size` and raw `runtime_flags`. citeturn15view1turn15view0  

Example Compose file:
```yaml
services:
  app:
    image: my-app
    models:
      llm:
        endpoint_var: AI_MODEL_URL
        model_var: AI_MODEL_NAME
    environment:
      # Optional: you can still pass additional defaults here
      OPENAI_API_KEY: "not-needed"

models:
  llm:
    model: ai/smollm2
    context_size: 4096
    runtime_flags:
      - "--temp"
      - "0.7"
      - "--top-p"
      - "0.9"
```
This pattern is documented as a way for the platform (DMR) to pull/run the model and inject endpoint URLs into the service. citeturn15view1turn15view0  

Note the prerequisites: Docker’s “Use AI models in Compose” page states it requires Docker Compose 2.38+ and a platform that supports Compose models such as DMR. citeturn15view1  

### Performance and tuning layer

Make tuning a first-class part of your platform, because it directly affects latency, throughput, and memory.

Key knobs Docker documents:

- Default context sizing:
  - llama.cpp defaults to 4096 tokens. citeturn15view0  
  - Context increases can quickly increase memory use; Docker’s configuration page gives a rough rule of thumb: each additional 1,000 tokens may require ~100–500 MB of extra memory depending on model size. citeturn15view0  
- GPU offload for llama.cpp:
  - The docs list flags like `--n-gpu-layers` and note the default is “All (if GPU available),” and recommend reducing layers if you run out of VRAM. citeturn15view0  
- Sampling and batch parameters:
  - `--temp`, `--top-p`, `--batch-size`, thread controls, etc. citeturn15view0  

Platform strategy:
- Define “presets” (code, chat, creative, long-context) and enforce them via Compose `runtime_flags` or `docker model configure`. Docker’s docs even provide preset examples you can adopt. citeturn15view0turn15view1  
- For embeddings models, Docker’s Compose models doc notes you must add `--embeddings` runtime flag for `/v1/embeddings` usage. citeturn15view1  

### Observability and developer debugging

Use DMR’s built-in tooling as your first observability layer:

- The “Get started” doc describes Docker Desktop’s ability to display logs and inspect requests/responses, including prompt, token usage, and timing. citeturn1view2  
- The DMR API includes native endpoints for listing models (`GET /models`), pulling (`POST /models/create`), and deleting—useful for platform-side health checks and inventory. citeturn14view0  

Recommended platform additions:
- Add a thin “gateway” that logs request IDs and attaches metadata (caller service name, stack name, user).
- Add rate limiting per caller if you will share one Mac across multiple services/agents.

Those are architectural recommendations layered on top of DMR’s documented endpoints. citeturn14view0turn1view2

## Risks, security boundaries, and realistic expectations

### The core limitation to accept

You cannot currently make a Linux container on macOS directly use Metal GPU “like NVIDIA containers” do with CUDA, because Metal GPU passthrough is not available to containers. citeturn4view0turn5view0

Your platform can still feel “NVIDIA-like” in developer experience if you standardize on “GPU inference as a service” reachable from any container.

### Host-executed inference changes your threat model

Docker’s own design write-up acknowledges the compromise: because GPU passthrough across Docker Desktop’s VM boundary isn’t practical, inference runs outside the VM and API calls are proxied from the VM to the host. They characterize the risk as roughly on par with allowing access to host-side services via `host.docker.internal`. citeturn5view0

Also note: Docker’s design write-up says the special `model-runner.docker.internal` endpoint is accessible from containers, but “not currently from Enhanced Container Isolation (ECI) containers.” If you depend on ECI for security hardening, you need to plan for that constraint. citeturn5view0

### Backend choice tradeoffs on Mac

Docker’s vllm-metal announcement includes a benchmark where llama.cpp is reported ~1.2× faster than vLLM-Metal on their tested setup and model/quantization pairing, and it emphasizes that quantization methods differ so comparisons cover “full stack” rather than engine alone. Treat this as directional, not universal. citeturn4view1turn4view0

If you use upstream vllm-metal directly, it has significant performance-related toggles (paged attention, memory fraction controls) that can change behavior materially, but paged attention is marked experimental and may have “rough edges” for some models. citeturn17view0

### Optional research path: experimental GPU bridging into guests

If your end goal is literal GPU access *inside* a guest/container environment, there are experimental research paths (e.g., virtio-gpu + Vulkan command serialization + translation layers) that aim to build a transport for GPU workloads between guest and host without passthrough. This is complex and not the approach DMR takes, but it exists as a research direction. citeturn3search20