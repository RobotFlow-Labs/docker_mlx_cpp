/**
 * Node.js example: Metal GPU from a Docker container.
 * Proves docker_mlx_cpp works with ANY language, not just Python.
 */

const MLX_URL = process.env.MLX_URL || "http://mlx-gateway:8080";

async function main() {
  console.log(`[node-client] Connecting to Metal GPU via ${MLX_URL}`);

  // GPU info
  const gpu = await fetch(`${MLX_URL}/compute/devices`).then(r => r.json());
  console.log(`  Device: ${gpu.default_device}`);
  console.log(`  Metal: ${gpu.metal_available}`);

  // Matmul on Metal GPU
  const matmul = await fetch(`${MLX_URL}/compute/eval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op: "matmul", args: { a: { shape: [1024, 1024] }, b: { shape: [1024, 1024] } } }),
  }).then(r => r.json());
  console.log(`  Matmul 1024x1024: ${matmul.elapsed_ms}ms`);

  // Chat completion
  const chat = await fetch(`${MLX_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "chat-small",
      messages: [{ role: "user", content: "Hello from Node.js inside Docker!" }],
      max_tokens: 64,
    }),
  }).then(r => r.json());

  if (chat.choices) {
    console.log(`  LLM: ${chat.choices[0].message.content}`);
  }

  console.log("\n[node-client] Metal GPU from Node.js container — working!");
}

main().catch(console.error);
