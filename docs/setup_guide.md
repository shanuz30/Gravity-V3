# Gravity-V3: Sovereign Agent Setup Guide
*Replication Protocol for the Antigravity Architecture*

---

## 1. Prerequisites
To run a Tier-3 Sovereign Agent, the host machine must support local execution of structural and semantic memory engines.

- **OS:** Windows 10/11 with **WSL2** installed.
- **Docker Desktop:** Required for Memgraph (Graph) and Qdrant (Vector).
- **Runtimes:** 
    - **Python 3.10+** (for Mechanical Gate & Arbitrator)
    - **Node.js 18+** (for TypeScript ports & MCP servers)

---

## 2. Infrastructure Deployment
1. **Clone the Repository:**
   ```powershell
   git clone https://github.com/shanuz30/Gravity-V3.git
   cd Gravity-V3
   ```

2. **Spin up the Memory Engines:**
   ```powershell
   docker-compose up -d
   ```
   *Note: This starts Memgraph on port 7687 and Qdrant on port 6333.*

3. **Verify Health:**
   Check if the engines are alive:
   - Memgraph: `http://localhost:3000` (Lab UI)
   - Qdrant: `http://localhost:6333/dashboard`

---

## 3. Registering the Sovereign Gate
The "Mechanical Gate" prevents industrial hallucinations by intercepting LLM commands before they hit the shell.

1. **Python Gate Setup:**
   ```powershell
   pip install -r scripts/requirements.txt
   ```

2. **TypeScript Gate Setup:**
   ```powershell
   cd scripts
   npm install
   npx tsc mechanical_gate.ts
   ```

---

## 4. MCP Configuration
To allow your AI agent to "see" and "steer" using this architecture, update your `mcp_config.json`:

```json
{
  "mcpServers": {
    "lx-dig": {
      "serverUrl": "http://localhost:9000"
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

---

## 5. Verification: The Sovereignty Test
Run the following command to ensure the Mechanical Gate is active:

```powershell
python scripts/mechanical_gate.py --intent "DROP TABLE users;"
```

**Expected Result:** 
`[BLOCK] Potential destructive command detected: DROP TABLE`

If the gate blocks the command, your agent is now **Sovereign**. It can no longer be "tricked" into destructive behavior by its own probabilistic reasoning.
