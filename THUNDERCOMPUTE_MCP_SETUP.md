# Thunder Compute MCP Documentation Server Setup

## Overview
The Thunder Compute MCP server provides easy access to Thunder Compute documentation directly within Claude Code, eliminating the need to search for context when working with your GPU instances.

## What's Been Done
✅ Added Thunder Compute MCP configuration to `.mcp.json`

## Configuration Added
```json
"thundercompute": {
  "type": "stdio",
  "command": "npx",
  "args": [
    "-y",
    "@mintlify/mcp@latest",
    "thundercompute"
  ]
}
```

## How It Works
Once Claude Code restarts and loads the new MCP configuration, you'll be able to:
1. Access Thunder Compute documentation directly
2. Get instant answers about GPU instance management
3. Find API endpoints and configuration options
4. Understand pricing and instance types

## To Activate

### Option 1: Restart Claude Code
Simply restart Claude Code and it will automatically load the new MCP server from `.mcp.json`.

### Option 2: Manual Installation (if needed)
If the automatic loading doesn't work, you can manually install:
```bash
# Create MCP directory if it doesn't exist
mkdir -p ~/.mcp

# Clone or download the Thunder Compute MCP server
npx @mintlify/mcp@latest add thundercompute

# Or run directly
node ~/.mcp/thundercompute/src/index.js
```

## Usage in Claude Code
Once active, you can ask questions like:
- "How do I launch a GPU instance on Thunder Compute?"
- "What are the available GPU types on Thunder Compute?"
- "How do I connect to my Thunder Compute instance?"
- "What's the pricing for A100 instances?"

## Benefits
- 🚀 **Instant Documentation Access**: No need to browse external docs
- 🎯 **Context-Aware Help**: Get answers specific to your use case
- ⚡ **Faster Development**: Spend less time searching, more time coding
- 📚 **Always Up-to-Date**: Documentation server updates automatically

## Integration with Neural-Forecast Project
This MCP server will be particularly useful when:
- Setting up GPU instances for training
- Configuring the A100 for optimal performance
- Understanding resource limits and quotas
- Troubleshooting connection issues
- Managing multiple training jobs

## Current MCP Servers in Project
1. **sequential-thinking**: Complex reasoning and analysis
2. **context7**: Library and framework documentation
3. **thundercompute**: Thunder Compute GPU documentation (NEW)

## Next Steps
1. Restart Claude Code to load the new configuration
2. Test by asking a Thunder Compute-related question
3. Use the documentation to optimize your GPU workflow

## Troubleshooting
If the MCP server doesn't load:
1. Check Claude Code logs for errors
2. Verify `.mcp.json` syntax is correct
3. Try manual installation as shown above
4. Ensure you have npm/npx installed and working

---
*Configuration added on: 2025-08-17*