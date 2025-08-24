# Thunder Compute (TNR) Connection Guide for LLM Agents

## Connection Methods

### 1. Quick Connect
```bash
# Connect to instance (creates SSH alias 'tnr-0')
tnr connect <instance_id>

# Use the alias for direct SSH
ssh tnr-0
```

### 2. Port Forwarding
```bash
# Forward multiple ports
tnr connect <instance_id> -t 8000 -t 8080 -t 3000
```

### 3. File Transfer (SCP)
```bash
# Upload to instance
tnr scp ./local_file.txt <instance_id>:/remote/path/

# Download from instance  
tnr scp <instance_id>:/remote/file.txt ./local_path/
```

## Executing Commands

### Direct Command Execution
```bash
# After connecting
ssh tnr-0 "command to run"
ssh tnr-0 "python train.py"
ssh tnr-0 "nvidia-smi"
```

### Interactive Session
```bash
ssh tnr-0
# Now in instance shell
cd /workspace
python run_train.py
exit  # or Ctrl+D to disconnect
```

### Background Tasks
```bash
# Use nohup for long-running tasks
ssh tnr-0 "nohup python train.py > output.log 2>&1 &"

# Or use screen/tmux
ssh tnr-0 "screen -dmS training python train.py"
ssh tnr-0 "screen -r training"  # Reattach later
```

## Instance Management

### Check Status
```bash
tnr status  # List all instances and their states
```

### Start/Stop
```bash
tnr start <instance_id>
tnr stop <instance_id>
```

## Troubleshooting

### Connection Issues
```bash
# Reset SSH known hosts (if host key changed)
mv ~/.ssh/known_hosts ~/.ssh/known_hosts.old

# Fix SSH config permissions
chmod 600 ~/.ssh/config
chown $(whoami) ~/.ssh/config

# Test manual connection for detailed errors
ssh tnr-0 -v  # Verbose output
```

### Reconnect After Disconnect
```bash
# Exit current session
# Ctrl+D or exit

# Reconnect
tnr connect <instance_id>
ssh tnr-0
```

## Quick Reference

| Task | Command |
|------|---------|
| Connect | `tnr connect <id>` |
| SSH directly | `ssh tnr-0` |
| Run command | `ssh tnr-0 "cmd"` |
| Forward port | `tnr connect <id> -t PORT` |
| Upload file | `tnr scp file <id>:/path/` |
| Download file | `tnr scp <id>:/path/file ./` |
| Check GPU | `ssh tnr-0 "nvidia-smi"` |
| Instance status | `tnr status` |

## Notes for LLM Agents
- Instance ID defaults to `0` if only one instance exists
- SSH alias `tnr-0` persists after first `tnr connect`
- Use absolute paths in SCP commands for clarity
- Port forwarding supports multiple `-t` flags
- Connection timeout is 60 seconds for SCP operations
- All SSH operations use key-based authentication (managed by TNR CLI)