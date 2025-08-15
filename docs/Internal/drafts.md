Create an execution plan to finish the neuralforecast-model-factory. Read and analyze the complete spec by viewing all documents in .kiro/specs/neuralforecast-model-factory thoroughly. The create a plan to finish the remaining tasks. The plan you create will be executed using the most optimal sub agents, running in parallel IF AND ONLY IF the tasks support it. For example, you would run a validation agent in parallel with a direct implementation agent, but you can run multiple direct implementation agents if they have clear seperation of concerns and duties. You will find a list of sub agent in .claude/agents and your main project-level instructions @ CLAUDE.md provide you with guideance on how to use these sub agents so be sure to read it. When you deploy sub agents using task tool invocation, they will have detailed context and instructions in thier prompt, enough for the sub agents to work autonoumsly, following the core principles outlined in this workspace as well as referencing and following thier assigned task and the main core documentation docs/forecasting_sf_plan.md. They will also make use of sequential thinking throughout thier deployment and use context7 if and only if they need document reference or troubleshooting/debugging.


Also see below for further directives. 

## 🚨 CRITICAL: CONCURRENT EXECUTION FOR ALL ACTIONS

**ABSOLUTE RULE**: ALL operations MUST be concurrent/parallel in a single message:

### 🔴 MANDATORY CONCURRENT PATTERNS:
1. **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
2. **Task tool**: ALWAYS spawn ALL agents in ONE message with full instructions
3. **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
4. **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
5. **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**Examples of CORRECT concurrent execution:**
```javascript
// ✅ CORRECT: Everything in ONE message
[Single Message]:
  - TodoWrite { todos: [10+ todos with all statuses/priorities] }
  - Task("Agent 1 with full instructions and hooks")
  - Task("Agent 2 with full instructions and hooks")
  - Task("Agent 3 with full instructions and hooks")
  - Read("file1.js")
  - Read("file2.js")
  - Write("output1.js", content)
  - Write("output2.js", content)
  - Bash("npm install")
  - Bash("npm test")
  - Bash("npm run build")
```

**Examples of WRONG sequential execution:**
```javascript
// ❌ WRONG: Multiple messages (NEVER DO THIS)
Message 1: TodoWrite { todos: [single todo] }
Message 2: Task("Agent 1")
Message 3: Task("Agent 2")
Message 4: Read("file1.js")
Message 5: Write("output1.js")
Message 6: Bash("npm install")
// This is 6x slower and breaks coordination!
```

### 🎯 CONCURRENT EXECUTION CHECKLIST:

Before sending ANY message, ask yourself:
- ✅ Are ALL related TodoWrite operations batched together?
- ✅ Are ALL Task spawning operations in ONE message?
- ✅ Are ALL file operations (Read/Write/Edit) batched together?
- ✅ Are ALL bash commands grouped in ONE message?
- ✅ Are ALL memory operations concurrent?

If ANY answer is "No", you MUST combine operations into a single message!



-------

I would like this in jupyter notebooks instead of python script files. I would also like to note that that should not run full training and        │
│   implementations and this is the only time you will instruct the agents to use not the full sample size on the data....This is because I/we are     │
│   building this system on a mac with the following specs:   Chip:    Apple M4 Pro                                                                    │
│     Total Number of Cores:    12 (8 performance and 4 efficiency)                                                                                    │
│     Memory:    24 GB ---   Type:    GPU                                                                                                              │
│     Bus:    Built-In                                                                                                                                 │
│     Total Number of Cores:    16 ----- although this system is powerful, I have a a100 instance that I plan on running the actual full testing,      │
│   training, and production on once the foundation is built. This is to save costs.