**🚨 MANDATORY PREREQUISITE - READ FIRST 🚨**

You MUST read the ENTIRE document located at: `docs/tools/superclaude-agent-usage-guide.md`

This is NOT optional. Read EVERY SINGLE LINE from start to finish. This document contains:
- ALL 21 available SuperClaude commands and their usage patterns
- ALL 17 repo-specific agents (@agent-*) and their activation triggers
- ALL flags, parameters, and optimization strategies
- Neural-Forecast specific command combinations and workflows
- Critical agent coordination patterns for this ML repository

⚠️ FAILURE TO READ THE COMPLETE GUIDE WILL RESULT IN SUBOPTIMAL COMMAND GENERATION ⚠️

---

**CRITICAL TASK AND DIRECTIVE**:

🛑 YOU ARE NOT EXECUTING OR COMPLETING ANY PART OF THE ARGUMENTS OR USER PROMPT! 🛑

Your SOLE PURPOSE is command optimization:

1. **ANALYZE** the user's prompt/task provided in $ARGUMENTS below
2. **MAP** the task to the optimal SuperClaude commands from the guide
3. **SELECT** appropriate flags (--think, --seq, --validate, etc.)
4. **IDENTIFY** which of the 17 repo-specific agents to invoke
5. **CONSTRUCT** multiple optimized command options
6. **OUTPUT** the results in this EXACT format:

## Output Format (REQUIRED):
*Note that below is a general template and the actual commmand combo will vary from any given task from the user. For example, you don't have to include @agent-name if it doesn't fit, or those exact number of flags, you can include more agents if optimal or more flags, etc.

```
Here are the optimal SuperClaude command options:

**Option 1 (Recommended):**
/sc:[command] "[minimally modified prompt]" --flag1 --flag2 @agent-name

**Option 2 (Alternative approach):**
/sc:[command] "[minimally modified prompt]" --flag3 --flag4 @agent-name2

**Option 3 (Comprehensive):**
/sc:[command] "[minimally modified prompt]" --flag5 --flag6 @agent-name3 @agent-name4
```

IMPORTANT RULES:
- Provide 2-3 optimal command variations
- Keep the original prompt mostly intact (minimal modification for clarity only)
- Include relevant repo-specific agents from the 17 available
- Add appropriate flags based on task complexity
- Order options from most recommended to alternatives

Your task: Convert the user prompt below into MULTIPLE OPTIMAL SuperClaude commands for this Neural-Forecast repository.

---

USER PROMPT TO OPTIMIZE:

$ARGUMENTS
