Developer: Begin with a concise checklist (3–7 bullets) of the steps you will take to update the `superclaude-agent-usage-guide.md` document (attached), ensuring a structured and thorough process. Use the latest information from the SuperClaude GitHub Docs directory: https://github.com/SuperClaude-Org/SuperClaude_Framework/tree/master/Docs.

The `superclaude-agent-usage-guide.md` is the definitive reference for agents and large language models (LLMs) constructing SuperClaude command parameters from user prompts. Its primary function is to consolidate and summarize every available option within SuperClaude—excluding front-end, security, 'magic', and Playwright features—so agents and LLMs can efficiently write Claude code.

Checklist for the document update:
1. Review recent commits and release notes in the SuperClaude repo to identify all new features and changes.
2. Summarize and document each current option supported by SuperClaude commands (excluding front-end, security, 'magic', and Playwright-specific features).
3. Ensure adherence to the formatting and style of the existing documentation.
4. Explicitly state any assumptions about framework changes or documentation conventions used.
5. Present a clear, ready-to-review diff or summary of primary modifications for rapid review.
6. Perform a final validation to confirm all new features and options are covered, highlighting unaddressed gaps if complete coverage is not possible.

After each edit, validate in 1–2 lines that the guide accurately incorporates the updated features and options, and proceed or self-correct as needed. Reference content from the attached CLAUDE.MD to contextualize the repository when necessary.

Additional requirements:
- Exclude any information related to front-end, security, 'magic', or Playwright aspects.
- Maintain a concise, efficient, and current guide.
- After completion, briefly summarize (1–2 lines) the coverage of new features and note any content gaps due to unavailable information.


-------

If you can see in the task tracker document, I created and updated the document for starting the gpu instance and running training here -> docs/workflows/TRAINING_GUIDE.md - it's been validated before but I want you to conduct an in-depth validation in away that doesn't just validate what it says, but investigates the process as a whole from an outside perspective, using your own thinking logic and objectivity to verify the workflow is correct before I move to the instance, setup the environment, and conduct training. I do not want you to edit or create any files, only internalize your own steps for a sequential workflow for this task and then compare that to the TRAINING_GUIDE.md workflow/document.

--------

  /sc:analyze @docs/workflows/TRAINING_GUIDE.md \
    --focus quality \
    --depth deep \
    --think-hard \
    --seq \
    --persona-nf-validation-expert \
    --persona-risk-mitigation-specialist \
    --validate \
    --scope system



Here is an example of what I want from you in your output when I input the slash command:

User: 
/superclaude-command-opt "[ORIGINAL_PROMPT]""

AGENT (YOU):
*ROUGHLY SPEAKING (JUST AN EXAMPLE, NOT TO BE TAKEN VERBATIM)
Here's three optimal superclaude options:
/sc:analyze "[OPTIMIZED_PROMPT]*The optimized prompt should be minimally optimized and closely aligned with the user's original prompt" --flag1 --flag2 ...
/sc:...
/sc:...

Revise the command document to make this happen.

------

/sc:analyze "docs/workflows/TRAINING_GUIDE.md TASK_TRACKER.md" --focus quality --think-hard --seq --validate --systematic @agent-nf-validation-expert @agent-quality-gate-validator @agent-risk-mitigation-specialist



-----


"Read the latest couple of changes in /Users/mac-main/Neural-Forecast/TASK_TRACKER.md. I’ve created and updated the document for starting the GPU instance and running training. 

Fully read and analyze: docs/workflows/TRAINING_GUIDE.md

- it's been validated before but I want you to conduct an in-depth validation in away that doesn't just validate what it says, but investigates the process as a whole from an outside perspective, using your own thinking logic and objectivity to verify the workflow is correct before I move to the instance, setup the environment, and conduct training. I do not want you to edit or create any files, only internalize your own steps for a sequential workflow for this task and then compare that to the TRAINING_GUIDE.md workflow/document." --focus quality --think-hard --seq --validate --systematic @agent-nf-validation-expert @agent-quality-gate-validator @agent-risk-mitigation-specialist



/sc:reflect "Investigate and validate the system further, exploring avenues you may have overlooked. Re-read your global system instructions/claude.md file as they have been modified. Read sections 1-5 at .kiro/specs/proposed-spec-structure.md and read the specific line references inside docs/forecasting_sf_plan.md that the specs refer to. Examine how those core documents relate to docs/workflows/TRAINING_GUIDE.md and how setup.sh fits in. I need you to shift your thinking to a more outside the box and high-level big picture sort of thinking. I need you to take what I say and FULLY INVESTIGATE AND TRACE the system. Act as if you were autonomously being put into this instance and asked to setup the enviornment and train the models successfully. NOTE THAT THE INSTANCE IS NOT RUNNING AND WE ARE IN THE LOCAL ENVIORNMENT. This is to ensure everything is CORRECT BEFORE STARTING THE INSTANCE. But, act and imagine as if you were thrust into this situation"
