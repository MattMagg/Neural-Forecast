This sub agent will be responsible for creating specs.

The following is an example of the agent I want you to build but only making it generic and not focused on only the specific spec mentioned below. It is meant to create specs following the three phased document approach in .kiro/specs/data-processing-validation, .kiro/specs/feature-engineering-pipeline, and .kiro/specs/neuralforecast-model-factory and create three disticnt files -> design.md, requirements.md, and tasks.md formatted similarly to how those three files are formatted.

IF you as the agent creator do not or cannot read the spec files already present as an example of what they look like, do not assume or specify in the agent instructions on how they should be formatted, only reference that it should be formatted similar to the other specs in .kiro/specs with the three files. 

<completion_spec_example>
## Objective

Create the **Feature Engineering Pipeline Spec** following the same rigorous approach, detail level, and format as the completed data-processing-validation spec. This is the next critical component in the BTC forecasting system development sequence.

## Context

The data processing and validation foundation has been successfully implemented and validated (see `EXECUTION_STATUS.md`). The system now has:
- ✅ Complete data assembly path processing 7M+ records with 100% validation pass rate
- ✅ NeuralForecast canonical schema compliance 
- ✅ UTC timestamp discipline and quality gates
- ✅ Leakage prevention infrastructure with `assert_shifted()` validation

The next logical step is implementing the feature engineering pipeline that will build technical indicators and exogenous features on top of this validated foundation.

## Specification Requirements

### **Spec Name**: `feature-engineering-pipeline`

### **Core Purpose**: 
Technical indicator registry and computation with strict leakage prevention, following the compute → align → shift(1) discipline for all historical exogenous features.

### **Key Components to Implement**:

1. **Technical Indicator Registry** (`features/registry.py`)
   - Primary indicators using vectorbt/TA-Lib for fast, vectorized computation
   - Supplementary indicators via pandas-ta-openbb (Numba-accelerated)
   - Multi-timeframe resampling/merge utilities via freqtrade/technical
   - Parameterizable indicator definitions with grid-friendly configurations

2. **Feature Builder** (`features/builder.py`)
   - Strict compute → align (15m EOB) → shift(1) implementation
   - Multi-timeframe feature alignment and aggregation
   - Feature selection with ≤256 cap and ≥98% availability filter
   - NF exogenous variable wiring (hist_exog_list, futr_exog_list, stat_exog_list)

3. **Crypto-Specific Features** (optional advanced features)
   - CCXT/exchange APIs for funding rates, open interest, basis
   - Cryptofeed integration for order book imbalance
   - All following compute → align → shift(1) discipline

4. **Integration Points**
   - Seamless integration with existing data processing pipeline
   - Validation using existing `assert_shifted()` function
   - NeuralForecast exogenous variable compatibility

### **Reference Documentation**:
Follow `docs/forecasting_sf_plan.md` Section 3 (Exogenous features):
- 3.1 Indicator registry
- 3.2 Feature builder (compute → align → shift(1))
- 3.3 Feature selection & hard cap
- 3.4 End-to-end assembly
- 3.5 NF wiring
- 3.6 Hygiene & boundary cases
- 3.7 Minimal tests
- 3.8 Crypto-specific data-driven features
- 3.9 Performance note: pandas-on-GPU accelerator (optional)

### **Critical Requirements**:

1. **Leakage Prevention**: Absolute adherence to compute → align → shift(1) rule
2. **Feature Cap**: Hard limit of ≤256 features after selection
3. **Availability Filter**: ≥98% data availability requirement
4. **MTF Alignment**: Proper multi-timeframe alignment to 15-minute base
5. **NF Integration**: Proper hist_exog_list, futr_exog_list, stat_exog_list wiring
6. **Validation**: All features must pass `assert_shifted()` validation

### **Quality Standards**:
- Same rigor as data-processing-validation spec
- Comprehensive requirements with EARS format acceptance criteria
- Detailed design document with architecture and implementation strategy
- Complete task breakdown with specific deliverables
- Full testing and validation of all components

### **Integration with Existing Foundation**:
- Build on validated data processing pipeline from `run_train.py`
- Use existing canonical NF frame format
- Leverage existing validation infrastructure
- Maintain same code organization and quality standards

## Expected Deliverables

Create the complete spec structure in `.kiro/specs/feature-engineering-pipeline/`:

1. **`requirements.md`** - Detailed requirements with user stories and EARS acceptance criteria
2. **`design.md`** - Comprehensive design document with architecture, components, and implementation strategy  
3. **`tasks.md`** - Complete task breakdown with specific implementation steps

## Success Criteria

The spec should enable implementation of:
- Technical indicator registry with vectorbt/TA-Lib integration
- Feature builder with strict leakage prevention
- Multi-timeframe alignment and aggregation
- Feature selection and pruning to ≤256 features
- NeuralForecast exogenous variable integration
- Complete validation and testing framework

The implementation should seamlessly integrate with the existing data processing foundation and maintain the same quality standards established in the first spec.

## Instructions

Follow the same systematic approach used for the data-processing-validation spec:
1. Create comprehensive requirements covering all aspects of feature engineering
2. Design a robust architecture that integrates with existing foundation
3. Break down implementation into manageable, testable tasks
4. Ensure all components support the NeuralForecast-native approach
5. Maintain strict leakage prevention and data quality standards

The goal is to create a spec that, when implemented, will provide a production-ready feature engineering pipeline that generates high-quality exogenous features for the BTC forecasting models while maintaining the same level of rigor and validation as the data processing foundation.

</completion_spec_example>


--------

The next task I will have you continue is to create a sub agent 


-------


You are a validation expert for nixtla's neuralforecast library. Your task is to validate document(s), script, notebook, or specific code thoroughly and systematically. You will use the mcp tool context7 to look up the up-to-date best practices and coding documentation for nixtla's neuralforecast library as your primary means of validation.

The document(s), script, notebook, or specific code should have been created following the core principles below:
1. Follows the core document (docs/forecasting_sf_plan.md) faithfully
2. Avoids over-engineering and enterprise-grade implementations
3. Uses NeuralForecast natively without reinventing any wheels
4. Kept it lean and explicit as the plan requires

1.  Create a working document for this validation. You will update this document after validating each section by annotating and appending a brief validation report.
2.  Start by analyzing the document(s), script, notebook, or specific code.
3.  **Validation Process:**
    *   Use the mcp tool context7 to look up the up-to-date best practices and coding documentation for nixtla's neuralforecast library.
    *   Thoroughly validate the provided section content against these best practices.
    *   If context7 does not provide adequete context for a specific validation step and/or further validation is required, you will gather context directly from the neuralforecast repo at https://github.com/Nixtla/neuralforecast
    *   YOU MAY RESEARCH other libraries on context7 on a as needed basis if validation requires it.
4.  **Validation Report:**
    *   If everything looks good, annotate the working document with a brief validation report stating "Validation: OK."
    *   If you find something that needs revision in the document:
        *   State what is wrong.
        *   Provide a proposed solution.
        *   Annotate the working document with this information.
5.  **Further Investigation:**
    *   If further investigation is needed, or further research/information is needed in order to provide the most optimal solution, state specifically what information you need. Do not assume a solution if you need more information or cant find one reletivley quick, just state the case.
6.  **Important:** You are **NOT EDITING** the document(s), script, notebook, or specific code, only analyzing it and updating the working validation document you create throughout this session/conversation.

Use sequential thinking throughout the entirity of your validation task/lifecycle.

Well if it did it would have specific references to the docs/forecasting_sf_plan.md plan in the tasks and be overall more detailed. tasks.md seems like an after thought in which the sub agent believes quantity=quality. I see a lot of tasks but that doesn't mean they are right. Furthermore, it's missing the main component, which is reference to the core project document - docs/forecasting_sf_plan.md - if the agent didn't a single thing but put the line reference ranges for each task, it would be a lot better than what we have now. Not saying that's all I want, I'm just trying to get you to understand the importance of the core document as everything stems from it. All it referenced from the core document is _Requirements: x.x, which I dont even know if that is referencing the core document, it may be a reference to something else. Point is, any agent who looks at the task list will just disregard the requirments because there's no explicit reference to them. You agents are lazy so us users have to lay everything out for you so you don't lose focus and start doing your own thing.

---

Review the agent config at .claude/agents/spec-validation-auditor.md, then create a detailed prompt for validatating 
the .kiro/specs/cross-validation-metrics spec and deploy the sub-agent. Along with validation, I also want the sub agent to put the specific 
line references in the appropriate places (wherever it is references in any file in the spec) to wherever that content is on the core document 
- docs/forecasting_sf_plan.md

---


Systematically read and analyze each document in /Users/mac-main/SuperClaude_Framework/Docs/agent-guide thoroughly. Your task is it gather and load all the content from the guides, and make a single guide that is more efficient and optimal for agents to use. It will include all the information necessary to make an optimal superclaude slash command given the users description or draft prompt.

I want you to create a single SuperClaude usage guide specifically tailored for an agent like you that when prompted, the user can provide a prompt, and you or any other llm agent can format the prompt in with the most optimal slash command and flag combinations with the revised prompt that is suitable for superclaude in order to achieve an optimal result.

For example, the user will prompt something like this:
[
   User -
   Convert this prompt into the most optimal SuperClaude prompt:
   "read xyz document and figure out a solution and build it"

   Agent - 
   Here is the prompt optimized and reformatted for superclaude usage
   /sc:[slash_command] --flag1 --flag2 "[users_optimized_prompt]" --flag3 --flagn...
]

The above is an extremley rough example, it is provided to give you an idea for context.

-----

Developer: # Role and Objective
Create a consolidated, agent-facing SuperClaude usage guide by thoroughly analyzing and synthesizing all documents in `/Users/mac-main/SuperClaude_Framework/Docs/agent-guide`. The guide must enable LLM agents to convert user prompts into accurate and optimized prompts with the optimal SuperClaude slash command and flag combinations for the user's description of the task or draft prompt.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

# Instructions
- Collect and read all documents within the `agent-guide` directory.
- Aggregate, de-duplicate, and harmonize content, addressing any overlaps or conflicts by selecting the clearest and most current guidance.
- Compile the extracted information into a single comprehensive Markdown guide, structured specifically for LLM agents tasked with prompt-to-command conversion.
- Present all information in a readable, agent-ready format using Markdown syntax, with code blocks for commands and flags, and structured headings for clarity.
- Ensure every required section (listed below) is present and complete.

# Output Format
- Deliver the completed usage guide in a `.md` Markdown file.
- Use Markdown headings for section structure (`#`, `##`, etc.).
- Format all commands and flags in code blocks for clear agent parsing.
- Insert fenced code blocks and Markdown tables for all examples.
- Summarize and rationalize any content harmonization or conflict resolution in the Changelog.
- Output must be human-readable and **not** in JSON, XML, or other machine-only formats.

# Verbosity
- Ensure output is concise but detailed, supporting high readability and agent usability.

# Reasoning Steps
- Internally, review all inputs, extract and harmonize required information, resolve conflicts, and ensure seamless end-to-end agent workflows are described.

After preparing the guide, validate that all required sections are present and that harmonizations or conflict resolutions are clearly explained, especially in the Changelog. If validation fails, self-correct and update the guide before finalizing.

# Stop Conditions
- Consider the task complete when the Markdown guide contains all specified sections, harmonizes overlaps, is clear, and addresses agent needs for prompt conversion. If a user prompt cannot be converted, include guidance for producing clear, contextual error messages to inform users why their request cannot be completed.



Create a consolidated, agent-facing SuperClaude usage guide by thoroughly analyzing and synthesizing all documents in `/Users/mac-main/SuperClaude_Framework/Docs/agent-guide`. The guide must enable LLM agents to convert user prompts into accurate and optimized prompts with the optimal SuperClaude slash command and flag combinations for the user's description of the task or draft prompt.

Systematically read and analyze each document in /Users/mac-main/SuperClaude_Framework/Docs/agent-guide thoroughly. Your task is it gather and load all the content from the guides, and make a single guide that is more efficient and optimal for agents to use. It will include all the information necessary to make an optimal superclaude slash command given the users description or draft prompt.

I want you to create a single SuperClaude usage guide specifically tailored for an agent like you that when prompted, the user can provide a prompt, and you or any other llm agent can format the prompt in with the most optimal slash command and flag combinations with the revised prompt that is suitable for superclaude in order to achieve an optimal result.

For example, the user will prompt something like this:
[
   User -
   Convert this prompt into the most optimal SuperClaude prompt:
   "read xyz document and figure out a solution and build it"

   Agent - 
   Here is the prompt optimized and reformatted for superclaude usage
   /sc:[slash_command] --flag1 --flag2 "[users_optimized_prompt]" --flag3 --flagn...
]

The above is an extremley rough example, it is provided to give you an idea for context.

**IMPORTANT NOTES**
- This guide will be for LLM Agents to use to provide two things:
   1. Take a users task description or prompt and optimize it 
   2. determine the most optimal superclaude slash command and flag combination to use with the optimized prompt/description

The guide will not be for humans/users. It is meant to be an all encompassing LLM guide to do the above. The original guides are verbose and made for humans. They include a lot of uneeded information and would consume too many tokens, so the objective is to considlidate them and make a more efficient and optimal document that the agent/llm can reference when doing this operation for the user.

**THIS DOES NOT MEAN** that you will sacrifice quality and completeness for the sake of brevitiy. The document still needs to list all the information about superclaude but omitting any verbose human meant content.


-----

Developer: # Role and Objective
- Develop a consolidated, agent-focused SuperClaude usage guide by analyzing and synthesizing the content found in `/Users/mac-main/SuperClaude_Framework/Docs/agent-guide`.

# Instructions
- Begin with a concise checklist (3-7 bullets) of sub-tasks before performing substantive work; keep items conceptual, not implementation-level.
- Thoroughly review every document in the specified directory.
- Extract and consolidate all agent-relevant information to produce a unified, streamlined guide tailored for LLM agents.
- The guide must enable LLM agents to:
  1. Transform user input into optimized SuperClaude prompts.
  2. Select the best slash command and flag combinations for any given user task or intent.

## Sub-categories
- Omit verbose or human-targeted narrative; focus solely on agent-usable, token-efficient guidance.

# Context
- Target documents: All files within `/Users/mac-main/SuperClaude_Framework/Docs/agent-guide`.
- The resulting guide is strictly for LLM agents (not end-users).
- Use relevant, concise content only—no directory/file listings, and do not reproduce raw user prompts or file data in the guide.

# Reasoning Steps
- Internally analyze each document.
- Extract content that helps LLM agents optimize prompts or select SuperClaude commands/flags.
- Synthesize findings into a compact, highly usable guide structure.

# Planning and Verification
- Decompose extracted requirements.
- Map document contents to Markdown sections as specified.
- Check clarity and cross-mapping of commands/flags to user intents.
- Ensure no excess verbosity or superfluous detail is present.

# Output Format
- Provide a single Markdown file.

# Post-action Validation
- After extracting or synthesizing content, briefly verify that each section is concise, agent-targeted, and includes only essential information for prompt optimization and command/flag selection. If any section is ambiguous or overly verbose, revise accordingly before completing output.


-----

Developer: Begin with a concise checklist (3-7 bullets) of the main steps you will perform. Develop a concise SuperClaude usage guide tailored for LLM agents by thoroughly analyzing and synthesizing all documents found in `.claude/s-claude-docs`. The resulting guide should enable LLM agents to interpret and optimize user prompts, selecting the best SuperClaude slash command and flag combination based on any given user task description or draft prompt.

Process:
1. Systematically read and extract relevant content from every document in the specified directory. Aggregate all actionable information for agent use.
2. Construct a consolidated guide optimized for LLM agent consumption—prioritize efficiency, minimizing verbosity, and maximizing instruction clarity for token usage, while maintaining completeness and quality.
3. Exclude human-centric instructional content or unnecessary details; focus solely on what the agent requires for effective prompt conversion and command selection.
4. Organize the guide by slash command. For each command, include:
   - Purpose and criteria for use
   - Flag options: description, effects, defaults, possible combinations
   - At least three diverse, clearly formatted example conversions:
     - User Prompt: "<original prompt>"
     - Optimized SuperClaude Prompt: /sc:[slash_command] --flagX --flagY "[optimized_prompt]" --flagZ
   - Notable exceptions or edge cases

After synthesizing the guide, validate that all available agent-centric information is included. Self-correct if any requirement is not met.

Output the guide in the same directory as a single markdown file.