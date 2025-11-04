# Sample RAG Query Examples

This directory contains example queries demonstrating how to use the RAG system with realistic scenarios.

## Traffic Simulator Design Scenario

A realistic example of someone designing a traffic simulation system, with:
- **Chat History**: Conversation about the project goals and technical considerations
- **Scratchpad**: In-progress stakeholder analysis showing the designer's thinking
- **Query**: Asking for help thinking through stakeholders

### Files

- `sample_rag_query.json` - Complete query in JSON format
- `test_sample_query.py` - Python script to test the scenario
- `sample_curl_command.sh` - Bash/curl command example
- `sample_powershell_command.ps1` - PowerShell command example

## Usage

### Python Script

```bash
python examples/test_sample_query.py
```

### Direct API Call (PowerShell)

```powershell
# Copy and run the contents of sample_powershell_command.ps1
# Or use Invoke-RestMethod with the JSON from sample_rag_query.json
```

### Direct API Call (curl/Bash)

```bash
bash examples/sample_curl_command.sh
```

## What This Demonstrates

This example shows how the RAG system can:
1. Use chat history to understand the design context
2. Incorporate scratchpad notes (stakeholder analysis in progress)
3. Generate a relevant nudge prompt that guides the designer's thinking
4. Reference relevant knowledge chunks from processed documents

The generated nudge should be something like:
- "Who might have a say over this project, or an opinion about its shape, that is not a direct user per se?"
- "Are any goals by different stakeholders in opposition? If so, how will you resolve them?"
- "Have you considered the alignment between how people might expect to use the app and how your design supports that?"

## Customizing

You can modify `sample_rag_query.json` to test different scenarios:
- Change the query to focus on different aspects
- Modify chat history to simulate different conversation paths
- Update scratchpad to reflect different design stages

