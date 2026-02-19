"""Extract batch results from MCP tool output file and process into CSVs."""
import json
import sys
import glob
import os

def extract_from_tool_output(filepath):
    """Parse the MCP tool output JSON and extract match data."""
    with open(filepath) as f:
        raw = json.load(f)
    text = raw[0]['text']
    prefix = "### Result\n"
    json_str = text[len(prefix):].split('\n')[0]
    return json.loads(json.loads(json_str))

def find_latest_tool_output():
    """Find the most recent browser_run_code output file that hasn't been processed yet."""
    pattern = os.path.expanduser(
        "~/.claude/projects/-Users-mateuszdelpercio-Code-Python-CSeSportPredictor/*/tool-results/mcp-plugin_playwright_playwright-browser_run_code-*.txt"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    # Track last processed file to avoid reprocessing
    tracker = 'data/_last_processed.txt'
    last_processed = None
    if os.path.exists(tracker):
        with open(tracker) as f:
            last_processed = f.read().strip()
    latest = max(files, key=os.path.getmtime)
    if latest == last_processed:
        return None  # already processed this one
    return latest

def mark_processed(filepath):
    """Record that this file has been processed."""
    with open('data/_last_processed.txt', 'w') as f:
        f.write(filepath)

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else find_latest_tool_output()
    if not filepath:
        print("No new tool output file found (already processed or none exist)")
        sys.exit(1)

    data = extract_from_tool_output(filepath)
    print(f"Extracted: {data['count']} matches, {data['errors']} errors")

    with open('data/batch_temp.json', 'w') as f:
        json.dump(data['results'], f)

    mark_processed(filepath)

    # Now process
    import process_batch
    process_batch.process_batch('data/batch_temp.json')
