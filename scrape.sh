#!/bin/bash
# Daily CS2 scrape: launches Chrome with CDP, runs scraper, closes Chrome

PORT=9222
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CDP_PROFILE="/tmp/chrome-cdp-profile"

# Kill any leftover CDP Chrome from a previous run
if lsof -i :$PORT &>/dev/null; then
    echo "Port $PORT already in use, killing..."
    lsof -ti :$PORT | xargs kill 2>/dev/null
    sleep 1
fi

# Launch Chrome with remote debugging and separate profile
"$CHROME" --remote-debugging-port=$PORT --user-data-dir="$CDP_PROFILE" &>/dev/null &
CHROME_PID=$!

# Wait for CDP to be ready
echo "Waiting for Chrome to start..."
for i in {1..20}; do
    if curl -s "http://127.0.0.1:$PORT/json/version" &>/dev/null; then
        echo "Chrome ready on port $PORT."
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "ERROR: Chrome CDP not responding after 20s. Exiting."
        kill $CHROME_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Run scraper
echo "Running scraper..."
cd "$(dirname "$0")"
/opt/anaconda3/bin/python -m website.scrape_hltv all --port $PORT
echo "Scraper exit code: $?"

# Kill CDP Chrome
kill $CHROME_PID 2>/dev/null
echo "Done."
